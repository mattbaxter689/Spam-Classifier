import optuna
import torch
import mlflow
import mlflow.pytorch
import pytorch_lightning as pl
import torch.nn as nn
from dataclasses import asdict
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from sklearn.metrics import precision_recall_curve
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from optuna.integration import PyTorchLightningPruningCallback
from spam.modeling.model import SpamClassifier
from spam.data_configs.data_config import ModelHyperParams, TestMetrics
from optuna.trial import FrozenTrial
from typing import Callable


class TrainingManager:
    def __init__(
        self,
        model: Callable[[], nn.Module],
        train_data: Dataset,
        val_data: Dataset,
        test_data: Dataset,
        experiment_name: str = "spam-classifier",
    ):
        self._model_factory = model
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.experiment_name = experiment_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _objective(self, trial: optuna.Trial) -> float:
        params = ModelHyperParams(
            lr=trial.suggest_float("lr", 1e-5, 3e-5, log=True),
            dropout=trial.suggest_float("dropout", 0.05, 0.2),
            threshold=0.5,
            batch_size=trial.suggest_categorical("batch_size", [2, 4]),
        )
        encoder = self._model_factory()

        model = SpamClassifier(
            encoder=encoder,
            params=params,
            freeze_encoder=True,
        )
        logger = MLFlowLogger(
            experiment_name=self.experiment_name,
            run_name=f"trial_{trial.number}",
            tracking_uri=mlflow.get_tracking_uri(),
        )
        checkpoint_cb = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )

        trainer = pl.Trainer(
            max_epochs=5,
            accelerator="auto",
            devices="auto",
            logger=logger,
            callbacks=[
                checkpoint_cb,
                EarlyStopping("val_loss", patience=3),
                PyTorchLightningPruningCallback(trial, monitor="val_loss"),
            ],
            enable_progress_bar=True,
        )

        trainer.fit(
            model,
            DataLoader(
                self.train_data,
                batch_size=params.batch_size,
                num_workers=2,
                shuffle=True,
            ),
            DataLoader(self.val_data, batch_size=params.batch_size, num_workers=2),
        )

        best_model = SpamClassifier.load_from_checkpoint(
            checkpoint_cb.best_model_path, encoder=encoder, params=params
        )
        trainer.validate(
            model=best_model,
            dataloaders=DataLoader(
                self.val_data, batch_size=params.batch_size, num_workers=2
            ),
        )
        recall = trainer.callback_metrics["val_recall"].item()

        trial.set_user_attr("best_checkpoint", checkpoint_cb.best_model_path)
        return recall

    def evaluate(
        self, model: nn.Module, dataset: Dataset, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:

        probs, labels = [], []
        model.eval()
        with torch.no_grad():
            for batch in DataLoader(dataset, batch_size=batch_size):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                )
                probs.append(torch.sigmoid(logits).cpu())
                labels.append(batch["labels"].cpu())

        return torch.cat(probs), torch.cat(labels)

    def select_threshold_from_pr(
        self,
        y_true: torch.Tensor,
        y_prob: torch.Tensor,
        target_precision: float = 0.90,
        lambda_penalty: float = 0.5,
    ) -> float:

        precision, recall, thresholds = precision_recall_curve(
            y_true.numpy(), y_prob.numpy()
        )

        best_score = -float("inf")
        best_threshold = 0.5

        for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
            penalty = max(0.0, target_precision - p)
            score = r - lambda_penalty * penalty

            if score > best_score:
                best_score = score
                best_threshold = t

        return float(best_threshold)

    def tune(self, n_trials: int = 3) -> None:
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective, n_trials=n_trials)

        self.best_trial: FrozenTrial = study.best_trial
        self.best_params = ModelHyperParams(**self.best_trial.params)
        self.best_checkpoint = self.best_trial.user_attrs["best_checkpoint"]

    def tune_threshold(self) -> None:
        model = SpamClassifier.load_from_checkpoint(
            self.best_checkpoint, encoder=self._model_factory(), params=self.best_params
        ).to(self.device)

        probs, labels = self.evaluate(
            model, self.val_data, batch_size=self.best_params.batch_size
        )

        self.threshold = self.select_threshold_from_pr(labels, probs)

    def train_final(self) -> TestMetrics:
        assert (
            self.threshold is not None
        ), "Threshold must be tuned first. Please tune the threshold"

        self.best_params.threshold = self.threshold

        combined_ds = ConcatDataset([self.train_data, self.val_data])
        model = SpamClassifier(
            encoder=self._model_factory(),
            params=self.best_params,
            freeze_encoder=True,
        )
        checkpoint_cb = ModelCheckpoint(
            monitor="train_loss",
            mode="min",
            save_top_k=1,
        )

        trainer = pl.Trainer(
            max_epochs=3,
            accelerator="auto",
            devices="auto",
            callbacks=[EarlyStopping(monitor="train_loss", patience=3), checkpoint_cb],
        )

        trainer.fit(
            model,
            DataLoader(
                combined_ds, batch_size=self.best_params.batch_size, shuffle=True
            ),
        )
        best_model = SpamClassifier.load_from_checkpoint(
            checkpoint_cb.best_model_path,
            encoder=self._model_factory(),
            params=self.best_params,
        )
        best_model.set_threshold(self.threshold)
        trainer.save_checkpoint("spam_final_model.ckpt")

        trainer.test(
            best_model,
            dataloaders=DataLoader(
                self.test_data, batch_size=self.best_params.batch_size, num_workers=2
            ),
        )
        test_metrics = TestMetrics(
            test_accuracy=trainer.callback_metrics["test_accuracy"].item(),
            test_precision=trainer.callback_metrics["test_precision"].item(),
            test_recall=trainer.callback_metrics["test_recall"].item(),
        )

        with mlflow.start_run(run_name="final_model") as run:
            mlflow.log_metrics(asdict(test_metrics))
            mlflow.log_metrics(asdict(self.best_params))
            mlflow.pytorch.log_model(pytorch_model=best_model, artifact_path="model")
            run_uri = f"runs:/{run.info.run_id}/model"
            mlflow.register_model(run_uri, "SpamClassifier")

        return test_metrics
