import optuna
import torch
import mlflow
import mlflow.pytorch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from sklearn.metrics import precision_recall_curve
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from optuna.integration import PyTorchLightningPruningCallback
from spam.modeling.model import SpamClassifier
from optuna.trial import FrozenTrial

MIN_PRECISION = 0.995


class TrainingManager:
    def __init__(
        self,
        train_data: Dataset,
        val_data: Dataset,
        experiment_name: str = "spam-classifier",
    ):
        self.train_data = train_data
        self.val_data = val_data
        self.experiment_name = experiment_name

    def _objective(self, trial: optuna.Trial) -> float:
        params = {
            "lr": trial.suggest_float("lr", 1e-5, 3e-5, log=True),
            "dropout": trial.suggest_float("dropout", 0.05, 0.2),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
        }

        model = SpamClassifier(
            lr=params["lr"], dropout=params["dropout"], freeze_encoder=True
        )

        logger = MLFlowLogger(
            experiment_name=self.experiment_name, tracking_uri=mlflow.get_tracking_uri()
        )

        trainer = pl.Trainer(
            max_epochs=20,
            accelerator="auto",
            devices="auto",
            logger=logger,
            callbacks=[
                EarlyStopping("val_loss", patience=3),
                PyTorchLightningPruningCallback(trial, monitor="val_loss"),
            ],
            enable_progress_bar=True,
        )

        trainer.fit(
            model,
            DataLoader(self.train_data, batch_size=params["batch_size"], shuffle=True),
            DataLoader(self.val_data, batch_size=params["batch_size"]),
        )

        # Evaluate precision-recall
        probs, labels = [], []
        model.eval()
        with torch.no_grad():
            for batch in DataLoader(self.val_data, batch_size=params["batch_size"]):
                logits = model(**batch)
                probs.extend(torch.sigmoid(logits).cpu().numpy())
                labels.extend(batch["labels"].cpu().numpy())

        precision, recall, thresholds = precision_recall_curve(labels, probs)
        for p, r, t in zip(precision, recall, thresholds):
            if p >= MIN_PRECISION:
                trial.set_user_attr("threshold", float(t))
                return float(r)

        return 0.0

    def tune(self, n_trials: int = 3) -> None:
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective, n_trials=n_trials)
        self.best_trial: FrozenTrial = study.best_trial

    def train_final(self):
        params = self.best_trial.params

        combined_ds = ConcatDataset([self.train_data, self.val_data])
        model = SpamClassifier(**params, freeze_encoder=True)

        trainer = pl.Trainer(
            max_epochs=3,
            accelerator="auto",
            devices="auto",
            logger=MLFlowLogger(
                experiment_name=self.experiment_name,
                tracking_uri=mlflow.get_tracking_uri(),
            ),
            callbacks=[EarlyStopping(monitor="train_loss", patience=3)],
        )

        trainer.fit(
            model,
            DataLoader(combined_ds, batch_size=params["batch_size"], shuffle=True),
        )

        with mlflow.start_run(run_name="final_model"):
            mlflow.log_params(params)
            mlflow.log_param("threshold", self.best_trial.user_attrs["threshold"])
            mlflow.pytorch.log_model(
                model,
                artifact_path="model",
                registered_model_name="spam_classifier",
            )
