import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall
from spam.data_configs.data_config import (
    ModelHyperParams,
    ValidationMetrics,
    TestMetrics,
)
from dataclasses import asdict


class SpamClassifier(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        params: ModelHyperParams,
        freeze_encoder: bool = True,
    ) -> None:

        super().__init__()
        self.save_hyperparameters()
        self._encoder = encoder
        self.lr = params.lr
        self.threshold = params.threshold

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        hidden = encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(params.dropout), nn.Linear(hidden, 1)
        )
        # We could also add a weight associated with this, but I will leave it for now
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Add in the metrics. We want separate Validation and Test metrics
        self.val_acc = BinaryAccuracy(threshold=self.threshold)
        self.val_prec = BinaryPrecision(threshold=self.threshold)
        self.val_rec = BinaryRecall(threshold=self.threshold)
        self.test_acc = BinaryAccuracy(threshold=self.threshold)
        self.test_prec = BinaryPrecision(threshold=self.threshold)
        self.test_rec = BinaryRecall(threshold=self.threshold)

    def set_threshold(self, threshold: float) -> None:
        self.threshold = threshold
        for metric in [
            self.val_acc,
            self.val_prec,
            self.val_rec,
            self.test_acc,
            self.test_prec,
            self.test_rec,
        ]:
            metric.threshold = threshold

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        out = self._encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]

        return self.classifier(cls).squeeze(-1)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:

        logits = self(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        loss = self.loss_fn(logits, batch["labels"].float())
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        logits = self(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        loss = self.loss_fn(logits, batch["labels"].float())
        probs = torch.sigmoid(logits)

        self.val_acc.update(probs, batch["labels"])
        self.val_prec.update(probs, batch["labels"])
        self.val_rec.update(probs, batch["labels"])
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        metrics = ValidationMetrics(
            val_accuracy=self.val_acc.compute(),
            val_precision=self.val_prec.compute(),
            val_recall=self.val_rec.compute(),
        )
        self.log_dict(asdict(metrics), prog_bar=True)
        self.val_acc.reset()
        self.val_prec.reset()
        self.val_acc.reset()

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        logits = self(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        loss = self.loss_fn(logits, batch["labels"].float())
        probs = torch.sigmoid(logits)

        self.test_acc.update(probs, batch["labels"])
        self.test_rec.update(probs, batch["labels"])
        self.test_prec.update(probs, batch["labels"])

        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def on_test_epoch_end(self) -> None:
        metrics = TestMetrics(
            test_accuracy=self.test_acc.compute(),
            test_precision=self.test_prec.compute(),
            test_recall=self.test_rec.compute(),
        )
        self.log_dict(asdict(metrics), prog_bar=True)
        self.test_acc.reset()
        self.test_prec.reset()
        self.test_acc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
        )

    @property
    def encoder(self) -> nn.Module:
        return self._encoder
