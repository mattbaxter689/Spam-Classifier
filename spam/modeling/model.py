import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel
from pytorch_lightning.utilities.types import STEP_OUTPUT


class SpamClassifier(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        lr: float = 2e-5,
        dropout: float = 0.2,
        freeze_encoder: bool = True,
    ) -> None:

        super().__init__()
        self.save_hyperparameters()
        self._encoder = encoder
        self.lr = lr

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        hidden = encoder.config.hidden_size
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.loss_fn = nn.BCEWithLogitsLoss()

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
        loss = self.loss_fn(logits, batch["labels"])
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        logits = self(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        loss = self.loss_fn(logits, batch["labels"])
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        logits = self(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        loss = self.loss_fn(logits, batch["labels"])
        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
        )

    @property
    def encoder(self) -> nn.Module:
        return self._encoder
