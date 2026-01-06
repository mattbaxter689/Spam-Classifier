import torch
from torch import nn
import pytorch_lightning as pl
from transformers import AutoModel
from pytorch_lightning.utilities.types import STEP_OUTPUT


class SpamClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        lr: float = 2e-5,
        dropout: float = 0.2,
        freeze_encoder: bool = True,
    ) -> None:

        super().__init__()
        self.save_hyperparameters()
        self.encoder = AutoModel.from_pretrained(model_name)

        self.lr = lr

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]

        return self.classifier(cls).squeeze(-1)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:

        logits = self(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        loss = self.loss_fn(logits, batch["labels"])
        self.log("Train loss", loss)
        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        logits = self(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        loss = self.loss_fn(logits, batch["labels"])
        self.log("Validation loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(trainable_params, lr=self.lr)

        return optimizer
