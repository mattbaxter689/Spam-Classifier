from dataclasses import dataclass
import torch


@dataclass
class ModelHyperParams:
    lr: float
    dropout: float
    batch_size: int
    threshold: float | None = None


@dataclass
class ValidationMetrics:
    val_accuracy: torch.Tensor
    val_precision: torch.Tensor
    val_recall: torch.Tensor


@dataclass
class TestMetrics:
    test_accuracy: torch.Tensor | float
    test_precision: torch.Tensor | float
    test_recall: torch.Tensor | float
