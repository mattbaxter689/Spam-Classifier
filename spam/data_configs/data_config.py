from dataclasses import dataclass
import torch


@dataclass
class ModelHyperParams:
    lr: float
    dropout: float
    threshold: float


@dataclass
class ValidationMetrics:
    val_accuracy: torch.Tensor
    val_precision: torch.Tensor
    val_recall: torch.Tensor


@dataclass
class TestMetrics:
    test_accuracy: torch.Tensor
    test_precision: torch.Tensor
    test_recall: torch.Tensor
