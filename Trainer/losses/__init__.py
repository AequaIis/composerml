
from .linear_loss import LinearLoss
from .bce_loss import BCELoss
from .ce_loss import CrossEntropyLoss
from .loss import Loss

__all__ = [
    "Loss",
    "LinearLoss",
    "BCELoss",
    "CrossEntropyLoss",
]