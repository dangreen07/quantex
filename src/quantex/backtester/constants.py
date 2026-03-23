from enum import Enum

from ..broker.types import CommissionType  # noqa: F401 Re-export for backward compatibility


class DataSplitMode(Enum):
    """Enumeration for data split modes in optimization."""
    TRAIN = "train"
    VALIDATE = "validate"
    TEST = "test"