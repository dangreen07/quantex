from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TrainValidateTestSplit:
    """
    Container for train/validate/test data splits.
    
    This class holds the split configuration and indices for dividing
    historical data into training, validation, and test sets for
    machine learning-style optimization workflows.
    
    Attributes:
        train_start (int): Starting index for training data.
        train_end (int): Ending index for training data.
        validate_start (int): Starting index for validation data.
        validate_end (int): Ending index for validation data.
        test_start (int): Starting index for test data.
        test_end (int): Ending index for test data.
        train_ratio (float): Ratio of data used for training.
        validate_ratio (float): Ratio of data used for validation.
        test_ratio (float): Ratio of data used for testing.
    """
    train_start: int
    train_end: int
    validate_start: int
    validate_end: int
    test_start: int
    test_end: int
    train_ratio: float = 0.6
    validate_ratio: float = 0.2
    test_ratio: float = 0.2
    
    def __post_init__(self):
        """Validate split ratios sum to 1.0."""
        total = self.train_ratio + self.validate_ratio + self.test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.3f}"
            )


def create_train_validate_test_split(
    data_length: int,
    train_ratio: float = 0.6,
    validate_ratio: float = 0.2,
    test_ratio: float = 0.2
) -> TrainValidateTestSplit:
    """
    Create indices for train/validate/test split.
    
    This function divides the data indices into three sets for ML-style
    optimization: training (parameter fitting), validation (hyperparameter
    selection), and testing (final evaluation).
    
    Args:
        data_length (int): Total number of data points.
        train_ratio (float, optional): Fraction of data for training.
            Defaults to 0.6 (60%).
        validate_ratio (float, optional): Fraction of data for validation.
            Defaults to 0.2 (20%).
        test_ratio (float, optional): Fraction of data for testing.
            Defaults to 0.2 (20%).
            
    Returns:
        TrainValidateTestSplit: Object containing start/end indices for
            each split.
            
    Raises:
        ValueError: If ratios don't sum to 1.0 or are invalid.
        
    Example:
        >>> split = create_train_validate_test_split(1000, 0.6, 0.2, 0.2)
        >>> print(f"Train: {split.train_start}-{split.train_end}")
        >>> print(f"Validate: {split.validate_start}-{split.validate_end}")
        >>> print(f"Test: {split.test_start}-{split.test_end}")
    """
    if not np.isclose(train_ratio + validate_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")
    
    if train_ratio <= 0 or validate_ratio <= 0 or test_ratio <= 0:
        raise ValueError("All split ratios must be positive")
    
    train_end = int(data_length * train_ratio)
    validate_end = int(data_length * (train_ratio + validate_ratio))
    
    return TrainValidateTestSplit(
        train_start=0,
        train_end=train_end,
        validate_start=train_end,
        validate_end=validate_end,
        test_start=validate_end,
        test_end=data_length,
        train_ratio=train_ratio,
        validate_ratio=validate_ratio,
        test_ratio=test_ratio
    )
