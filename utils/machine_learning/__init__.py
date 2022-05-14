from abc import ABC, abstractmethod
import numpy as np


class AbstractTransformer(ABC):
    """Abstract class implementing fit, transform and fit_transform methods
    """
    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def transform(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit_transform(self, *args, **kwargs):
        pass


def one_hot_encoder(Y: np.ndarray) -> np.ndarray:
    """One-hot encoder

    Args:
        Y (np.ndarray): Data to encode with shape (n_samples, n_classes)

    Returns:
        np.ndarray: encoded data with shape (n_samples,)
    """
    y = list()
    n_classes = len(np.unique(Y))

    for val in Y:
        new_y_value = np.zeros(n_classes)
        new_y_value[val] = 1
        y.append(new_y_value)

    return np.array(y)


def one_hot_decoder(y: np.array) -> np.array:
    """One-hot decoder

    Args:
        Y (np.ndarray): Data to decode with shape (n_samples,)

    Returns:
        np.ndarray: decoded data with shape (n_samples, n_classes)
    """
    y_decoded = list()
    for val in y:
        y_decoded.append(np.where(val == val.max())[0][0])

    return np.array(y_decoded)
