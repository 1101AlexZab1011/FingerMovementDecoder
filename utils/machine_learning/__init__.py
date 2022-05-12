from abc import ABC, abstractmethod
import numpy as np


class AbstractTransformer(ABC):
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
    y = list()
    n_classes = len(np.unique(Y))

    for val in Y:
        new_y_value = np.zeros(n_classes)
        new_y_value[val] = 1
        y.append(new_y_value)

    return np.array(y)


def one_hot_decoder(y: np.array) -> np.array:
    y_decoded = list()
    for val in y:
        y_decoded.append(np.where(val == val.max())[0][0])

    return np.array(y_decoded)
