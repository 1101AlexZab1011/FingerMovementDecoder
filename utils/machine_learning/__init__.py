from abc import ABC, abstractmethod


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