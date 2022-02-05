from abc import abstractmethod, ABC
from collections import UserList
from typing import Iterable, Union, Optional, NoReturn, List, Dict, Tuple
import numpy as np
from mne import EpochsArray
from sklearn.model_selection import train_test_split


class AbstractCombiner(ABC):
    @property
    @abstractmethod
    def X(self):
        pass

    @property
    @abstractmethod
    def Y(self):
        pass

    @abstractmethod
    def combine(self, *args, **kwargs):
        pass


class EpochsCombiner(AbstractCombiner, UserList):
    def __init__(self, *args: EpochsArray) -> NoReturn:
        super().__init__()
        data = list(args)
        if len(data) < 2:
            raise AttributeError('At least 2 EpochsArrays must be given')
        self.__storage: Dict[str, Optional[List[EpochsArray]]] = {
            'original': data,
            'filtered': None,
            'cropped': None
        }
        self.data: list[EpochsArray] = self.__storage['original']
        self._X: Optional[np.ndarray] = None
        self._Y: Optional[np.ndarray] = None

    def __str__(self) -> str:
        return f'EpochsCombiner of {len(self.data)} EpochsArrays which contain: ' \
               f'{", ".join([str(len(epochs)) + " Epochs_old" for epochs in self.data])}\n' \
               f'{super(EpochsCombiner, self).__str__()}'

    @property
    def X(self) -> np.ndarray:
        return self._X

    @X.setter
    def X(self, value: np.ndarray) -> NoReturn:
        raise AttributeError('Can not set X. It must be computed from the given EpochsArrays')

    @property
    def Y(self) -> np.ndarray:
        return self._Y

    @Y.setter
    def Y(self, value: np.ndarray) -> NoReturn:
        raise AttributeError('Can not set Y. It must be computed from the given EpochsArrays')

    @property
    def original_data(self) -> List[EpochsArray]:
        return self.__storage['original']

    @original_data.setter
    def original_data(self, value: List[EpochsArray]) -> NoReturn:
        raise AttributeError('EpochsCombiner data can not be set. It can be changed only via new EpochsCombiner object')

    @property
    def filtered_data(self) -> List[EpochsArray]:
        return self.__storage['filtered']

    @filtered_data.setter
    def filtered_data(self, value: List[EpochsArray]) -> NoReturn:
        raise AttributeError('Filtered data can not be set. It must be computed from the given EpochsArrays')

    @property
    def cropped_data(self) -> List[EpochsArray]:
        return self.__storage['cropped']

    @cropped_data.setter
    def cropped_data(self, value: List[EpochsArray]) -> NoReturn:
        raise AttributeError('Cropped data can not be set. It must be computed from the given EpochsArrays')

    def switch_data(self, data_to_use: str):
        if data_to_use not in self.__storage.keys():
            raise ValueError(f'Wrong switch option: {data_to_use}. '
                             f'Possible data to switch: {", ".join([key for key in self.__storage.keys()])}.')
        self.data = self.__storage[data_to_use]
        return self

    def filter(self, l_freq: float, h_freq: float,  *args, **kwargs):
        self.__storage['filtered'] = [
            epochs_array.copy().filter(l_freq, h_freq, *args, **kwargs)
            for epochs_array in self.data
        ]
        self.data = self.__storage['filtered']
        return self

    def crop(self, tmin: float, tmax: float, *args, **kwargs):
        self.__storage['cropped'] = [
            epochs_array.copy().crop(tmin, tmax, *args, **kwargs)
            for epochs_array in self.data
        ]
        self.data = self.__storage['cropped']
        return self

    def combine(
            self,
            *args: Union[int, Tuple[int, ...]],
            shuffle: Optional[bool] = False
    ) -> NoReturn:

        def format_indices(indices: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:

            def check_range(index):
                if index > len(self.data):
                    raise ValueError(f'Indices out of range: '
                                     f'this EpochsCombiner contains only {len(self.data)} EpochsArrays, '
                                     f'but index {index} is given')

            if isinstance(indices, int):
                check_range(indices)
                return indices,

            elif isinstance(indices, Iterable):
                for index in indices:
                    check_range(index)
                return tuple(indices)

            else:
                raise ValueError(f'Class indices must be integers of any iterable of integers, '
                                 f'{type(indices)} is given instead')

        args = [format_indices(indices) for indices in args]
        
        all_class_data = list()
        all_class_labels = list()
        for i, class_indices in enumerate(args):
            class_data = np.concatenate(
                [self.data[i].get_data() for i in class_indices],
                axis=0
            )
            all_class_data.append(class_data)
            all_class_labels.append(np.array([
                i for _ in range(class_data.shape[0])
            ]))
        
        X = np.array(all_class_data)
        Y = np.concatenate(all_class_labels)

        if shuffle:
            p = np.random.permutation(Y.shape[0])
            self._X, self._Y = X[p, :, :], Y[p]
        else:
            self._X, self._Y = X, Y

        return self

    def train_test_split(self, *args, shuffle: Optional[bool] = False, **kwargs):

        X = self.X
        Y = self.Y

        if shuffle:
            p = np.random.permutation(Y.shape[0])
            X, Y = X[p, :, :], Y[p]

        return train_test_split(X, Y, *args, **kwargs)
