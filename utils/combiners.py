from abc import abstractmethod, ABC
from collections import UserList
from typing import Iterable, Union, Optional, NoReturn, List, Dict, Tuple
import numpy as np
from mne import EpochsArray
from sklearn.model_selection import train_test_split


class AbstractCombiner(ABC):
    """
    Abstract base class for data combiners.

    Attributes:
        X: A property for the data matrix X.
        Y: A property for the label vector Y.
    """
    @property
    @abstractmethod
    def X(self):
        """
        Get the data matrix X.

        Returns:
            np.ndarray: The data matrix X.
        """
        pass

    @property
    @abstractmethod
    def Y(self):
        """
        Get the label vector Y.

        Returns:
            np.ndarray: The label vector Y.
        """
        pass

    @abstractmethod
    def combine(self, *args, **kwargs):
        """
        Combine data from multiple sources.

        Args:
            *args: Variable-length arguments for specifying data sources.
            **kwargs: Additional keyword arguments.

        Returns:
            NoReturn: This method should not return anything.
        """
        pass


class EpochsCombiner(AbstractCombiner, UserList):
    """
    Combines multiple EpochsArray objects into a single data structure.

    Attributes:
        original_data: A property for accessing the original EpochsArray objects.
        filtered_data: A property for accessing the filtered EpochsArray objects.
        cropped_data: A property for accessing the cropped EpochsArray objects.
        X: A property for the data matrix X.
        Y: A property for the label vector Y.
    """
    def __init__(self, *args: EpochsArray) -> NoReturn:
        """
        Initialize an EpochsCombiner instance.

        Args:
            *args: Variable-length arguments representing EpochsArray objects.

        Raises:
            AttributeError: If fewer than 2 EpochsArray objects are provided.
        """
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
        """
        Return a string representation of the EpochsCombiner.

        Returns:
            str: A string describing the EpochsCombiner.
        """
        return f'EpochsCombiner of {len(self.data)} EpochsArrays which contain: ' \
               f'{", ".join([str(len(epochs)) + " Epochs_old" for epochs in self.data])}\n' \
               f'{super(EpochsCombiner, self).__str__()}'

    @property
    def X(self) -> np.ndarray:
        """
        Get the data matrix X. This property cannot be set directly.

        Returns:
            np.ndarray: The data matrix X.
        """
        return self._X

    @X.setter
    def X(self, value: np.ndarray) -> NoReturn:
        """
        Set the data matrix X. It must be computed from the given EpochsArrays.

        Args:
            value: The data matrix X.

        Raises:
            AttributeError: X cannot be set directly.
        """
        raise AttributeError('Can not set X. It must be computed from the given EpochsArrays')

    @property
    def Y(self) -> np.ndarray:
        """
        Get the label vector Y. This property cannot be set directly.

        Returns:
            np.ndarray: The label vector Y.
        """
        return self._Y

    @Y.setter
    def Y(self, value: np.ndarray) -> NoReturn:
        """
        Set the label vector Y. It must be computed from the given EpochsArrays.

        Args:
            value: The label vector Y.

        Raises:
            AttributeError: Y cannot be set directly.
        """
        raise AttributeError('Can not set Y. It must be computed from the given EpochsArrays')

    @property
    def original_data(self) -> List[EpochsArray]:
        """
        Get the original EpochsArray objects.

        Returns:
            List[EpochsArray]: List of original EpochsArray objects.
        """
        return self.__storage['original']

    @original_data.setter
    def original_data(self, value: List[EpochsArray]) -> NoReturn:
        """
        Set the original data. It can be changed only via a new EpochsCombiner object.

        Args:
            value: List of EpochsArray objects.

        Raises:
            AttributeError: Original data cannot be set directly.
        """
        raise AttributeError(
            'EpochsCombiner data can not be set. '
            'It can be changed only via new EpochsCombiner object'
        )

    @property
    def filtered_data(self) -> List[EpochsArray]:
        """
        Get the filtered EpochsArray objects.

        Returns:
            List[EpochsArray]: List of filtered EpochsArray objects.
        """
        return self.__storage['filtered']

    @filtered_data.setter
    def filtered_data(self, value: List[EpochsArray]) -> NoReturn:
        """
        Set the filtered data. It must be computed from the given EpochsArrays.

        Args:
            value: List of filtered EpochsArray objects.

        Raises:
            AttributeError: Filtered data cannot be set directly.
        """
        raise AttributeError(
            'Filtered data can not be set. '
            'It must be computed from the given EpochsArrays'
        )

    @property
    def cropped_data(self) -> List[EpochsArray]:
        """
        Get the cropped EpochsArray objects.

        Returns:
            List[EpochsArray]: List of cropped EpochsArray objects.
        """
        return self.__storage['cropped']

    @cropped_data.setter
    def cropped_data(self, value: List[EpochsArray]) -> NoReturn:
        """
        Set the cropped data. It must be computed from the given EpochsArrays.

        Args:
            value: List of cropped EpochsArray objects.

        Raises:
            AttributeError: Cropped data cannot be set directly.
        """
        raise AttributeError(
            'Cropped data can not be set. '
            'It must be computed from the given EpochsArrays'
        )

    def switch_data(self, data_to_use: str):
        """
        Switch between different data views (original, filtered, or cropped).

        Args:
            data_to_use (str): The data view to switch to (e.g., 'original', 'filtered', 'cropped').

        Raises:
            ValueError: If an invalid data view is provided.
        """
        if data_to_use not in self.__storage.keys():
            raise ValueError(
                f'Wrong switch option: {data_to_use}. '
                f'Possible data to switch: {", ".join([key for key in self.__storage.keys()])}.'
            )
        self.data = self.__storage[data_to_use]
        return self

    def filter(self, l_freq: float, h_freq: float,  *args, **kwargs):
        """
        Apply filtering to the EpochsArrays in the current data view.

        Args:
            l_freq (float): The low-frequency cutoff.
            h_freq (float): The high-frequency cutoff.
            *args: Additional arguments to pass to the filter method.
            **kwargs: Additional keyword arguments to pass to the filter method.

        Returns:
            EpochsCombiner: A reference to the current EpochsCombiner object.
        """
        self.__storage['filtered'] = [
            epochs_array.copy().filter(l_freq, h_freq, *args, **kwargs)
            for epochs_array in self.data
        ]
        self.data = self.__storage['filtered']
        return self

    def crop(self, tmin: float, tmax: float, *args, **kwargs):
        """
        Apply cropping to the EpochsArrays in the current data view.

        Args:
            tmin (float): The start time for cropping.
            tmax (float): The end time for cropping.
            *args: Additional arguments to pass to the crop method.
            **kwargs: Additional keyword arguments to pass to the crop method.

        Returns:
            EpochsCombiner: A reference to the current EpochsCombiner object.
        """
        self.__storage['cropped'] = [
            epochs_array.copy().crop(tmin, tmax, *args, **kwargs)
            for epochs_array in self.data
        ]
        self.data = self.__storage['cropped']
        return self

    def shuffle(self):
        """
        Shuffle the data matrix and label vector.

        This method shuffles the data matrix (X) and label vector (Y) in-place.

        Returns:
            None
        """
        X = self.X
        Y = self.Y
        p = np.random.permutation(Y.shape[0])
        self._X, self._Y = X[p, :, :], Y[p]

    def combine(
            self,
            *args: Union[int, Tuple[int, ...]],
            shuffle: Optional[bool] = False
    ) -> NoReturn:
        """
        Combine data from multiple classes into a single dataset.

        Args:
            *args: Variable-length arguments representing class indices to combine.
            shuffle (bool, optional): Whether to shuffle the combined data. Defaults to False.

        Returns:
            EpochsCombiner: A reference to the current EpochsCombiner object.

        Raises:
            ValueError: If class indices are out of range or in an invalid format.
        """

        def format_indices(indices: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:

            def check_range(index):
                if index > len(self.data):
                    raise ValueError(
                        f'Indices out of range: '
                        f'this EpochsCombiner contains only {len(self.data)} EpochsArrays, '
                        f'but index {index} is given'
                    )

            if isinstance(indices, int):
                check_range(indices)
                return indices,

            elif isinstance(indices, Iterable):
                for index in indices:
                    check_range(index)
                return tuple(indices)

            else:
                raise ValueError(
                    f'Class indices must be integers of any iterable of integers, '
                    f'{type(indices)} is given instead'
                )

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

        X = np.concatenate(all_class_data, axis=0)
        Y = np.concatenate(all_class_labels)

        if shuffle:
            p = np.random.permutation(Y.shape[0])
            self._X, self._Y = X[p, :, :], Y[p]
        else:
            self._X, self._Y = X, Y

        return self

    def train_test_split(self, *args, shuffle: Optional[bool] = False, **kwargs):
        """
        Split the data into training and testing sets.

        Args:
            *args: Arguments to pass to the train_test_split function.
            shuffle (bool, optional): Whether to shuffle the data before splitting. Defaults to False.
            **kwargs: Keyword arguments to pass to the train_test_split function.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing X_train, X_test, Y_train, Y_test.
        """

        X = self.X
        Y = self.Y

        if shuffle:
            p = np.random.permutation(Y.shape[0])
            X, Y = X[p, :, :], Y[p]

        return train_test_split(X, Y, *args, **kwargs)
