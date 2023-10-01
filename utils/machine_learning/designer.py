from typing import Union, Callable, Optional, Any
import tensorflow as tf
import numpy as np
from utils.structures import Pipeline, Deploy
from utils.data_management import convert_base
from copy import deepcopy
from abc import ABC, abstractmethod


class AbstractDesign(ABC):
    """
    Abstract base class representing a design flow for building neural network models.

    This abstract class defines the basic structure and methods for creating a design flow for building
    neural network models. Subclasses should implement the `run_flow` property to specify the sequence of
    layers or transformations in the design.

    Attributes:
        run_flow: An abstract property representing the sequence of layers or transformations in the design.
    """
    def __iter__(self):
        """
        Allows iterating through the design flow.
        """
        return iter(self.run_flow)

    @property
    @abstractmethod
    def run_flow(self):
        """
        Abstract property representing the sequence of layers or transformations in the design.
        """
        pass

    def copy(self):
        """
        Create a deep copy of the design.

        Returns:
            AbstractDesign: A deep copy of the design.
        """
        copied = deepcopy(self)
        run_flow = list(copied.run_flow)
        for i, member in enumerate(copied):
            if issubclass(type(run_flow[i]), tf.keras.layers.Layer):
                run_flow[i]._name += f'_{convert_base(hash(id(copied._run_flow[i])), 36)}'
            elif hasattr(member, 'copy'):
                run_flow[i] = run_flow[i].copy()
            elif hasattr(member, '_name'):
                run_flow[i]._name += f'_{convert_base(hash(id(copied._run_flow[i])), 36)}'

        copied._run_flow = tuple(run_flow)

        return copied


class ModelDesign(AbstractDesign, Pipeline):
    """
    Class representing a neural network model design.

    This class extends the AbstractDesign class and provides functionality for defining neural network model designs.

    Args:
        *args (Union[Callable, Deploy, tf.Tensor, tf.keras.layers.Layer]): Variable-length arguments representing
            the components of the model design.

    Attributes:
        run_flow: A property for accessing the sequence of layers or transformations in the model design.
    """

    def __init__(self, *args: Union[Callable, Deploy, tf.Tensor, tf.keras.layers.Layer]):
        """
        Initialize a ModelDesign instance.

        Args:
            *args (Union[Callable, Deploy, tf.Tensor, tf.keras.layers.Layer]): Variable-length arguments representing
                the components of the model design. The first argument is expected to be the input tensor or shape,
                and the remaining arguments represent the layers or transformations in the design.
        """
        self._inputs = args[0]
        super().__init__(*args[1:])

    def __call__(
        self,
        inputs: Optional[Union[np.ndarray, tf.Tensor]] = None,
        kwargs: Optional[Union[dict[str, Any], tuple[dict[str, Any]]]] = None
    ) -> tf.Tensor:
        """
        Execute the model design by applying it to input data.

        Args:
            inputs (Optional[Union[np.ndarray, tf.Tensor]]): Input data for the model.
            kwargs (Optional[Union[dict[str, Any], tuple[dict[str, Any]]]]): Additional keyword arguments.

        Returns:
            tf.Tensor: Output tensor produced by the model design.
        """

        if inputs is None:
            inputs = self._inputs

        return super().__call__(inputs, kwargs=kwargs)

    @property
    def run_flow(self):
        """
        A property for accessing the sequence of layers or transformations in the model design.

        Returns:
            Tuple[Union[Callable, Deploy, tf.Tensor, tf.keras.layers.Layer]]: The sequence of layers or transformations
                in the model design.
        """
        return self._run_flow

    @run_flow.setter
    def run_flow(self, value):
        """
        Setter for the run_flow property.

        Args:
            value: The value to set (not used).

        Raises:
            AttributeError: This property is read-only and cannot be set.
        """
        raise AttributeError('Impossible to set new design')

    def build(self, **kwargs):
        """
        Build and compile a Keras model based on the design.

        Args:
            **kwargs: Additional keyword arguments for model compilation.

        Returns:
            tf.keras.Model: The compiled Keras model.
        """
        return tf.keras.Model(inputs=self._inputs, outputs=self(), **kwargs)

    def copy(self):
        """
        Create a deep copy of the model design.

        Returns:
            ModelDesign: A deep copy of the model design.
        """
        copied = super().copy()

        if hasattr(copied._inputs, '_name'):
            copied._inputs._name += f'_{convert_base(hash(id(copied._inputs)), 36)}'

        return copied


class ParallelDesign(AbstractDesign):
    """
    Class representing a parallel neural network model design.

    This class allows you to create a design that combines multiple parallel neural network designs or layers
    with an optional activation function.

    Args:
        *args (Union[None, ModelDesign, tf.keras.layers.Layer]): Variable-length arguments representing
            the parallel neural network designs or layers.
        activation (Union[str, tf.keras.layers.Activation]): Activation function to apply to the parallel outputs.

    Attributes:
        run_flow: A property for accessing the sequence of parallel neural network designs and activation.
    """
    def __init__(
        self,
        *args: Union[None, ModelDesign, tf.keras.layers.Layer],
        activation: Union[str, tf.keras.layers.Activation] = None
    ):
        """
        Initialize a ParallelDesign instance.

        Args:
            *args (Union[None, ModelDesign, tf.keras.layers.Layer]): Variable-length arguments representing
                the parallel neural network designs or layers.
            activation (Union[str, tf.keras.layers.Activation]): Activation function to apply to the parallel outputs.
                If a string is provided, it will be converted to a Keras Activation layer.

        Attributes:
            _run_flow (Tuple[Union[None, ModelDesign, tf.keras.layers.Layer]]): A tuple containing the parallel neural
                network designs or layers and the optional activation function.
            _parallels (Tuple[Union[None, ModelDesign, tf.keras.layers.Layer]]): A tuple containing the parallel neural
                network designs or layers.
        """

        if isinstance(activation, str):
            self.activation = tf.keras.layers.Activation(activation)
        else:
            self.activation = activation

        self._run_flow = (
            arg for arg in args
            if arg is not None
        ) if self.activation is None\
            else (
            *(
                arg
                for arg in args
                if arg is not None
            ),
            self.activation
        )
        self._parallels = args

    def __call__(self, inputs: Optional[Union[np.ndarray, tf.Tensor]] = None) -> tf.Tensor:
        """
        Execute the parallel neural network design by applying it to input data.

        Args:
            inputs (Optional[Union[np.ndarray, tf.Tensor]]): Input data for the parallel design.

        Returns:
            tf.Tensor: Output tensor produced by the parallel design, optionally with the activation applied.
        """
        vars = [design(inputs) if design is not None else inputs for design in self._parallels]
        return self.activation(tf.keras.layers.Add()(vars))\
            if self.activation is not None\
            else tf.keras.layers.Add()(vars)

    @property
    def run_flow(self):
        """
        A property for accessing the sequence of parallel neural network designs and activation.

        Returns:
            Tuple[Union[None, ModelDesign, tf.keras.layers.Layer]]: The sequence of parallel neural network designs
                or layers and the optional activation function.
        """
        return self._run_flow

    @run_flow.setter
    def run_flow(self, value):
        """
        Setter for the run_flow property.

        Args:
            value: The value to set (not used).

        Raises:
            AttributeError: This property is read-only and cannot be set.
        """
        raise AttributeError('Impossible to set new parallel designs')

    def replace(self, design: ModelDesign, index: int):
        """
        Replace one of the parallel designs with a new design.

        Args:
            design (ModelDesign): The new parallel neural network design to replace the existing design.
            index (int): The index of the design to be replaced.

        Raises:
            ValueError: If the specified index is out of range.

        Note:
            This method allows you to replace a specific parallel design within the parallel design sequence.
        """
        parralels = list(self.run_flow)
        parralels[index] = design
        self._parralels = tuple(parralels)
        self._run_flow = self._parralels\
            if self.activation is None\
            else (*self._parralels, self.activation)


class LayerDesign(AbstractDesign, Deploy):
    """
    Class representing a layer-based neural network design.

    This class allows you to create a design that consists of a sequence of layers or transformations.

    Args:
        layer_transformator (Callable): A callable function used to initialize the design.
        *args: Variable-length arguments representing layers or transformations to include in the design.
        **kwargs: Keyword arguments representing additional layers or transformations to include in the design.

    Attributes:
        run_flow: A property for accessing the sequence of layers or transformations in the design.
    """
    def __init__(self, layer_transformator: Callable, *args, **kwargs):
        """
        Initialize a LayerDesign instance.

        Args:
            layer_transformator (Callable): A callable function used to initialize the design.
            *args: Variable-length arguments representing layers or transformations to include in the design.
            **kwargs: Keyword arguments representing additional layers or transformations to include in the design.

        Attributes:
            _run_flow (Tuple[Union[tf.keras.layers.Layer, AbstractDesign]]): A tuple containing the layers or
                transformations in the design.
        """
        super().__init__(layer_transformator, *args, **kwargs)
        self._run_flow = (
            arg for arg in [
                *args,
                *list(kwargs.values())
            ]
            if issubclass(
                type(arg),
                (
                    tf.keras.layers.Layer,
                    AbstractDesign
                )
            )
        )

    def __call__(self, input: tf.Tensor, **kwargs):
        """
        Execute the layer-based neural network design by applying it to input data.

        Args:
            input (tf.Tensor): Input tensor or data to apply the design to.
            **kwargs: Additional keyword arguments.

        Returns:
            tf.Tensor: Output tensor produced by the design.
        """
        return super().__call__(input, **kwargs)

    @property
    def run_flow(self):
        """
        A property for accessing the sequence of layers or transformations in the design.

        Returns:
            Tuple[Union[tf.keras.layers.Layer, AbstractDesign]]: The sequence of layers or transformations in the design.
        """
        return self._run_flow

    @run_flow.setter
    def run_flow(self, value):
        """
        Setter for the run_flow property.

        Args:
            value: The value to set (not used).

        Raises:
            AttributeError: This property is read-only and cannot be set.
        """
        raise AttributeError('Impossible to set run_flow')
