from typing import Union, Callable, Optional, Any
import tensorflow as tf
import numpy as np
from utils.structures import Pipeline, Deploy
from utils.data_management import convert_base
from copy import deepcopy
from abc import ABC, abstractmethod


class AbstractDesign(ABC):

    def __iter__(self):
        return iter(self.run_flow)

    @property
    @abstractmethod
    def run_flow(self):
        pass

    def copy(self):
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

    def __init__(self, *args: Union[Callable, Deploy, tf.Tensor, tf.keras.layers.Layer]):
        self._inputs = args[0]
        super().__init__(*args[1:])

    def __call__(
        self,
        inputs: Optional[Union[np.ndarray, tf.Tensor]] = None,
        kwargs: Optional[Union[dict[str, Any], tuple[dict[str, Any]]]] = None
    ) -> tf.Tensor:

        if inputs is None:
            inputs = self._inputs

        return super().__call__(inputs, kwargs=kwargs)

    @property
    def run_flow(self):
        return self._run_flow

    @run_flow.setter
    def run_flow(self, value):
        raise AttributeError('Impossible to set new design')

    def build(self, **kwargs):
        return tf.keras.Model(inputs=self._inputs, outputs=self(), **kwargs)

    def copy(self):
        copied = super().copy()

        if hasattr(copied._inputs, '_name'):
            copied._inputs._name += f'_{convert_base(hash(id(copied._inputs)), 36)}'

        return copied


class ParallelDesign(AbstractDesign):
    def __init__(
        self,
        *args: Union[None, ModelDesign, tf.keras.layers.Layer],
        activation: Union[str, tf.keras.layers.Activation] = None
    ):

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
        vars = [design(inputs) if design is not None else inputs for design in self._parallels]
        return self.activation(tf.keras.layers.Add()(vars))\
            if self.activation is not None\
            else tf.keras.layers.Add()(vars)

    @property
    def run_flow(self):
        return self._run_flow

    @run_flow.setter
    def run_flow(self, value):
        raise AttributeError('Impossible to set new parallel designs')

    def replace(self, design: ModelDesign, index: int):
        parralels = list(self.run_flow)
        parralels[index] = design
        self._parralels = tuple(parralels)
        self._run_flow = self._parralels\
            if self.activation is None\
            else (*self._parralels, self.activation)


class LayerDesign(AbstractDesign, Deploy):
    def __init__(self, layer_transformator: Callable, *args, **kwargs):
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
        return super().__call__(input, **kwargs)

    @property
    def run_flow(self):
        return self._run_flow

    @run_flow.setter
    def run_flow(self, value):
        raise AttributeError('Impossible to set run_flow')
