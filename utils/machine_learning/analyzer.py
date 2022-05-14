import tensorflow as tf
import copy
from collections import namedtuple
from utils.structures import NumberedDict
import numpy as np
from typing import Union, Optional
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import mne
import matplotlib as mpl


LayerContent = namedtuple('LayerContent', 'name data weights biases shape original_shape')


class ModelAnalyzer(object):
    """Class for analyzing tensorflow-based neural networks

    Args:
        model (tf.keras.Model): Model to analyze

    Raises:
        ValueError: If layer has unexpected number of weights
    """
    def __init__(self, model: tf.keras.Model):
        self._model = model
        self._layers = NumberedDict()

        for layer in model.layers:

            if len(layer.weights):

                if len(layer.weights) <= 3:
                    w = layer.weights[0].numpy()
                    w_shape = w.shape

                    # deleting empty dimensions, e.g. (1, m, n, 1) -> (m, n)
                    shape_sorted = dict(
                        sorted(
                            {i: shape for i, shape in enumerate(w.shape)}.items(),
                            key=lambda item: item[1] == 1,
                            reverse=True
                        )
                    )
                    reshape_order = list(shape_sorted.keys())
                    empty_shapes_count = list(shape_sorted.values()).count(1)
                    w = np.transpose(w, reshape_order)

                    if empty_shapes_count:

                        for _ in range(empty_shapes_count):
                            w = w[0]

                    # convert biases to the form of a one-dimensional matrix
                    # (to protect the summation of weights and biases), e.g. (n,) -> (1, n)
                    b = layer.weights[1].numpy()
                    b_shape = b.shape

                    # each layer has name, data (weights + biases), weights, biases,
                    # current shape of weights and biases, initial shape of weights and biases

                    if len(layer.weights) == 2:
                        b = b.reshape(1, -1)
                        self._layers.append(
                            layer.name,
                            LayerContent(
                                layer.name,
                                w + b,
                                w,
                                b,
                                (
                                    w.shape,
                                    b.shape
                                ),
                                (
                                    w_shape,
                                    b_shape
                                )
                            )
                        )
                    else:
                        print(
                            layer.weights[0].numpy().shape,
                            layer.weights[1].numpy().shape,
                            layer.weights[2].numpy().shape
                        )
                        self._layers.append(
                            layer.name,
                            LayerContent(
                                layer.name,
                                (
                                    w,
                                    b,
                                    layer.weights[2].numpy()
                                ),
                                w,
                                b,
                                (
                                    w.shape,
                                    b.shape
                                ),
                                (
                                    w_shape,
                                    b_shape
                                )
                            )
                        )
                else:
                    raise ValueError(
                        f'The layer {layer.name} has '
                        f'unexpected number of trainable variables: {len(layer.weights)}'
                    )

    @property
    def layers(self):
        """Data of layers containing names, weights, biases andshapes

        Returns:
            :obj:`list` of `LayerContent`: _description_

        Note:
            See `LayerContent` namedtuple for details of layers structure
        """
        return self._layers

    @layers.setter
    def layers(self, _):
        raise AttributeError('Can not set layers content directly')

    def plot_metrics(
        self,
        metrics_names: Union[str, list[str], tuple[str, ...]],
        title: Optional[str] = '',
        xlabel: Optional[str] = '',
        ylabel: Optional[str] = '',
        colormap: Optional[mc.Colormap] = plt.cm.Set3,
        colormap_size: Optional[int] = 11,
        inverse_colormap: Optional[bool] = True,
        legend: Optional[bool] = True,
        show: Optional[bool] = True,
    ) -> mpl.figure.Figure:
        """Plot metrics from a history of a model

        Args:
            metrics_names (:obj:`str` or :obj:`list` of :obj:`str` or :obj:`tuple` of :obj:`str`):
                metrics existing in `model.history.history` to plot
            title (str, optional): Title of the resulting figure. Defaults to ''.
            xlabel (str, optional): Label for x-axis. Defaults to ''.
            ylabel (str, optional): Label for y-axis. Defaults to ''.
            colormap (`matplotlib.colors.Colormap`, optional): Colormap to figure.
                Defaults to `matplotlib.pyplot.cm.Set3`.
            colormap_size (int, optional): Amount of colors to use. Defaults to 11.
            inverse_colormap (bool, optional): If True, invert colormap. Defaults to True.
            legend (bool, optional): If True, show legend. Defaults to True.
            show (bool, optional): If True, show figure. Defaults to True.

        Raises:
            KeyError: If wrong metric name is given

        Returns:
            matplotlib.figure.Figure: The resulting figure
        """
        if isinstance(metrics_names, str):
            metrics_names = metrics_names,
        try:
            for i, metric_name in enumerate(metrics_names):

                if inverse_colormap:
                    color_index = colormap_size - i % colormap_size
                else:
                    color_index = i % colormap_size
                plt.plot(self._model.history.history[metric_name], color=colormap(color_index))

            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

            if legend:
                plt.legend(metrics_names)

            if show:
                plt.show()

            return plt.gcf()

        except KeyError:
            raise KeyError(
                'Available metrics are: '
                f'{self.get_available_metrics()}, '
                f'but {metrics_names} were given'
            )

    @staticmethod
    def __check_given_figsize(figsize: Union[int, tuple[int, int]]) -> tuple[int, int]:

        if isinstance(figsize, int):
            return figsize, figsize
        elif not isinstance(figsize, (tuple, list)):
            raise ValueError(
                'Size of the figure can be set either '
                'by a number or by a tuple of two numbers'
            )
        return figsize

    def plot_1d_weights(
        self,
        layer_identifier: Union[int, str],
        figsize: Optional[Union[int, tuple[int, int]]] = (8, 8),
        title: Optional[str] = '',
        xlabel: Optional[str] = '',
        ylabel: Optional[str] = '',
        color: Optional[str] = '#6fa8dc',
        fmt: Optional[str] = 'x',
        linewidth: Optional[float] = 1,
        transpose: Optional[bool] = False,
        show: Optional[bool] = True,
        **kwargs
    ) -> mpl.figure.Figure:
        """Plots 1-dimensional weigths of layer

        Args:
            layer_identifier (:obj:`int` or :obj:`str`): An index or a name of a layer to visualize
            figsize (:obj:`int` or :obj:`tuple` of :obj:`int` abd `int`, optional):
                Size of a figure. If only one number is given, length is equal to width.
                Defaults to (8, 8).
            title (str, optional): Title of a figure. Defaults to ''.
            xlabel (str, optional): Label of x-axis. Defaults to ''.
            ylabel (str, optional): Label of y-axis. Defaults to ''.
            color (str, optional): Color of a figure content. Defaults to '#6fa8dc'.
            fmt (str, optional): Style of a figure content. Defaults to 'x'.
            linewidth (float, optional): Width of lines. Defaults to 1.
            transpose (bool, optional): If True, transpose image. Defaults to False.
            show (bool, optional): If True, show figure. Defaults to True.
            **kwargs: Keyword arguments for matplotlib.pyplot.plot() function

        Raises:
            ValueError: If the given layer has more that one dimension

        Returns:
            matplotlib.figure.Figure: Resulting figure
        """
        plt.figure(figsize=self.__check_given_figsize(figsize), dpi=100)
        _1d_weights = self.layers[layer_identifier].data
        if _1d_weights.ndim > 2:
            raise ValueError(
                'Dimensionality of the layer must be '
                f'lesser than 2 but it has {_1d_weights.ndim} dimensions'
            )
        elif _1d_weights.shape[0] == 1:
            _1d_weights = _1d_weights[0]

        if transpose:
            _1d_weights = _1d_weights.T
            xlabel, ylabel = ylabel, xlabel

        plt.plot(_1d_weights, fmt, color=color, linewidth=linewidth, **kwargs)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if show:
            plt.show()

        return plt.gcf()

    def plot_2d_weights(
        self,
        layer_identifier: Union[int, str],
        figsize: Optional[Union[int, tuple[int, int]]] = (8, 8),
        title: Optional[str] = '',
        xlabel: Optional[str] = '',
        ylabel: Optional[str] = '',
        colormap: Optional[mc.Colormap] = plt.cm.Reds,
        aspect: Optional[Union[float, str]] = 'auto',
        transpose: Optional[bool] = False,
        colorbar: Optional[bool] = True,
        show: Optional[bool] = True,
        **kwargs
    ):
        """Plots 2-dimensional weigths of layer

        Args:
            layer_identifier (:obj:`int` or :obj:`str`): An index or a name of a layer to visualize
            figsize (:obj:`int` or :obj:`tuple` of :obj:`int` abd `int`, optional):
                Size of a figure. If only one number is given, length is equal to width.
                Defaults to (8, 8).
            title (str, optional): Title of a figure. Defaults to ''.
            xlabel (str, optional): Label of x-axis. Defaults to ''.
            ylabel (str, optional): Label of y-axis. Defaults to ''.
            colormap (matplotlib.colors.Colormap, optional):
                Colormap of a figure content. Defaults to matplotlib.pyplot.cm.Reds.
            aspect (:obj:`float` or :obj:`str`, optional): Length-to-width ratio
                for a figure content. Defaults to 'auto'.
            transpose (bool, optional): If True, transpose image. Defaults to False.
            colorbar (bool, optional): Show colorbar. Defaults to True.
            show (bool, optional): If True, show figure. Defaults to True.
            **kwargs: Keyword arguments for matplotlib.pyplot.imshow() function

        Raises:
            ValueError: Dimensionality of the given layer is not equal to 2

        Returns:
            matplotlib.figure.Figure: Resulting figure
        """

        plt.figure(figsize=self.__check_given_figsize(figsize), dpi=100)
        _2d_weights = self.layers[layer_identifier].data

        if _2d_weights.ndim != 2:
            raise ValueError(
                'Dimensionality of the layer must '
                f'be equal 2 but it has {_2d_weights.ndim} dimensions'
            )

        if transpose:
            _2d_weights = _2d_weights.T
            xlabel, ylabel = ylabel, xlabel

        plt.imshow(_2d_weights, cmap=colormap, aspect=aspect, **kwargs)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if colorbar:
            plt.colorbar(cax=plt.axes([.925, .125, 0.075, .7555]))

        if show:
            plt.show()

        return plt.gcf()


class LFCNNAnalyzer(ModelAnalyzer):
    def __init__(self, model: tf.keras.Model):
        super().__init__(model)

    def plot_spatial_weights(
        self,
        figsize: Optional[Union[int, tuple[int, int]]] = (8, 8),
        title: Optional[str] = 'Spatial Weights',
        xlabel: Optional[str] = 'Latent Sources',
        ylabel: Optional[str] = 'Channels',
        colormap: Optional[mc.Colormap] = plt.cm.Reds,
        aspect: Optional[Union[float, str]] = 'auto',
        transpose: Optional[bool] = True,
        colorbar: Optional[bool] = True,
        show: Optional[bool] = True,
        **kwargs
    ):
        return self.plot_2d_weights(
            'spatial_filters_layer',
            figsize,
            title,
            xlabel,
            ylabel,
            colormap,
            aspect,
            transpose,
            colorbar,
            show,
            **kwargs
        )

    def plot_patterns(
        self,
        info: mne.Info,
        vmin: Optional[Union[int, float]] = None,
        vmax: Optional[Union[int, float]] = None,
        cmap: Optional[str] = 'RdBu_r',
        sensors: Optional[bool] = True,
        colorbar: Optional[bool] = False,
        res: Optional[int] = 64,
        size: Optional[int] = 1,
        cbar_fmt: Optional[str] = '%3.1f',
        name_format: Optional[str] = 'Latent\nSource %01d',
        show: Optional[bool] = True,
        show_names: Optional[bool] = False,
        title: Optional[str] = None,
        outlines: Optional[bool] = 'head',
        contours: Optional[int] = 6,
        image_interp: Optional[str] = 'bilinear',
        scalings: Optional[Union[dict[str, float], str]] = None
    ):

        if not title:
            title = 'Extracted Patterns'

        info = copy.deepcopy(info)
        info['sfreq'] = 1.
        patterns = mne.EvokedArray(self.layers['spatial_filters_layer'].data, info, tmin=0)

        return patterns.plot_topomap(
            times=range(self.layers['spatial_filters_layer'].data.shape[1]),
            vmin=vmin, vmax=vmax,
            cmap=cmap, colorbar=colorbar, res=res,
            cbar_fmt=cbar_fmt, sensors=sensors, units=None, time_unit='s',
            time_format=name_format, size=size, show_names=show_names,
            title=title, outlines=outlines,
            contours=contours, image_interp=image_interp,
            show=show, scalings=scalings
        )

    def plot_temporal_weights(
        self,
        figsize: Optional[Union[int, tuple[int, int]]] = (8, 8),
        title: Optional[str] = 'Temporal Weights',
        xlabel: Optional[str] = 'Latent Sources',
        ylabel: Optional[str] = 'Timepoints',
        colormap: Optional[mc.Colormap] = plt.cm.Reds,
        aspect: Optional[Union[float, str]] = 'auto',
        transpose: Optional[bool] = False,
        colorbar: Optional[bool] = True,
        show: Optional[bool] = True,
        **kwargs
    ):
        return self.plot_2d_weights(
            'temporal_filters_layer',
            figsize,
            title,
            xlabel,
            ylabel,
            colormap,
            aspect,
            transpose,
            colorbar,
            show,
            **kwargs
        )
