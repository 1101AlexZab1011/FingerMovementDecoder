import matplotlib.pyplot as plt
import numpy as np
import mne
from LFCNN_decoder import SpatialParameters, TemporalParameters, WaveForms
from dataclasses import dataclass
import matplotlib as mp
import copy
import scipy as sp
import argparse
import os
from utils.storage_management import check_path
from typing import NoReturn, Optional, Union
import matplotlib as mpl
from matplotlib.widgets import Button
from utils.storage_management import read_pkl
import mneflow as mf


def plot_patterns(
    patterns, info, order=None, axes=None, cmap='RdBu_r', sensors=True,
    colorbar=False, res=64,
    size=1, cbar_fmt='%3.1f', name_format='Latent\nSource %01d',
    show=True, show_names=False, title=None,
    outlines='head', contours=6,
    image_interp='linear', **kwargs
) -> mpl.figure.Figure:
    """
    Plot topographic patterns of source activities.

    This function plots topographic patterns of source activities, typically obtained from source localization techniques
    in neuroimaging. It uses the MNE-Python library's `plot_topomap` method.

    Args:
        patterns (array-like): The topographic patterns of source activities to be plotted.
        info (mne.Info): The MNE-Python Info object containing information about the data.
        order (list, optional): The order in which patterns are plotted. Default is None, which uses the natural order.
        axes (mpl.axes.Axes | None, optional): The matplotlib axes to plot the patterns on. Default is None, which creates
            new axes.
        cmap (str, optional): The colormap to use for plotting the patterns. Default is 'RdBu_r'.
        sensors (bool, optional): Whether to show sensor positions on the plot. Default is True.
        colorbar (bool, optional): Whether to include a colorbar. Default is False.
        res (int, optional): The resolution of the topomap. Default is 64.
        size (float, optional): The size of the topomap. Default is 1.
        cbar_fmt (str, optional): The format string for the colorbar labels. Default is '%3.1f'.
        name_format (str, optional): The format for naming each topomap. Default is 'Latent\nSource %01d'.
        show (bool, optional): Whether to display the plot. Default is True.
        show_names (bool, optional): Whether to show channel names on the plot. Default is False.
        title (str, optional): The title of the plot. Default is 'Computed patterns'.
        outlines (str | None, optional): The outlines to use for the head. Default is 'head'.
        contours (int, optional): The number of contour lines to draw. Default is 6.
        image_interp (str, optional): The interpolation method for the topomap image. Default is 'linear'.
        **kwargs: Additional keyword arguments to pass to `plot_topomap`.

    Returns:
        mpl.figure.Figure: The matplotlib Figure object containing the topographic patterns plot.

    Note:
        - This function relies on the MNE-Python library for topomap visualization.
    """
    if order is None:
        order = range(patterns.shape[1])
    if title is None:
        title = 'Computed patterns'
    info = copy.deepcopy(info)
    info.__setstate__(dict(_unlocked=True))
    info['sfreq'] = 1.
    patterns = mne.EvokedArray(patterns, info, tmin=0)
    return patterns.plot_topomap(
        times=order,
        axes=axes,
        cmap=cmap, colorbar=colorbar, res=res,
        cbar_fmt=cbar_fmt, sensors=sensors, units=None, time_unit='s',
        time_format=name_format, size=size, show_names=show_names,
        outlines=outlines,
        contours=contours, image_interp=image_interp, show=show, **kwargs)


def plot_waveforms(model: mf.models.BaseModel, sorting='compwise_loss', tmin=0, class_names=None) -> mpl.figure.Figure:
    """
    Plot waveforms and related information for latent components of a machine learning model.

    This function plots waveforms, filter coefficients, convolution output, and feature relevance maps
    for latent components of a machine learning model. It provides insights into how each latent component
    contributes to the model's predictions.

    Args:
        model (mf.models.BaseModel): The machine learning model containing latent components to be visualized.
        sorting (str): The sorting criteria for ordering the latent components. Default is 'compwise_loss'.
        tmin (float): The starting time for plotting waveforms. Default is 0.
        class_names (list[str]): Optional list of class names for the latent components.

    Returns:
        mpl.figure.Figure: A Matplotlib figure containing the plots for latent components.

    """

    fs = model.dataset.h_params['fs']

    if not hasattr(model, 'lat_tcs'):
        model.compute_patterns(model.dataset)

    if not hasattr(model, 'uorder'):
        order, _ = model._sorting(sorting)
        model.uorder = order.ravel()

    if np.any(model.uorder):

        for jj, uo in enumerate(model.uorder):
            f, ax = plt.subplots(2, 2)
            f.set_size_inches([16, 16])
            nt = model.dataset.h_params['n_t']
            model.waveforms = np.squeeze(model.lat_tcs.reshape(
                [model.specs['n_latent'], -1, nt]
            ).mean(1))
            tstep = 1 / float(fs)
            times = tmin + tstep * np.arange(nt)
            scaling = 3 * np.mean(np.std(model.waveforms, -1))
            [
                ax[0, 0].plot(times, wf + scaling * i)
                for i, wf in enumerate(model.waveforms) if i not in model.uorder
            ]
            ax[0, 0].plot(times, model.waveforms[uo] + scaling * uo, 'k', linewidth=5.)
            ax[0, 0].set_title('Latent component waveforms')
            bias = model.tconv.b.numpy()[uo]
            ax[0, 1].stem(model.filters.T[uo], use_line_collection=True)
            ax[0, 1].hlines(bias, 0, len(model.filters.T[uo]), linestyle='--', label='Bias')
            ax[0, 1].legend()
            ax[0, 1].set_title('Filter coefficients')
            conv = np.convolve(model.filters.T[uo], model.waveforms[uo], mode='same')
            vmin = conv.min()
            vmax = conv.max()
            ax[1, 0].plot(times + 0.5 * model.specs['filter_length'] / float(fs), conv)
            tstep = float(model.specs['stride']) / fs
            strides = np.arange(times[0], times[-1] + tstep / 2, tstep)[1:-1]
            pool_bins = np.arange(times[0], times[-1] + tstep, model.specs['pooling'] / fs)[1:]
            ax[1, 0].vlines(strides, vmin, vmax, linestyle='--', color='c', label='Strides')
            ax[1, 0].vlines(pool_bins, vmin, vmax, linestyle='--', color='m', label='Pooling')
            ax[1, 0].set_xlim(times[0], times[-1])
            ax[1, 0].legend()
            ax[1, 0].set_title('Convolution output')
            strides1 = np.linspace(times[0], times[-1] + tstep / 2, model.F.shape[1])
            ax[1, 1].pcolor(strides1, np.arange(model.specs['n_latent']), model.F)
            ax[1, 1].hlines(uo, strides1[0], strides1[-1], color='r')
            ax[1, 1].set_title('Feature relevance map')

            if class_names:
                comp_name = class_names[jj]
            else:
                comp_name = "Class " + str(jj)

            f.suptitle(comp_name, fontsize=16)

        return f


def plot_spectra(
    temporal_parameters: TemporalParameters,
    order: list, title: str = '',
    xlim: tuple = None,
    ylim: tuple = None,
    legend: bool = None
) -> mpl.figure.Figure:
    """
    Plot spectral information of temporal parameters.

    This function plots spectral information of temporal parameters, typically obtained from signal processing or analysis.
    It visualizes filter input, filter output, and filter response for a set of temporal components.

    Args:
        temporal_parameters (TemporalParameters): The temporal parameters object containing spectral information.
        order (list): The order in which spectral information for components should be plotted.
        title (str, optional): The title of the plot. Default is an empty string.
        xlim (tuple, optional): The x-axis limits for the plot. Default is None.
        ylim (tuple, optional): The y-axis limits for the plot. Default is None.
        legend (list, optional): The legend labels for the plot. Default is None.

    Returns:
        mpl.figure.Figure: The matplotlib Figure object containing the spectral information plot.

    Note:
        - This function organizes the spectral information into a grid of subplots based on the number of components.
        - The `temporal_parameters` object should contain spectral information for filter input, filter output, and filter response.
    """

    if not len(order) % 3:
        n_cols = 3
        n_rows = len(order) // 3
    elif len(order) == 3:
        n_rows = 1
        n_cols = 3
    else:
        n_rows = len(order) // 3 + 1
        n_cols = 3

    if legend is None:
        legend = ['Filter input', 'Filter output', 'Filter responce']

    fig, axs = plt.subplots(n_rows, n_cols)

    if len(axs.shape) == 1:
        axs = np.reshape(axs, (1, -1))

    fig.set_size_inches(n_cols * 5, n_rows * 3.75)
    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.5)

    current_comp = 0

    for i in range(n_rows):
        for j in range(n_cols):
            if current_comp < len(order):
                n_component = order[current_comp]
                axs[i, j].set_title(f'Latent Source {n_component}')
                axs[i, j].plot(
                    temporal_parameters.franges,
                    temporal_parameters.finputs[n_component],
                    temporal_parameters.franges,
                    temporal_parameters.foutputs[n_component],
                    temporal_parameters.franges,
                    temporal_parameters.fresponces[n_component],
                )

                if xlim:
                    axs[i, j].set_xlim(xlim)

                if ylim:
                    axs[i, j].set_ylim(ylim)
            else:
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
                axs[i, j].spines['top'].set_visible(False)
                axs[i, j].spines['right'].set_visible(False)
                axs[i, j].spines['bottom'].set_visible(False)
                axs[i, j].spines['left'].set_visible(False)

            current_comp += 1

    fig.legend(legend, loc='upper right')
    return fig


def plot_tempospectral(
    spatial_parameters: Union[SpatialParameters, list[SpatialParameters]],
    temporal_parameters: Union[TemporalParameters, list[TemporalParameters]],
    orders: Union[np.ndarray, list[np.ndarray]],
    info: mne.Info,
    subject_names: Optional[Union[str, list[str]]] = None,
    class_names: Optional[Union[str, list[str]]] = None,
    title: Optional[str] = None,
    xlim: Optional[Union[int, float]] = None,
    ylim: Optional[Union[int, float]] = None,
    legend: Optional[Union[int, float]] = None,
    spatial_data_type: Optional[str] = 'patterns',
    topomap_kwargs: Optional[dict] = None
) -> mp.figure.Figure:
    """
    Plot temporal-spectral information for multiple subjects and components.

    This function generates a complex plot that combines temporal and spectral information for multiple subjects
    and components. It is especially useful for visualizing analysis results in neuroscience or signal processing.

    Args:
        spatial_parameters (Union[SpatialParameters, list[SpatialParameters]]): Spatial parameters or a list of
            spatial parameters for each subject.
        temporal_parameters (Union[TemporalParameters, list[TemporalParameters]]): Temporal parameters or a list of
            temporal parameters for each subject.
        orders (Union[np.ndarray, list[np.ndarray]]): The order in which spectral information for components should
            be plotted.
        info (mne.Info): The MNE info object containing subject information.
        subject_names (Optional[Union[str, list[str]]], optional): Names of subjects. Default is None.
        class_names (Optional[Union[str, list[str]]], optional): Names of classes or components. Default is None.
        title (Optional[str], optional): The title of the plot. Default is None.
        xlim (Optional[Union[int, float]], optional): The x-axis limits for the plot. Default is None.
        ylim (Optional[Union[int, float]], optional): The y-axis limits for the plot. Default is None.
        legend (Optional[Union[int, float]], optional): Legend labels for the plot. Default is None.
        spatial_data_type (Optional[str], optional): Type of spatial data ('patterns' or 'filters'). Default is 'patterns'.
        topomap_kwargs (Optional[dict], optional): Additional keyword arguments for topomap plotting. Default is None.

    Returns:
        mp.figure.Figure: The matplotlib Figure object containing the combined temporal-spectral information plot.
    """

    def wrap_in_list(content):
        return [content] if not isinstance(content, list) else content

    def validate_length(*iterables):
        length = len(iterables[0])
        for i, sample in enumerate(iterables[1:]):
            if len(sample) != length:
                raise ValueError(
                    'Length validation failed: all elements must have length equal '
                    f'to {length}, but element {i} has length: {len(sample)}'
                )
        return length

    spatial_parameters = wrap_in_list(spatial_parameters)
    temporal_parameters = wrap_in_list(temporal_parameters)
    info = wrap_in_list(info)

    if isinstance(orders, np.ndarray):
        n_components = len(orders)
    elif isinstance(orders[0], np.ndarray):
        n_components = len(orders[0])
    else:
        raise ValueError('"orders" must be either np.ndarray or list of np.ndarray')

    orders = wrap_in_list(orders)
    n_subjects = validate_length(spatial_parameters, temporal_parameters, info, orders)

    if subject_names is None:
        subject_names = [f'Subject {i}' for i in range(n_subjects)]
    elif isinstance(subject_names, str):
        subject_names = [f'{subject_names} {i}' for i in range(n_subjects)]
    elif isinstance(subject_names, list):

        if len(subject_names) != n_subjects:
            raise ValueError('Not all subjects have names provided')

    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_components)]
    elif isinstance(class_names, str):
        subject_names = [f'{class_names} {i}' for i in range(n_components)]
    elif isinstance(class_names, list):

        if len(class_names) != n_components:
            raise ValueError('Not all classes have names provided')

    if not n_components % 3:
        n_cols = 3
        n_rows = n_components // 3
    elif n_components == 3:
        n_rows = 1
        n_cols = 3
    elif n_components == 4:
        n_rows = 1
        n_cols = 4
    elif not n_components % 4:
        n_cols = 4
        n_rows = n_components // 4
    elif not n_components % 2:
        n_cols = 2
        n_rows = n_components // 2
    else:
        n_rows = n_components // 3 + 1
        n_cols = 3

    n_rows_per_subject = 2 * n_rows
    n_rows = n_subjects * n_rows_per_subject

    if legend is None:
        legend = ['Filter input', 'Filter output', 'Filter responce']

    fig, axs = plt.subplots(n_rows, n_cols)

    if len(axs.shape) == 1:
        axs = np.reshape(axs, (1, -1))

    fig.set_size_inches(n_cols * 5, n_rows * 3.75)

    subplots_map = np.ones((n_rows, n_cols)).astype(bool)

    current_comp = 0
    for i in range(0, n_rows, 2):
        for j in range(n_cols):

            if current_comp >= n_components:
                subplots_map[i, j], subplots_map[i + 1, j] = False, False

            current_comp += 1

        if current_comp >= n_components:
            current_comp = 0

    @dataclass
    class Tracker(object):
        top: int
        bottom: int
        subject: int

    tracker = Tracker(0, 0, 0)
    for i in range(n_rows):
        for j in range(n_cols):

            if subplots_map[i, j]:

                if i % 2 and j == 0:
                    axs[i, j].set_ylabel(subject_names[tracker.subject])
                    axs[i, j].tick_params(axis='y', pad=300)
                elif not i % 2 and j == 0:
                    axs[i, j].set_ylabel(subject_names[tracker.subject])

                if not i % 2:
                    axs[i, j].set_title(
                        'Latent Source '
                        f'{orders[tracker.subject][tracker.top]} ({class_names[tracker.top]})'
                    )
                    axs[i, j].plot(
                        temporal_parameters[tracker.subject].franges,
                        temporal_parameters[tracker.subject]
                        .finputs[orders[tracker.subject][tracker.top]],
                        temporal_parameters[tracker.subject].franges,
                        temporal_parameters[tracker.subject]
                        .foutputs[orders[tracker.subject][tracker.top]],
                        temporal_parameters[tracker.subject].franges,
                        temporal_parameters[tracker.subject]
                        .fresponces[orders[tracker.subject][tracker.top]],
                    )

                    axs[i, j].legend(legend, loc='upper right')

                    if xlim:
                        axs[i, j].set_xlim(xlim)

                    if ylim:
                        axs[i, j].set_ylim(ylim)
                    tracker.top += 1

                else:
                    subject_info = copy.deepcopy(info[tracker.subject])
                    subject_info['sfreq'] = 1.

                    if spatial_data_type == 'patterns':
                        data = spatial_parameters[tracker.subject].patterns
                    elif spatial_data_type == 'filters':
                        data = spatial_parameters[tracker.subject].filters

                    patterns = mne.EvokedArray(data, subject_info, tmin=0)

                    topomap_parameters = dict(
                        times=orders[tracker.subject][tracker.bottom],
                        time_format='',
                        cmap='RdBu_r', colorbar=False, res=64,
                        units=None, time_unit='s',
                        size=1, outlines='head', contours=6,
                        image_interp='bilinear', show=False,
                        axes=axs[i, j]
                    )

                    if topomap_kwargs is not None:
                        topomap_parameters.update(topomap_kwargs)

                    patterns.plot_topomap(**topomap_parameters)
                    tracker.bottom += 1
            else:
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
                axs[i, j].spines['top'].set_visible(False)
                axs[i, j].spines['right'].set_visible(False)
                axs[i, j].spines['bottom'].set_visible(False)
                axs[i, j].spines['left'].set_visible(False)

        if tracker.bottom >= n_components:
            assert tracker.bottom == tracker.top, 'Tracker\'s top and bottom do not match'
            tracker.top = 0
            tracker.bottom = 0
            tracker.subject += 1

    if title is not None:
        fig.suptitle(title, fontsize=20)

    return fig


def plot_spatial_weights(
    spatial_parameters: SpatialParameters,
    temporal_parameters: TemporalParameters,
    waveforms: WaveForms,
    info: mne.Info,
    summarize: Optional[Union[str, list[float]]] = 'sum',
    title: Optional[str] = 'Spatial Patterns',
    show: Optional[bool] = True,
    logscale: Optional[bool] = False
) -> Union[mp.figure.Figure, NoReturn]:
    """
    Plot spatial weights, patterns, and temporal responses.

    This function generates a complex plot that visualizes spatial patterns, spatial weights, and temporal responses
    of a given dataset. It can be used for analyzing and visualizing the results of source localization or other
    similar techniques in neuroscience.

    Args:
        spatial_parameters (SpatialParameters): Spatial parameters.
        temporal_parameters (TemporalParameters): Temporal parameters.
        waveforms (WaveForms): Waveform data.
        info (mne.Info): The MNE info object containing subject information.
        summarize (Optional[Union[str, list[float]]], optional): The method for summarizing spatial patterns.
            Options: 'sum' (sum of weights), 'sumabs' (sum of absolute values), 'abssum' (absolute sum).
            Default is 'sum'.
        title (Optional[str], optional): The title of the plot. Default is 'Spatial Patterns'.
        show (Optional[bool], optional): Whether to display the plot. Default is True.
        logscale (Optional[bool], optional): Whether to use a logarithmic scale for the y-axis. Default is False.

    Returns:
        Union[mp.figure.Figure, NoReturn]: The matplotlib Figure object containing the spatial weights plot,
        or None if `show` is set to False.
    """

    mp.use('Qt5Agg')

    def init_canvas(ax01, ax02, ax1, ax2):
        ax01.axes.xaxis.set_visible(False)
        ax01.axes.yaxis.set_visible(False)
        ax02.axes.xaxis.set_visible(False)
        ax02.axes.yaxis.set_visible(False)

        ax01.spines['right'].set_visible(False)
        ax01.spines['left'].set_visible(False)
        ax01.spines['top'].set_visible(False)
        ax01.spines['bottom'].set_visible(False)
        ax02.spines['right'].set_visible(False)
        ax02.spines['left'].set_visible(False)
        ax02.spines['top'].set_visible(False)
        ax02.spines['bottom'].set_visible(False)

        ax1.spines['right'].set_visible(False)
        ax1.set_xlabel('Channels')
        ax1.set_ylabel('Latent Sources')
        ax1.spines['left'].set_alpha(0.2)
        ax1.spines['bottom'].set_alpha(0.2)
        ax1.spines['top'].set_alpha(0.2)
        ax1.axes.yaxis.set_alpha(0.2)
        ax1.set_yticks(np.arange(y_lim))
        ax1.set_yticklabels(labels=[i + 1 for i in sorting_callback.sorted_indices])
        ax1.tick_params(axis='both', which='both', length=5, color='#00000050')

        ax2.axes.yaxis.set_visible(False)
        ax2.axes.xaxis.set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['top'].set_alpha(0.2)
        ax2.spines['right'].set_alpha(0.2)
        ax2.spines['bottom'].set_alpha(0.2)

    class SortingCallback:

        def __init__(
            self,
            button: Button,
            fig: plt.Figure,
            bar_ax: mp.axes.Axes,
            imshow_ax: mp.axes.Axes,
            indices: list[int]
        ):
            self._button = button
            self._fig = fig
            self._bar_ax = bar_ax
            self._imshow_ax = imshow_ax
            self._event = None
            self._sorted_indices = indices

        def __call__(self, event):
            self._event = event

            if '▼' in self._button.label._text:
                self.decreasing()
            else:
                self.increasing()

        @property
        def sorted_indices(self):
            return self._sorted_indices

        @sorted_indices.setter
        def sorted_indices(self, value):
            raise AttributeError('Impossible to set indices directly')

        def increasing(self):
            self._button.label.set_text('Sort ▼')
            self._sorted_indices = sorted(range(len(sums)), key=lambda k: sums[k], reverse=True)
            self.update()

        def decreasing(self):
            self._button.label.set_text('Sort ▲')
            self._sorted_indices = sorted(range(len(sums)), key=lambda k: sums[k])
            self.update()

        def update(self):
            self._imshow_ax.clear()
            self._imshow_ax.imshow(data.T[self._sorted_indices, :], aspect='auto', cmap='RdBu_r')
            self._bar_ax.clear()
            self._bar_ax.barh(
                range(len(sums)),
                np.abs(sums)[self._sorted_indices], color=colors[self._sorted_indices],
                height=.9
            )
            init_canvas(ax01, ax02, ax1, ax2)
            self._fig.canvas.draw()

    def onclick(event):
        flim = 70
        crop = .05
        shift = True

        if ax1.lines:

            for i in range(len(ax1.lines)):
                ax1.lines.remove(ax1.lines[i])

        _, iy = event.xdata, event.ydata

        if (event.inaxes == ax1 or event.inaxes == ax2) \
            and event.xdata is not None \
            and event.ydata is not None \
            and 0 < event.xdata < x_lim \
                and -.5 < event.ydata < y_lim:
            iy = int(np.rint(iy))
            induced = waveforms.induced.copy()[
                sorting_callback.sorted_indices[iy],
                :flim,
                :
            ]
            crop *= induced.shape[1] / 2

            for i, ind_course in enumerate(induced):
                induced[i] /= ind_course.mean()

            color = colors[sorting_callback._sorted_indices[iy]]
            line = mp.lines.Line2D([0, x_lim], [iy, iy], color=color, linewidth=16, alpha=.4)
            ax1.add_line(line)
            fig1.canvas.draw()
            fig2 = plt.figure(constrained_layout=False)
            gs2 = fig2.add_gridspec(
                nrows=10,
                ncols=3,
                bottom=.1,
                wspace=.05,
                hspace=.1
            )
            ax21 = fig2.add_subplot(gs2[:, :-1])
            ax22 = fig2.add_subplot(gs2[0:5, -1])
            ax23 = fig2.add_subplot(gs2[5:, -1])
            plot_patterns(
                data,
                info,
                sorting_callback.sorted_indices[iy],
                ax21,
                name_format='',
                title=''
            )
            ax22_t = ax22.twinx()
            ax22_t.plot(
                sp.stats.zscore(waveforms.evoked[sorting_callback.sorted_indices[iy]]),
                '#454545'
            )
            pos = ax22.imshow(
                induced,
                cmap='RdBu_r',
                origin='lower'
            )
            cb = fig2.colorbar(
                pos,
                ax=ax22,
                pad=0.12,
                orientation='horizontal',
                aspect=75,
                fraction=.12
            )
            ax22.set_aspect('auto')
            ax22_t.set_aspect('auto')
            # ax22_t.set_ylim(top=1, bottom=-1)
            ax23.plot(
                temporal_parameters.franges,
                sp.stats.zscore(temporal_parameters.finputs[sorting_callback.sorted_indices[iy]]),
                temporal_parameters.franges,
                sp.stats.zscore(temporal_parameters.foutputs[sorting_callback.sorted_indices[iy]]),
                temporal_parameters.franges,
                sp.stats.zscore(
                    temporal_parameters.fresponces[sorting_callback.sorted_indices[iy]]
                ),
            )

            ax22_t.set_ylabel('Amplitude', labelpad=12.5, rotation=270)
            ax22_t.spines['top'].set_alpha(.2)
            ax22_t.spines['right'].set_alpha(.2)
            ax22_t.spines['left'].set_alpha(.2)
            ax22_t.spines['bottom'].set_alpha(.2)
            ax22_t.tick_params(axis='both', which='both', length=5, color='#00000050')
            ax22.spines['top'].set_alpha(.2)
            ax22.spines['right'].set_alpha(.2)
            ax22.spines['left'].set_alpha(.2)
            ax22.spines['bottom'].set_alpha(.2)
            ax22.tick_params(axis='both', which='both', length=5, color='#00000050')
            cb.outline.set_color('#00000020')
            cb.ax.tick_params(axis='both', which='both', length=5, color='#00000050')
            times = np.unique(np.round(waveforms.times, 1))
            ranges = np.linspace(0, len(waveforms.times), len(times)).astype(int)

            if shift:
                times = np.round(times - times.mean(), 2)

            ax22.set_xticks(ranges)
            ax22.set_xticklabels(times)
            ax22.set_xlabel('Time (s)')
            ax22.set_ylabel('Frequency (Hz)')
            ax23.legend(['Filter input', 'Filter output', 'Filter responce'], loc='upper right')
            ax23.spines['top'].set_alpha(.2)
            ax23.spines['right'].set_alpha(.2)
            ax23.spines['left'].set_alpha(.2)
            ax23.spines['bottom'].set_alpha(.2)
            ax23.tick_params(axis='both', which='both', length=5, color='#00000050')
            ax23.set_xlabel('Frequency (Hz)')
            ax23.set_ylabel('Amplitude')
            # ax23.set_ylim(top=1.2)
            ax23.set_xlim([0, 70])
            ax22_t.set_xlim([2 * crop, len(waveforms.times) - 2 * crop])

            if logscale:
                ax23.set_yscale('log')

            fig2.suptitle(f'Latent source {sorting_callback.sorted_indices[iy] + 1}')
            plt.show()

    data = spatial_parameters.patterns.copy()
    x_lim, y_lim = data.shape
    fig1 = plt.figure()
    fig1.suptitle(title)
    gs = fig1.add_gridspec(2, 2, hspace=0, wspace=0, width_ratios=[1, .1], height_ratios=[.025, 1])

    (ax01, ax02), (ax1, ax2) = gs.subplots(sharex='col', sharey='row')

    if summarize == 'sum':
        sums = np.sum(data, axis=0)
    elif summarize == 'sumabs':
        sums = np.sum(np.abs(data), axis=0)
    elif summarize == 'abssum':
        sums = np.abs(np.sum(data, axis=0))
    elif isinstance(summarize, list) and len(summarize) == y_lim:
        sums = np.array(summarize)
    else:
        raise NotImplementedError(
            f'The "{summarize}" method not implemented. '
            'Available methods: "sum", "sumabs", "abssum"'
        )
    colors = np.array(['#f2827a' if sum_ >= 0 else '#8bbae5' for sum_ in sums])

    sort_button = Button(ax02, 'Sort')
    sorting_callback = SortingCallback(
        sort_button,
        fig1,
        ax2,
        ax1,
        sorted(range(len(sums)), reverse=False)
    )
    sort_button.on_clicked(sorting_callback)

    init_canvas(ax01, ax02, ax1, ax2)

    ax1.imshow(data.T, aspect='auto', cmap='RdBu_r')
    ax2.barh(sorting_callback.sorted_indices, np.abs(sums), color=colors, height=.9)

    cid1 = fig1.canvas.mpl_connect('button_press_event', onclick)
    cid2 = fig1.canvas.mpl_connect('close_event', lambda e: fig1.canvas.mpl_disconnect(cid1))

    if show:
        plt.show()

    fig1.canvas.mpl_disconnect(cid2)

    return fig1


if __name__ == '__main__':
    mpl.use('agg')
    parser = argparse.ArgumentParser(
        description='A script for applying the neural network "LFCNN" '
        'to the epoched data from gradiometers related to events for classification'
    )
    parser.add_argument('-s', '--sort', type=str,
                        default='l2', help='Method to sort components')
    parser.add_argument('-ep', '--exclude-participants', type=str, nargs='+',
                        default=[], help='IDs of subjects to exclude')
    parser.add_argument('-cmb', '--combine-cases', type=str, nargs='+',
                        default=['LI', 'LM', 'RI', 'RM'], help='Cases to consider '
                        '(must be the number of strings in which classes to combine '
                        'are written separated by a space, indices corresponds to '
                        'order of "--cases" parameter)')
    parser.add_argument('-sd', '--subjects-dir', type=str,
                        default=os.path.join(os.getcwd(), 'Source', 'Subjects'),
                        help='Path to the subjects directory')
    parser.add_argument('--name', type=str,
                        default=None, help='Name of a task')
    parser.add_argument('--postfix', type=str,
                        default='', help='String to append to a task name')
    parser.add_argument('--prefix', type=str,
                        default='', help='String to set in the start of a task name')
    parser.add_argument('--filters', help='Delete all files that are no longer needed',
                        action='store_true')

    sort, \
        excluded_subjects, \
        cases_to_combine, \
        subjects_dir, \
        classification_name,\
        classification_postfix,\
        classification_prefix, \
        filters = vars(parser.parse_args()).values()

    spatial_data_type = 'filters' if filters else 'patterns'

    if sort not in ['l2', 'compwise_loss', 'weight', 'output_corr', 'weight_corr']:
        raise ValueError(
            f'Wrong option for sorting: {sort}. '
            'Sortings can be \'l2\', \'compwise_loss\', '
            '\'weight\', \'output_corr\', \'weight_corr\''
        )

    cases_to_combine = [case.split(' ') for case in cases_to_combine]

    # cases_to_combine = sorted(cases_to_combine, reverse=True)
    class_names = [
        '&'.join(sorted(cases_combination, reverse=True)) for cases_combination in cases_to_combine
    ]
    if classification_name is None:
        classification_name = '_vs_'.join(class_names)

    classification_name_formatted = "_".join(list(filter(
        lambda s: s not in (None, ""),
        [classification_prefix, classification_name, classification_postfix]
    )))

    all_spatial_parameters = list()
    all_temporal_parameters = list()
    all_sortings = list()
    all_info = list()
    all_subjects = list()

    for subject_name in os.listdir(subjects_dir):

        if subject_name in excluded_subjects:
            continue

        all_subjects.append(subject_name)

        subject_path = os.path.join(subjects_dir, subject_name)
        parametes_path = os.path.join(subject_path, 'Parameters')
        subject_infopath = os.path.join(subject_path, 'Info')
        info = read_pkl(os.path.join(subject_infopath, os.listdir(subject_infopath)[0]))

        if not isinstance(info, mne.Info):
            if isinstance(info, list) and isinstance(info[0], mne.Info) and len(info) == 1:
                info = info[0]
            else:
                raise ValueError(f'Wrong info content:\n{info}')

        info.pick_channels(
            list(
                filter(
                    lambda ch_name: (
                        ch_name[-1] == '2'
                        or ch_name[-1] == '3'
                    ) and 'meg' in ch_name.lower(),
                    info['ch_names']
                )
            )
        )
        all_info.append(info)
        all_spatial_parameters.append(
            read_pkl(os.path.join(parametes_path, f'{classification_name_formatted}_spatial.pkl'))
        )
        all_temporal_parameters.append(
            read_pkl(os.path.join(parametes_path, f'{classification_name_formatted}_temporal.pkl'))
        )
        all_sortings.append(
            read_pkl(
                os.path.join(parametes_path, f'{classification_name_formatted}_sorting.pkl')
            )._asdict()[sort]
        )

    pics_path = os.path.join(os.path.dirname(subjects_dir), 'Pictures')
    spatiospectral_pics_path = os.path.join(pics_path, 'SpatioSpectral')
    spatispectral_case_path = os.path.join(spatiospectral_pics_path, classification_name_formatted)
    check_path(pics_path, spatiospectral_pics_path, spatispectral_case_path)
    fig = plot_tempospectral(
        all_spatial_parameters,
        all_temporal_parameters,
        all_sortings,
        all_info,
        all_subjects,
        class_names,
        spatial_data_type=spatial_data_type
    )
    fig.savefig(os.path.join(spatispectral_case_path, f'{spatial_data_type}_{sort}.png'))
