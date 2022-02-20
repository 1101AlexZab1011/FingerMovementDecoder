import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import matplotlib as mp
import mne
from combiners import EpochsCombiner
from typing import *
import matplotlib.pyplot as plt
import numpy as np
from utils.data_management import dict2str
from LFCNN_decoder import SpatialParameters, TemporalParameters, ComponentsOrder
from lfcnn_vis import plot_patterns
import pickle
from matplotlib.widgets import Button

def read_pkl(path: str) -> Any:
    with open(
            path,
            'rb'
        ) as file:
        content = pickle.load(
            file
        )
    return content


def plot_spatial_weights(
    spatial_parameters: SpatialParameters,
    temporal_parameters: TemporalParameters,
    info: mne.Info,
    summarize: Optional[Union[str, list[float]]] = 'sum',
    title: Optional[str] = 'Spatial Patterns',
    show: Optional[bool] = True,
    logscale: Optional[bool] = False
) -> Union[mp.figure.Figure, NoReturn]:
    
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
        ax1.set_yticklabels(labels = [i+1 for i in sorting_callback.sorted_indices])
        ax1.tick_params(axis='both', which='both',length=5, color='#00000050')

        ax2.axes.yaxis.set_visible(False)
        ax2.axes.xaxis.set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['top'].set_alpha(0.2)
        ax2.spines['right'].set_alpha(0.2)
        ax2.spines['bottom'].set_alpha(0.2)
        
    class SortingCallback:
            
        def __init__(self, button: Button, fig: plt.Figure, bar_ax: mp.axes.Axes, imshow_ax: mp.axes.Axes, indices: list[int]):
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
            self._bar_ax.barh(range(len(sums)), np.abs(sums)[self._sorted_indices], color=colors[self._sorted_indices], height=.9)
            init_canvas(ax01, ax02, ax1, ax2)
            self._fig.canvas.draw()
    
    def onclick(event):
        
        if ax1.lines:
            
            for i in range(len(ax1.lines)):
                ax1.lines.remove(ax1.lines[i])
                
        _, iy = event.xdata, event.ydata
        if (event.inaxes == ax1 or event.inaxes == ax2) and event.xdata is not None and event.ydata is not None and 0 < event.xdata < x_lim and -.5 < event.ydata < y_lim:
            iy = int(np.rint(iy))
            color = colors[sorting_callback._sorted_indices[iy]]
            line = mp.lines.Line2D([0, x_lim], [iy, iy], color=color, linewidth=16, alpha=.4)
            ax1.add_line(line)
            fig1.canvas.draw()
            fig2, (ax21, ax22) = plt.subplots(ncols=2, nrows=1)
            plot_patterns(data, info, sorting_callback.sorted_indices[iy], ax21, name_format='', title='')
            ax22.plot(
                                temporal_parameters.franges,
                                temporal_parameters.finputs[sorting_callback.sorted_indices[iy]],
                                temporal_parameters.franges,
                                temporal_parameters.foutputs[sorting_callback.sorted_indices[iy]],
                                temporal_parameters.franges,
                                temporal_parameters.fresponces[sorting_callback.sorted_indices[iy]],
                            )
            ax22.legend(['Filter input', 'Filter output', 'Filter responce'], loc='upper right')      
            ax22.spines['top'].set_alpha(.2)
            ax22.spines['right'].set_alpha(.2)
            ax22.spines['left'].set_alpha(.2)
            ax22.spines['bottom'].set_alpha(.2)
            ax22.tick_params(axis='both', which='both',length=5, color='#00000050')
            ax22.set_xlabel('Frequency (Hz)')
            ax22.set_ylabel('Amplitude (μV)')
            
            if logscale:
                ax22.set_aspect(25)  
                ax22.set_yscale('log')
            else:
                ax22.set_aspect(75)  
                
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
        raise NotImplementedError(f'The "{summarize}" method not implemented. Available methods: "sum", "sumabs", "abssum"')
    colors = np.array(['#f2827a' if sum_ >= 0 else '#8bbae5' for sum_ in sums])

    sort_button = Button(ax02, 'Sort')
    sorting_callback = SortingCallback(sort_button, fig1, ax2, ax1, sorted(range(len(sums)), reverse=True))
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
    spatial_parameters = read_pkl('./Source/Subjects/Ga_Fed_06/Parameters/LM&LI_vs_RM&RI_B1-B8_spatial.pkl')
    temporal_parameters = read_pkl('./Source/Subjects/Ga_Fed_06/Parameters/LM&LI_vs_RM&RI_B1-B8_temporal.pkl')
    orders = read_pkl('./Source/Subjects/Ga_Fed_06/Parameters/LM&LI_vs_RM&RI_B1-B8_sorting.pkl')
    info = read_pkl('./Source/Subjects/Ga_Fed_06/Info/ML_Subject06_P1_tsss_mc_trans_info.pkl')

    info.pick_channels(
        list(
            filter(
                lambda ch_name: (ch_name[-1] == '2' or ch_name[-1] == '3') and 'meg' in ch_name.lower(),
                info['ch_names']
            )
        )
    )
    
    plot_spatial_weights(
        spatial_parameters,
        temporal_parameters,
        info,
        # summarize = [np.random.random()*2 - 1 for _ in range(32)],
        summarize='sum',
        logscale=False
    )

