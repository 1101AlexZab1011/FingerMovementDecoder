import matplotlib.pyplot as plt
import numpy as np
import mne
import pickle
from typing import *
from LFCNN_decoder import SpatialParameters, TemporalParameters, ComponentsOrder
from dataclasses import dataclass
import matplotlib as mp
from typing import *
import copy
import argparse
from collections import namedtuple
import os
import re
import warnings
import mne
import numpy as np
import pandas as pd
from combiners import EpochsCombiner
from utils.console import Silence
from utils.console.spinner import spinner
from utils.data_management import dict2str
from utils.storage_management import check_path
import pickle
from typing import Any, NoReturn, Optional
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
import scipy.signal as sl
# from mat_to_fif import read_pkl

def read_pkl(path: str) -> Any:
    with open(
            path,
            'rb'
        ) as file:
        content = pickle.load(
            file
        )
    return content

def plot_patterns(patterns, info, order=None, axes=None, cmap='RdBu_r', sensors=True,
                colorbar=False, res=64,
                size=1, cbar_fmt='%3.1f', name_format='Latent\nSource %01d',
                show=True, show_names=False, title=None,
                outlines='head', contours=6,
                image_interp='bilinear'):
    
    if order is None:
        order = range(patterns.shape[1])
    
    if title is None:
        title=f'Computed patterns'
    
    import copy
    info = copy.deepcopy(info)
    info['sfreq'] = 1.
    patterns = mne.EvokedArray(patterns, info, tmin=0)
    return patterns.plot_topomap(
        times=order,
        axes=axes,
        cmap=cmap, colorbar=colorbar, res=res,
        cbar_fmt=cbar_fmt, sensors=sensors, units=None, time_unit='s',
        time_format=name_format, size=size, show_names=show_names,
        title=title, outlines=outlines,
        contours=contours, image_interp=image_interp, show=show)


def plot_spectra(temporal_parameters, order, title='', xlim=None, ylim=None, legend=None):
    
    if not len(order)%3:
        n_cols = 3
        n_rows = len(order)//3
    elif not len(order)%2:
        n_cols = 2
        n_rows = len(order)//2
    elif len(order) == 3:
        n_rows = 1
        n_cols = 3
    else:
        n_rows = len(order)//3+1
        n_cols = 3
    
    if legend is None:
        legend = ['Filter input', 'Filter output', 'Filter responce']
    
    fig, axs = plt.subplots(n_rows, n_cols)
    
    if len(axs.shape) == 1:
        axs = np.reshape(axs, (1, -1))
    
    fig.set_size_inches(n_cols*5, n_rows*3.75)
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
    title: Optional[str] = None,
    xlim: Optional[Union[int, float]] = None,
    ylim: Optional[Union[int, float]] = None,
    legend: Optional[Union[int, float]] = None,
    spatial_data_type: Optional[str] = 'patterns',
    topomap_kwargs: Optional[dict] = None
) -> mp.figure.Figure:
    
    def wrap_in_list(content):
        return [content] if not isinstance(content, list) else content
    
    def validate_length(*iterables):
        length = len(iterables[0])
        for i, sample in enumerate(iterables[1:]):
            if len(sample) != length:
                raise ValueError(f'Length validation failed: all elements must have length equal to {length}, but element {i} has length: {len(sample)}')
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
    
    if not n_components%3:
        n_cols = 3
        n_rows = n_components//3
    elif not n_components%2:
        n_cols = 2
        n_rows = n_components//2
    elif n_components == 3:
        n_rows = 1
        n_cols = 3
    else:
        n_rows = n_components//3+1
        n_cols = 3
    
    n_rows_per_subject = 2*n_rows
    n_rows = n_subjects*n_rows_per_subject
    
    if legend is None:
        legend = ['Filter input', 'Filter output', 'Filter responce']
    
    fig, axs = plt.subplots(n_rows, n_cols)
    
    if len(axs.shape) == 1:
        axs = np.reshape(axs, (1, -1))
    
    fig.set_size_inches(n_cols*5, n_rows*3.75)
    
    subplots_map = np.ones((n_rows, n_cols)).astype(bool)

    current_comp = 0
    for i in range(0, n_rows, 2):
        for j in range(n_cols):
            
            if current_comp >= n_components:
                subplots_map[i, j], subplots_map[i+1, j] = False, False
                
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
                
                if i%2 and j == 0:
                    axs[i, j].set_ylabel(subject_names[tracker.subject])
                    axs[i, j].tick_params(axis='y', pad=300)
                elif not i%2 and j == 0:
                    axs[i, j].set_ylabel(subject_names[tracker.subject])
                    
                if not i%2:
                    axs[i, j].set_title(f'Latent Source {orders[tracker.subject][tracker.top]}')
                    axs[i, j].plot(
                        temporal_parameters[tracker.subject].franges,
                        temporal_parameters[tracker.subject].finputs[orders[tracker.subject][tracker.top]],
                        temporal_parameters[tracker.subject].franges,
                        temporal_parameters[tracker.subject].foutputs[orders[tracker.subject][tracker.top]],
                        temporal_parameters[tracker.subject].franges,
                        temporal_parameters[tracker.subject].fresponces[orders[tracker.subject][tracker.top]],
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


if __name__ == '__main__':
    mpl.use('agg')
    parser = argparse.ArgumentParser(
        description='A script for applying the neural network "LFCNN" to the epoched data from gradiometers related to events for classification'
    )
    parser.add_argument('-s', '--sort', type=str,
                        default='l2', help='Method to sort components')
    parser.add_argument('-ep', '--exclude-participants', type=str, nargs='+',
                        default=[], help='IDs of subjects to exclude')
    parser.add_argument('-cmb', '--combine-cases', type=str, nargs='+',
                        default=None, help='Cases to consider (must be the number of strings in which classes to combine are written separated by a space, indices corresponds to order of "--cases" parameter)')
    parser.add_argument('-sd', '--subjects-dir', type=str,
                        default=os.path.join(os.getcwd(), 'Source', 'Subjects'), help='Path to the subjects directory')
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
        raise ValueError(f'Wrong option for sorting: {sort}. Sortings can be \'l2\', \'compwise_loss\', \'weight\', \'output_corr\', \'weight_corr\'')
    
    cases_to_combine = [case.split(' ') for case in cases_to_combine]
    
    class_names = ['&'.join(sorted(cases_combination, reverse=True)) for cases_combination in cases_to_combine]
        
    if classification_name is None:
        classification_name = '_vs_'.join(class_names)
        
    classification_name_formatted = "_".join(list(filter(lambda s: s not in (None, ""), [classification_prefix, classification_name, classification_postfix])))
    
    
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
                    lambda ch_name: (ch_name[-1] == '2' or ch_name[-1] == '3') and 'meg' in ch_name.lower(),
                    info['ch_names']
                )
            )
        )
        all_info.append(info)
        all_spatial_parameters.append(read_pkl(os.path.join(parametes_path, f'{classification_name_formatted}_spatial.pkl')))
        all_temporal_parameters.append(read_pkl(os.path.join(parametes_path, f'{classification_name_formatted}_temporal.pkl')))
        all_sortings.append(read_pkl(os.path.join(parametes_path, f'{classification_name_formatted}_sorting.pkl'))._asdict()[sort])
    
    pics_path = os.path.join(os.path.dirname(subjects_dir), 'Pictures')
    tempospectral_pics_path = os.path.join(pics_path, 'TempoSpectral')
    check_path(pics_path, tempospectral_pics_path)
    fig = plot_tempospectral(all_spatial_parameters, all_temporal_parameters, all_sortings, all_info, all_subjects, spatial_data_type=spatial_data_type)
    fig.savefig(os.path.join(tempospectral_pics_path, f'{classification_name_formatted}_{spatial_data_type}_{sort}.png'))
    