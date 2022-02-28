import argparse
import os
import re
import warnings
import mne
import numpy as np
import pandas as pd
import tensorflow as tf
import mneflow as mf
from combiners import EpochsCombiner
from utils.console import Silence, edit_previous_line
from utils.storage_management import check_path
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn
from LFCNN_decoder import Predictions, SpatialParameters, TemporalParameters, ComponentsOrder, compute_temporal_parameters, save_parameters, save_model_weights, plot_patterns, plot_waveforms

if __name__ == '__main__':
    mpl.use('agg')
    parser = argparse.ArgumentParser(
        description='A script for applying the neural network "LFCNN" to the epoched data from gradiometers related to events for classification with all subjects concatenated in one pseudo-subject'
    )
    parser.add_argument('-es', '--exclude-sessions', type=str, nargs='+',
                        default=[], help='Sessions to exclude')
    parser.add_argument('-s', '--subjects', type=str, nargs='+',
                        default=[], help='IDs of subject to concatenate epochs')
    parser.add_argument('-l', '--lock', type=str,
                        default='RespCor', help='Stimulus lock to consider')
    parser.add_argument('-c', '--cases', type=str, nargs='+',
                        default=['LI', 'LM', 'RI', 'RM'], help='Cases to consider (must match epochs file names for the respective classes)')
    parser.add_argument('-sd', '--subjects-dir', type=str,
                        default=os.path.join(os.getcwd(), 'Source', 'Subjects'), help='Path to the subjects directory')
    parser.add_argument('--trials-name', type=str,
                        default='B', help='Name of trials')
    parser.add_argument('--name', str_type=,
                        default='concatenated_epochs', help='Name of a task')
    parser.add_argument('--postfix', type=str,
                        default='', help='String to append to a task name')
    parser.add_argument('--prefix', type=str,
                        default='', help='String to set in the start of a task name')
    parser.add_argument('--project_name', type=str,
                        default='fingers_movement_epochs', help='Name of a project')
    
    excluded_sessions, \
    subject_name, \
    lock, \
    cases, \
    subjects_dir, \
    sessions_name,\
    classification_name,\
    classification_postfix,\
    classification_prefix, \
    project_name = vars(parser.parse_args()).values()
    
    if excluded_sessions:
        excluded_sessions = [sessions_name + session if sessions_name not in session else session for session in excluded_sessions]
    
    subject_path = os.path.join(subjects_dir, subject_name)
    epochs_path = os.path.join(subject_path, 'Epochs')
    concatenated_epochs_path = os.path.join(subject_path, 'ConcatenatedEpochs')
    savepath = os.path.join(concatenated_epochs_path, f'{classification_prefix}_{classification_name}_{classification_postfix}')
    check_path(concatenated_epochs_path, savepath)
    
    epochs = {case: list() for case in cases}
    msg = f'Reading {subject_name}\t'
    print(msg)
    
    for epochs_file in os.listdir(epochs_path):
        if lock not in epochs_file:
            continue
        
        session = re.findall(r'_{0}\d\d?'.format(sessions_name), epochs_file)[0][1:]
        
        if session in excluded_sessions:
            continue
        msg += '.'
        edit_previous_line(msg)
        for case in cases:
            if case in epochs_file:
                with Silence(), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    epochs_ = mne.read_epochs(os.path.join(epochs_path, epochs_file))
                    epochs_.resample(200)
                    
                    if any_info is None:
                        any_info = epochs_.info
                    
                    epochs[case].append(epochs_)
                    
    edit_previous_line(msg + '\tOK')
    
    print('Concatenating epochs...')
    epochs = dict(
                zip(
                    epochs.keys(),
                    map(
                        mne.concatenate_epochs,
                        list(epochs.values())
                    )
                )
            )
    print('Epochs are concatenated')
    
    for case, epoch in epochs.items():
        epoch.save(os.path.join(savepath, f'epochs_{case}.fif'))
    