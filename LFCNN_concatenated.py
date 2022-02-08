import argparse
from collections import namedtuple
import os
import re
import warnings
import mne
import numpy as np
import pandas as pd
import tensorflow as tf
import mneflow as mf
from combiners import EpochsCombiner
from utils.console import Silence
from utils.console.spinner import spinner
from utils.data_management import dict2str
from utils.storage_management import check_path
import pickle
from typing import Any, NoReturn
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
from LFCNN_decoder import SpatialParameters, save_spatial_parameters, save_model_weights, plot_patterns, plot_waveforms


if __name__ == '__main__':
    mpl.use('agg')
    parser = argparse.ArgumentParser(
        description='A script for applying the neural network "LFCNN" to the epoched data from gradiometers related to events for classification with all subjects concatenated in one pseudo-subject'
    )
    parser.add_argument('-es', '--exclude-sessions', type=str, nargs='+',
                        default=[], help='Sessions to exclude')
    parser.add_argument('-ep', '--exclude-participants', type=str, nargs='+',
                        default=[], help='IDs of subjects to exclude')
    parser.add_argument('-l', '--lock', type=str,
                        default='RespCor', help='Stimulus lock to consider')
    parser.add_argument('-c', '--cases', type=str, nargs='+',
                        default=['LI', 'LM', 'RI', 'RM'], help='Cases to consider (must match epochs file names for the respective classes)')
    parser.add_argument('-cmb', '--combine-cases', type=str, nargs='+',
                        default=None, help='Cases to consider (must be the number of strings in which classes to combine are written separated by a space, indices corresponds to order of "--cases" parameter)')
    parser.add_argument('-sd', '--subjects-dir', type=str,
                        default=os.path.join(os.getcwd(), 'Source', 'Subjects'), help='Path to the subjects directory')
    parser.add_argument('--trials-name', type=str,
                        default='B', help='Name of trials')
    parser.add_argument('--name', type=str,
                        default=None, help='Name of a task')
    parser.add_argument('--postfix', type=str,
                        default='', help='String to append to a task name')
    parser.add_argument('--prefix', type=str,
                        default='', help='String to set in the start of a task name')
    parser.add_argument('--project_name', type=str,
                        default='fingers_movement_epochs', help='Name of a project')
    
    excluded_sessions, \
    excluded_subjects, \
    lock, \
    cases, \
    cases_to_combine, \
    subjects_dir, \
    sessions_name,\
    classification_name,\
    classification_postfix,\
    classification_prefix, \
    project_name = vars(parser.parse_args()).values()
    
    if excluded_sessions:
        excluded_sessions = [sessions_name + session if sessions_name not in session else session for session in excluded_sessions]
    
    cases_to_combine = [case.split(' ') for case in cases] if cases_to_combine is None else [case.split(' ') for case in cases_to_combine]
    cases = list(filter(lambda case: any([case in cmb for cmb in cases_to_combine]), cases))
    perf_tables_path = os.path.join(os.path.dirname(subjects_dir), 'perf_tables')
    check_path(perf_tables_path)
    subjects_performance = list()
    
    for subject_name in os.listdir(subjects_dir):
        print(f'Reading {subject_name}', end='')
        if subject_name in excluded_subjects:
            continue
        
        train_acc, train_loss, val_acc, val_loss, test_acc, test_loss = (list() for _ in range(6))
        subject_path = os.path.join(subjects_dir, subject_name)
        epochs_path = os.path.join(subject_path, 'Epochs')
        epochs = {case: list() for case in cases}
        any_info = None
        
        for epochs_file in os.listdir(epochs_path):
            if lock not in epochs_file:
                continue
            
            session = re.findall(r'_{0}\d\d?'.format(sessions_name), epochs_file)[0][1:]
            
            if session in excluded_sessions:
                continue
            
            print(f'.', end='')
            for case in cases:
                if case in epochs_file:
                    with Silence(), warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        epochs_ = mne.read_epochs(os.path.join(epochs_path, epochs_file))
                        epochs_.resample(200)
                        
                        if any_info is None:
                            any_info = epochs_.info
                        
                        epochs[case].append(epochs_)
                        
        print('\tOK')
    
    print('Concatenating epochs...', end='')
    epochs = dict(
                zip(
                    epochs.keys(),
                    map(
                        mne.concatenate_epochs,
                        list(epochs.values())
                    )
                )
            )
    print('\tOK')
    cases_to_combine_list = list()
    cases_indices_to_combine = list()
    
    i = 0
    for combination in cases_to_combine:
        cases_indices_to_combine.append(list())
        
        for j, case in enumerate(combination):
            i += j
            cases_indices_to_combine[-1].append(i)
            cases_to_combine_list.append(epochs[case])
            
        i += 1
    
    class_names = ['&'.join(cases_combination) for cases_combination in cases_to_combine]
    
    if classification_name is None:
        classification_name = '_vs_'.join(class_names)
        
    combiner = EpochsCombiner(*cases_to_combine_list).combine(*cases_indices_to_combine)
    n_classes, classes_samples = np.unique(combiner.Y, return_counts=True)
    n_classes = len(n_classes)
    classes_samples = classes_samples.tolist()
    combiner.shuffle()
    pseudo_subject_home = os.path.join('tmp', 'FingerMovementDecoder')
    subject_path = os.path.join(pseudo_subject_home, 'PseudoSubject')
    subject_name = f'Pse_Udo_{classification_postfix}'
    tfr_path = os.path.join(subject_path, 'TFR')
    check_path(pseudo_subject_home, subject_path, tfr_path)
    classification_name_formatted = "_".join(list(filter(lambda s: s not in (None, ""), [classification_prefix, classification_name, classification_postfix])))
    savepath = os.path.join(
        tfr_path,
        classification_name_formatted
    )
    import_opt = dict(
            savepath=savepath,
            out_name=project_name,
            fs=200,
            input_type='trials',
            target_type='int',
            picks={'meg': 'grad'},
            scale=True,
            crop_baseline=True,
            decimate=None,
            scale_interval=(0, 60),
            n_folds=5,
            overwrite=True,
            segment=False,
            test_set='holdout'
        )
    meta = mf.produce_tfrecords((combiner.X, combiner.Y), **import_opt)
    dataset = mf.Dataset(meta, train_batch=100)
    lf_params = dict(
            n_latent=32,
            filter_length=50,
            nonlin=tf.keras.activations.elu,
            padding='SAME',
            pooling=10,
            stride=10,
            pool_type='max',
            model_path=import_opt['savepath'],
            dropout=.4,
            l2_scope=["weights"],
            l2=1e-6
    )
    model = mf.models.LFCNN(dataset, lf_params)
    model.build()
    model.train(n_epochs=25, eval_step=100, early_stopping=5)
    train_loss_, train_acc_ = model.evaluate(meta['train_paths'])
    test_loss_, test_acc_ = model.evaluate(meta['test_paths'])
    model.compute_patterns(os.path.join(savepath, f'{project_name}_train_0.tfrecord'))
    patterns = model.patterns.copy()
    model.compute_patterns(os.path.join(savepath, f'{project_name}_train_0.tfrecord'), output='filters')
    filters = model.patterns.copy()
    sp_path = os.path.join(subject_path, 'Parameters')
    check_path(sp_path)
    save_spatial_parameters(SpatialParameters(patterns, filters), os.path.join(sp_path, f'{classification_name_formatted}.pkl'))
    pics_path = os.path.join(os.path.dirname(subjects_dir), 'Pictures')
    patterns_pics_path = os.path.join(pics_path, 'Patterns')
    filters_pics_path = os.path.join(pics_path, 'Filters')
    spectra_pics_path = os.path.join(pics_path, 'Spectra')
    wf_pics_path = os.path.join(pics_path, 'WaveForms')
    check_path(pics_path, patterns_pics_path, filters_pics_path, spectra_pics_path, wf_pics_path)
    patterns_fig = plot_patterns(patterns, any_info)
    patterns_fig.savefig(os.path.join(patterns_pics_path, f'{subject_name}_{classification_name_formatted}.png'))
    plt.close(patterns_fig)
    filters_fig = plot_patterns(filters, any_info)
    filters_fig.savefig(os.path.join(filters_pics_path, f'{subject_name}_{classification_name_formatted}.png'))
    plt.close(filters_fig)
    spectra_fig = model.plot_spectra(sorting='weight_corr', class_names=class_names)
    spectra_fig.savefig(os.path.join(spectra_pics_path, f'{subject_name}_{classification_name_formatted}.png'))
    plt.close(spectra_fig)
    wf_fig = plot_waveforms(model, class_names=class_names)
    wf_fig.savefig(os.path.join(wf_pics_path, f'{subject_name}_{classification_name_formatted}.png'))
    plt.close(wf_fig)
    weights_path = os.path.join(subject_path, 'Weights')
    check_path(weights_path)
    save_model_weights(
        model,
        os.path.join(
            weights_path,
            f'{classification_name_formatted}.h5'
        )
    )
    perf_table_path = os.path.join(
        perf_tables_path,
        f'{classification_name_formatted}.csv'
    )
    processed_df = pd.Series(
        [
            n_classes,
            *classes_samples,
            sum(classes_samples),
            train_acc_,
            train_loss_,
            test_acc_,
            test_loss_,
            model.v_metric,
            model.v_loss,
            
        ],
        index=['n_classes', *class_names, 'total', 'train_acc', 'train_loss', 'test_acc', 'test_loss', 'val_acc', 'val_loss'],
        name=subject_name
    ).to_frame().T
    
    if os.path.exists(perf_table_path):
        pd.concat([pd.read_csv(perf_table_path, index_col=0, header=0), processed_df], axis=0)\
            .to_csv(perf_table_path)
    else:
        processed_df\
        .to_csv(perf_table_path)
        