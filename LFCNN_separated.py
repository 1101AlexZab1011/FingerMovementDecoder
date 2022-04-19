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
from utils.machine_learning import one_hot_decoder
import pickle
from typing import Any, NoReturn, Optional
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
import scipy.signal as sl
import sklearn
import scipy as sp
from LFCNN_decoder import *
from dataclasses import dataclass
from itertools import product
from LFRNN_decoder import LFRNN

@dataclass
class DatasetContainer(object):
    name: str
    n_classes: float 
    classes_samples: list
    meta: dict
    dataset: mf.Dataset

if __name__ == '__main__':
    mpl.use('agg')
    parser = argparse.ArgumentParser(
        description='A script for applying the neural network "LFCNN" to the epoched data from gradiometers related to events for classification'
    )
    parser.add_argument('-cms', '--combine-sessions', type=str, nargs='+',
                        default=[], help='Sessions to combine')
    parser.add_argument('-ep', '--exclude-participants', type=str, nargs='+',
                        default=[], help='IDs of subjects to exclude')
    parser.add_argument('-l', '--lock', type=str,
                        default='RespCor', help='Stimulus lock to consider')
    parser.add_argument('-c', '--cases', type=str, nargs='+',
                        default=['LI', 'LM', 'RI', 'RM'], help='Cases to consider (must match epochs file names for the respective classes)')
    parser.add_argument('-cmc', '--combine-cases', type=str, nargs='+',
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
    parser.add_argument('--project-name', type=str,
                        default='fingers_movement_epochs', help='Name of a project')
    parser.add_argument('-hp', '--high-pass', type=float,
                        default=None, help='High-pass filter (Hz)')
    parser.add_argument('-m', '--model', type=str,
                        default='LFCNN', help='Model to use')
    parser.add_argument('--use-train', action='store_true', help='Use train set from separated dataset to test a model')
    parser.add_argument('--no-params', action='store_true', help='Do not compute parameters')
    
    
    combined_sessions, \
    excluded_subjects, \
    lock, \
    cases, \
    cases_to_combine, \
    subjects_dir, \
    sessions_name,\
    classification_name,\
    classification_postfix,\
    classification_prefix, \
    project_name, \
    lfreq, \
    model_name, \
    use_train, \
    no_params = vars(parser.parse_args()).values()
    
    if model_name == 'LFCNN':
        classifier = mf.models.LFCNN
    elif model_name == 'LFRNN':
        classifier = LFRNN
    else:
        raise NotImplementedError(f'This model is not implemented: {model_name}')
    
    assert len(combined_sessions) == 2, f'Script is implemented for only two combinations of sessions, {len(combined_sessions)} is given'
    
    sessions1, sessions2 = tuple(map(lambda data: tuple(data.split(' ')), combined_sessions))
    cases_to_combine = [case.split(' ') for case in cases] if cases_to_combine is None else [case.split(' ') for case in cases_to_combine]
    cases = list(filter(lambda case: any([case in cmb for cmb in cases_to_combine]), cases))
    cases_to_combine = sorted(cases_to_combine, reverse=True)
    class_names = ['&'.join(sorted(cases_combination, reverse=True)) for cases_combination in cases_to_combine]
    
    if classification_name is None:
        classification_name = '_vs_'.join(class_names)
    
    classification_name_formatted = "_".join(list(filter(lambda s: s not in (None, ""), [classification_prefix, classification_name, classification_postfix])))
    
    perf_tables_path = os.path.join(os.path.dirname(subjects_dir), 'perf_tables')
    pics_path = os.path.join(os.path.dirname(subjects_dir), 'Pictures')
    check_path(perf_tables_path, pics_path)
    
    for subject_name in os.listdir(subjects_dir):
        
        if subject_name in excluded_subjects:
            continue
        
        print(subject_name)
        subject_path = os.path.join(subjects_dir, subject_name)
        epochs_path = os.path.join(subject_path, 'Epochs')
        epochs = {group: {case: list() for case in cases} for group in (sessions1, sessions2)}
        any_info = None
        
        for epochs_file in os.listdir(epochs_path):
            
            if lock not in epochs_file:
                continue
            
            session = re.findall(r'_{0}\d\d?'.format(sessions_name), epochs_file)[0][1:]
            
            if session not in [*sessions1, *sessions2]:
                continue
            
            for case in cases:
                if case in epochs_file:
                    with Silence(), warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        epochs_ = mne.read_epochs(os.path.join(epochs_path, epochs_file))
                        epochs_.resample(200)
                        
                        if any_info is None:
                            any_info = epochs_.info
                        
                        group = sessions1 if session in sessions1 else sessions2
                        epochs[group][case].append(epochs_)
        
        epochs = {
            group: dict(
                zip(
                    epochs[group].keys(),
                    map(
                        mne.concatenate_epochs,
                        list(epochs[group].values())
                    )
                )
            ) for group in (sessions1, sessions2)
        }
        
        tfr_path = os.path.join(subject_path, 'TFR')
        classification_path = os.path.join(
            tfr_path,
            classification_name_formatted
        )
        check_path(tfr_path, classification_path)
        
        datasets = dict()
        
        for group, group_epochs in epochs.items():
            group_name = '&'.join(group)
            
            i = 0
            cases_indices_to_combine = list()
            cases_to_combine_list = list()
            
            for combination in cases_to_combine:
                cases_indices_to_combine.append(list())
                
                for j, case in enumerate(combination):
                    
                    i += j
                    cases_indices_to_combine[-1].append(i)
                    if lfreq is None:
                        cases_to_combine_list.append(group_epochs[case])
                    else:
                        cases_to_combine_list.append(group_epochs[case].filter(lfreq, None))
                    
                i += 1
            
            savepath = os.path.join(
                classification_path,
                group_name
            )
            combiner = EpochsCombiner(*cases_to_combine_list).combine(*cases_indices_to_combine)
            n_classes, classes_samples = np.unique(combiner.Y, return_counts=True)
            n_classes = len(n_classes)
            classes_samples = classes_samples.tolist()
            combiner.shuffle()
            import_opt = dict(
                savepath=savepath+'/',
                out_name=project_name+f'_{group[0]}-{group[-1]}',
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
            
            X, Y = combiner.X, combiner.Y
            meta = mf.produce_tfrecords((X, Y), **import_opt)
            datasets.update({group: DatasetContainer(f'{group[0]}-{group[-1]}', n_classes, classes_samples, meta, mf.Dataset(meta, train_batch=100))})
        
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
        dataset_prev = None
        for dataset_train, dataset_test in product(datasets.values(), repeat=2):
            print(f'Using {dataset_train.name} as a train set and {dataset_test.name} as a test set')
            classification_name_formatted_sep = f'{classification_name_formatted}_train_{dataset_train.name}_test_{dataset_test.name}'
            
            if dataset_train.name != dataset_prev:
                dataset_prev = dataset_train.name 
                model = classifier(dataset_train.dataset, lf_params)
                model.build()
                model.train(n_epochs=25, eval_step=100, early_stopping=5)
                network_out_path = os.path.join(subject_path, f'{model_name}_train_{dataset_train.name}_test_{dataset_test.name}')
                yp_path = os.path.join(network_out_path, 'Predictions')
                sp_path = os.path.join(network_out_path, 'Parameters')
                check_path(network_out_path, yp_path, sp_path)
            
            test_data = dataset_test.dataset.train if dataset_train.name != dataset_test.name and use_train else dataset_test.dataset.test
            
            y_true_train, y_pred_train = model.predict(dataset_train.dataset.train)
            y_true_test, y_pred_test = model.predict(test_data)
            
            print(f'{model_name} performance (train {dataset_train.name}, test {dataset_test.name})')
            print('\ttrain-set: ', subject_name, sklearn.metrics.accuracy_score(one_hot_decoder(y_true_train), one_hot_decoder(y_pred_train)))
            print('\ttest-set: ', subject_name, sklearn.metrics.accuracy_score(one_hot_decoder(y_true_test), one_hot_decoder(y_pred_test)))
            
            train_loss_, train_acc_ = model.evaluate(dataset_train.dataset.train)
            test_loss_, test_acc_ = model.evaluate(test_data)
            
            if dataset_train.name != dataset_prev and not no_params:
                model.compute_patterns(meta['train_paths'])
                nt = model.dataset.h_params['n_t']
                time_courses = np.squeeze(model.lat_tcs.reshape([model.specs['n_latent'], -1, nt]))
                times = (1/float(model.dataset.h_params['fs']))*np.arange(model.dataset.h_params['n_t'])
                patterns = model.patterns.copy()
                model.compute_patterns(meta['train_paths'], output='filters')
                filters = model.patterns.copy()
                franges, finputs, foutputs, fresponces = compute_temporal_parameters(model)
                induced, times, time_courses = compute_waveforms(model)
                
                save_parameters(
                    Predictions(
                        y_pred_test,
                        y_true_test
                    ),
                    os.path.join(yp_path, f'{classification_name_formatted_sep}_pred.pkl'),
                    'predictions'
                )
                save_parameters(
                    WaveForms(time_courses.mean(1), induced, times, time_courses),
                    os.path.join(sp_path, f'{classification_name_formatted_sep}_waveforms.pkl'),
                    'WaveForms'
                )
                save_parameters(
                    SpatialParameters(patterns, filters),
                    os.path.join(sp_path, f'{classification_name_formatted_sep}_spatial.pkl'),
                    'spatial'
                )
                save_parameters(
                    TemporalParameters(franges, finputs, foutputs, fresponces),
                    os.path.join(sp_path, f'{classification_name_formatted_sep}_temporal.pkl'),
                    'temporal'
                )
                get_order = lambda order, ts: order.ravel()
                save_parameters(
                    ComponentsOrder(
                        get_order(*model._sorting('l2')),
                        get_order(*model._sorting('compwise_loss')),
                        get_order(*model._sorting('weight')),
                        get_order(*model._sorting('output_corr')),
                        get_order(*model._sorting('weight_corr')),
                    ),
                    os.path.join(sp_path, f'{classification_name_formatted_sep}_sorting.pkl'),
                    'sorting'
                )
            
            used_test_fold = 'train_size' if dataset_train.name != dataset_test.name and use_train else 'test_size'
            
            processed_df = pd.Series(
                [
                    f'{dataset_train.n_classes}/{dataset_test.n_classes}',
                    *[f'{cs1}/{cs2}' for cs1, cs2 in zip(dataset_train.classes_samples, dataset_test.classes_samples)],
                    f'{sum(dataset_train.classes_samples)}/{sum(dataset_test.classes_samples)}',
                    f'{dataset_train.meta["test_size"]}/{dataset_test.meta[used_test_fold]}',
                    train_acc_,
                    # train_loss_,
                    test_acc_,
                    # test_loss_,
                    model.v_metric,
                    # model.v_loss,
                ],
                index=[
                    'n_classes',
                    *class_names,
                    'total',
                    'test_set',
                    'train_acc',
                    # 'train_loss',
                    'test_acc',
                    # 'test_loss',
                    'val_acc',
                    # 'val_loss'
                ],
                name=subject_name
            ).to_frame().T
            perf_table_path = os.path.join(
                perf_tables_path,
                f'{classification_name_formatted}_train_{dataset_train.name}_test_{dataset_test.name}_sep.csv'
            )
            if os.path.exists(perf_table_path):
                pd.concat([pd.read_csv(perf_table_path, index_col=0, header=0), processed_df], axis=0)\
                .to_csv(perf_table_path)
            else:
                processed_df\
                .to_csv(perf_table_path)