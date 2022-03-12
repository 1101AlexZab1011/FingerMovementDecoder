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
import mneflow
from LFCNN_decoder import SpatialParameters, TemporalParameters,\
    ComponentsOrder, Predictions, WaveForms,\
    compute_temporal_parameters, save_parameters
from mneflow.models import BaseModel, LFCNN, VARCNN, Deep4
from utils.machine_learning.designer import ModelDesign, ParallelDesign, LayerDesign
from utils.machine_learning.analyzer import ModelAnalyzer
from mneflow.layers import DeMixing, LFTConv, TempPooling, Dense
from scipy.signal import freqz, welch
from scipy.stats import spearmanr
import tensorflow.keras.regularizers as k_reg
from tensorflow.keras.initializers import Constant

class LFRNN(BaseModel):
    def __init__(self, Dataset, specs=dict()):
        self.scope = 'lfcnn'
        specs.setdefault('filter_length', 7)
        specs.setdefault('n_latent', 32)
        specs.setdefault('pooling', 3)
        specs.setdefault('stride', 3)
        specs.setdefault('padding', 'SAME')
        specs.setdefault('pool_type', 'max')
        specs.setdefault('nonlin', tf.nn.relu)
        specs.setdefault('l1', 3e-4)
        specs.setdefault('l2', 0)
        specs.setdefault('l1_scope', ['fc', 'demix', 'lf_conv'])
        specs.setdefault('l2_scope', [])
        specs.setdefault('maxnorm_scope', [])
        
        super(LFRNN, self).__init__(Dataset, specs)

    def build_graph(self):
        # LFRNN
        # self.design = ModelDesign(
        #     self.inputs,
        #     LayerDesign(tf.squeeze, axis=1),
        #     tf.keras.layers.Bidirectional(
        #         tf.keras.layers.LSTM(
        #             self.specs['n_latent'],
        #             bias_regularizer='l1',
        #             return_sequences=True,
        #             kernel_regularizer=tf.keras.regularizers.L1(.01),
        #             recurrent_regularizer=tf.keras.regularizers.L1(.01),
        #             dropout=0.4,
        #             recurrent_dropout=0.4,
        #         ),
        #         merge_mode='sum'
        #     ),
        #     LayerDesign(tf.expand_dims, axis=1),
        #     LFTConv(
        #         size=self.specs['n_latent'],
        #         nonlin=self.specs['nonlin'],
        #         filter_length=self.specs['filter_length'],
        #         padding=self.specs['padding'],
        #         specs=self.specs
        #     ),
        #     TempPooling(
        #         pooling=self.specs['pooling'],
        #         pool_type=self.specs['pool_type'],
        #         stride=self.specs['stride'],
        #         padding=self.specs['padding'],
        #     ),
        #     tf.keras.layers.Dropout(self.specs['dropout'], noise_shape=None),
        #     Dense(size=self.out_dim, nonlin=tf.identity, specs=self.specs)
        # )
        # resLFRNN
        # self.design = ModelDesign(
        #     self.inputs,
        #     LayerDesign(tf.squeeze, axis=1),
        #     tf.keras.layers.Bidirectional(
        #         tf.keras.layers.LSTM(
        #             self.specs['n_latent'],
        #             bias_regularizer='l1',
        #             return_sequences=True,
        #             kernel_regularizer=tf.keras.regularizers.L1(.01),
        #             recurrent_regularizer=tf.keras.regularizers.L1(.01),
        #             dropout=0.4,
        #             recurrent_dropout=0.4,
        #         ),
        #         merge_mode='sum'
        #     ),
        #     LayerDesign(tf.expand_dims, axis=1),
        #     ParallelDesign(
        #         LFTConv(
        #             size=self.specs['n_latent'],
        #             nonlin=self.specs['nonlin'],
        #             filter_length=self.specs['filter_length']//2,
        #             padding=self.specs['padding'],
        #             specs=self.specs
        #         ),
        #         LFTConv(
        #             size=self.specs['n_latent'],
        #             nonlin=self.specs['nonlin'],
        #             filter_length=self.specs['filter_length'],
        #             padding=self.specs['padding'],
        #             specs=self.specs
        #         ),
        #         LFTConv(
        #             size=self.specs['n_latent'],
        #             nonlin=self.specs['nonlin'],
        #             filter_length=self.specs['filter_length']*2,
        #             padding=self.specs['padding'],
        #             specs=self.specs
        #         ),
        #     ),
        #     TempPooling(
        #         pooling=self.specs['pooling'],
        #         pool_type=self.specs['pool_type'],
        #         stride=self.specs['stride'],
        #         padding=self.specs['padding'],
        #     ),
        #     tf.keras.layers.Dropout(self.specs['dropout'], noise_shape=None),
        #     Dense(size=self.out_dim, nonlin=tf.identity, specs=self.specs)
        # )
        # resLFCNN
        # self.design = ModelDesign(
        #     self.inputs,
        #     DeMixing(size=self.specs['n_latent'], nonlin=tf.identity, axis=3, specs=self.specs),
        #     ParallelDesign(
        #         LFTConv(
        #             size=self.specs['n_latent'],
        #             nonlin=self.specs['nonlin'],
        #             filter_length=self.specs['filter_length']//2,
        #             padding=self.specs['padding'],
        #             specs=self.specs
        #         ),
        #         LFTConv(
        #             size=self.specs['n_latent'],
        #             nonlin=self.specs['nonlin'],
        #             filter_length=self.specs['filter_length'],
        #             padding=self.specs['padding'],
        #             specs=self.specs
        #         ),
        #         LFTConv(
        #             size=self.specs['n_latent'],
        #             nonlin=self.specs['nonlin'],
        #             filter_length=self.specs['filter_length']*2,
        #             padding=self.specs['padding'],
        #             specs=self.specs
        #         ),
        #     ),
        #     TempPooling(
        #         pooling=self.specs['pooling'],
        #         pool_type=self.specs['pool_type'],
        #         stride=self.specs['stride'],
        #         padding=self.specs['padding'],
        #     ),
        #     tf.keras.layers.Dropout(self.specs['dropout'], noise_shape=None),
        #     Dense(size=self.out_dim, nonlin=tf.identity, specs=self.specs)
        # )
        # sLFCNN
        # self.design = ModelDesign(
        #     self.inputs,
        #     DeMixing(size=self.specs['n_latent'], nonlin=tf.identity, axis=3, specs=self.specs),
        #     LFTConv(
        #         size=self.specs['n_latent'],
        #         nonlin=self.specs['nonlin'],
        #         filter_length=self.specs['filter_length'],
        #         padding=self.specs['padding'],
        #         specs=self.specs
        #     ),
        #     tf.keras.layers.DepthwiseConv2D((1, self.inputs.shape[2]), padding='valid', activation='relu', kernel_regularizer='l1'),
        #     tf.keras.layers.Dropout(self.specs['dropout'], noise_shape=None),
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(self.out_dim, kernel_regularizer='l1'),
        # )
        # sLFRNN
        # self.design = ModelDesign(
        #     self.inputs,
        #     LayerDesign(tf.squeeze, axis=1),
        #     tf.keras.layers.Bidirectional(
        #         tf.keras.layers.LSTM(
        #             self.specs['n_latent'],
        #             bias_regularizer='l1',
        #             return_sequences=True,
        #             kernel_regularizer=tf.keras.regularizers.L1(.01),
        #             recurrent_regularizer=tf.keras.regularizers.L1(.01),
        #             dropout=0.4,
        #             recurrent_dropout=0.4,
        #         ),
        #         merge_mode='sum'
        #     ),
        #     LayerDesign(tf.expand_dims, axis=1),
        #     LFTConv(
        #         size=self.specs['n_latent'],
        #         nonlin=self.specs['nonlin'],
        #         filter_length=self.specs['filter_length'],
        #         padding=self.specs['padding'],
        #         specs=self.specs
        #     ),
        #     tf.keras.layers.DepthwiseConv2D((1, self.inputs.shape[2]), padding='valid', activation='relu', kernel_regularizer='l1'),
        #     tf.keras.layers.Dropout(self.specs['dropout'], noise_shape=None),
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(self.out_dim, kernel_regularizer='l1'),
        # )
        
        #deep4
        ModelDesign(
            self.inputs,
            LayerDesign(tf.transpose, [0,3,2,1]),
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, self.specs['filter_length']),
                depth_multiplier = self.specs['n_latent'],
                strides=1,
                padding=self.specs['padding'],
                activation = tf.identity,
                kernel_initializer="he_uniform",
                bias_initializer=Constant(0.1),
                data_format="channels_last",
                kernel_regularizer=k_reg.l2(self.specs['l2'])
                #kernel_constraint="maxnorm"
            ),
            *[ModelDesign(
                tf.keras.layers.Conv2D(
                    filters=self.specs['n_latent'],
                    kernel_size=(self.dataset.h_params['n_ch'], 1),
                    strides=1,
                    padding=self.specs['padding'],
                    activation=self.specs['nonlin'],
                    kernel_initializer="he_uniform",
                    bias_initializer=Constant(0.1),
                    data_format="channels_last",
                    #data_format="channels_first",
                    kernel_regularizer=k_reg.l2(self.specs['l2'])
                ),
                TempPooling(
                    pooling=self.specs['pooling'],
                    pool_type="avg",
                    stride=self.specs['stride'],
                    padding='SAME',
                )
            ) for _ in range(4)],
            Dense(size=self.out_dim, nonlin=tf.softmax)
        )

        return self.design()
    



if __name__ == '__main__':
    mpl.use('agg')
    parser = argparse.ArgumentParser(
        description='A script for applying the neural network "LFCNN" to the epoched data from gradiometers related to events for classification'
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
    cases_to_combine = sorted(cases_to_combine, reverse=True)
    class_names = ['&'.join(sorted(cases_combination, reverse=True)) for cases_combination in cases_to_combine]
    
    if classification_name is None:
        classification_name = '_vs_'.join(class_names)
    
    perf_tables_path = os.path.join(os.path.dirname(subjects_dir), 'perf_tables')
    pics_path = os.path.join(os.path.dirname(subjects_dir), 'Pictures')
    check_path(perf_tables_path, pics_path)
    subjects_performance = list()
    
    for subject_name in os.listdir(subjects_dir):
        
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
            
            for case in cases:
                if case in epochs_file:
                    with Silence(), warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        epochs_ = mne.read_epochs(os.path.join(epochs_path, epochs_file))
                        epochs_.resample(200)
                        
                        if any_info is None:
                            any_info = epochs_.info
                        
                        epochs[case].append(epochs_)
        
        epochs = dict(
                    zip(
                        epochs.keys(),
                        map(
                            mne.concatenate_epochs,
                            list(epochs.values())
                        )
                    )
                )
        
        i = 0
        cases_indices_to_combine = list()
        cases_to_combine_list = list()
        for combination in cases_to_combine:
            cases_indices_to_combine.append(list())
            
            for j, case in enumerate(combination):
                
                if i == 0:
                    epo_sample_pics_path = os.path.join(pics_path, 'Epo_Samples')
                    check_path(epo_sample_pics_path)
                    fig = epochs[case].plot('MEG0113', show=False)
                    plt.savefig(os.path.join(epo_sample_pics_path, f'{subject_name}_unfiltered_epo.png'))
                    plt.close()
                    fig = epochs[case].plot_psd(show=False)
                    plt.savefig(os.path.join(epo_sample_pics_path, f'{subject_name}_unfiltered_epo_psd.png'))
                    plt.close()
                    
                
                i += j
                cases_indices_to_combine[-1].append(i)
                cases_to_combine_list.append(epochs[case].filter(3, None))
                
                if i == 0:
                    epo_sample_pics_path = os.path.join(pics_path, 'Epo_Samples')
                    check_path(epo_sample_pics_path)
                    fig = epochs[case].plot('MEG0113', show=False)
                    plt.savefig(os.path.join(epo_sample_pics_path, f'{subject_name}_filtered_epo.png'))
                    plt.close()
                    fig = epochs[case].plot_psd(show=False)
                    plt.savefig(os.path.join(epo_sample_pics_path, f'{subject_name}_filtered_epo_psd.png'))
                    plt.close()
                
            i += 1
        combiner = EpochsCombiner(*cases_to_combine_list).combine(*cases_indices_to_combine)
        n_classes, classes_samples = np.unique(combiner.Y, return_counts=True)
        n_classes = len(n_classes)
        classes_samples = classes_samples.tolist()
        combiner.shuffle()
        tfr_path = os.path.join(subject_path, 'TFR')
        check_path(tfr_path)
        classification_name_formatted = "_".join(list(filter(lambda s: s not in (None, ""), [classification_prefix, classification_name, classification_postfix])))
        savepath = os.path.join(
            tfr_path,
            classification_name_formatted
        )
        import_opt = dict(
                savepath=savepath + '/',
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
        
        X, Y = combiner.X, combiner.Y
        meta = mf.produce_tfrecords((X, Y), **import_opt)
        print('#'*100)
        print(np.array(meta['test_fold'][0]).shape)
        print('#'*100)
        dataset = mf.Dataset(meta, train_batch=100)
        lf_params = dict(
                n_latent=32,
                filter_length=50,
                nonlin=tf.keras.activations.elu,
                padding='SAME',
                pooling=2,
                stride=2,
                pool_type='max',
                model_path=import_opt['savepath'],
                dropout=.4,
                l2_scope=["weights"],
                l2=1e-6
        )
        
        model = LFRNN(dataset, lf_params)
        model.build()
        model.train(n_epochs=25, eval_step=100, early_stopping=5)
        network_out_path = os.path.join(subject_path, 'LFRNN')
        check_path(network_out_path)
        yp_path = os.path.join(network_out_path, 'Predictions')
        y_true_train, y_pred_train = model.predict(meta['train_paths'])
        y_true_test, y_pred_test = model.predict(meta['test_paths'])
        
        print('train-set: ', subject_name, sklearn.metrics.accuracy_score(one_hot_decoder(y_true_train), one_hot_decoder(y_pred_train)))
        print('test-set: ', subject_name, sklearn.metrics.accuracy_score(one_hot_decoder(y_true_test), one_hot_decoder(y_pred_test)))
        
        check_path(yp_path)
        save_parameters(
            Predictions(
                y_pred_test,
                y_true_test
            ),
            os.path.join(yp_path, f'{classification_name_formatted}_pred.pkl'),
            'predictions'
        )
        train_loss_, train_acc_ = model.evaluate(meta['train_paths'])
        test_loss_, test_acc_ = model.evaluate(meta['test_paths'])
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
        