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
from LFCNN_decoder import SpatialParameters, TemporalParameters,\
    ComponentsOrder, Predictions, WaveForms,\
    compute_temporal_parameters, save_parameters
from mneflow.models import BaseModel, LFCNN
from utils.machine_learning.designer import ModelDesign, ParallelDesign, LayerDesign
from utils.machine_learning.analyzer import ModelAnalyzer
from mneflow.layers import DeMixing, LFTConv, TempPooling, Dense
from scipy.signal import freqz, welch
from scipy.stats import spearmanr


class LFRNN(BaseModel):
    def __init__(self, Dataset, specs=dict()):
        self.scope = 'lfcnn'
        specs.setdefault('filter_length', 7)
        specs.setdefault('n_latent', 32)
        specs.setdefault('pooling', 2)
        specs.setdefault('stride', 2)
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
        print(self.inputs.shape, self.inputs.shape[3])
        self.design = ModelDesign(
            self.inputs,
            DeMixing(size=self.specs['n_latent'], nonlin=tf.identity, axis=3, specs=self.specs),
            LayerDesign(tf.expand_dims, axis=1),
            LFTConv(
                    size=self.specs['n_latent'],
                    nonlin=self.specs['nonlin'],
                    filter_length=self.specs['filter_length']*2,
                    padding=self.specs['padding'],
                    specs=self.specs
                ),
            tf.keras.layers.DepthwiseConv2D((1, 200), padding='valid', activation='relu', kernel_regularizer='l1'),
            tf.keras.layers.Dropout(self.specs['dropout'], noise_shape=None),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.out_dim, kernel_regularizer='l1'),
        )

        return self.design()
    
    def _get_spatial_covariance(self, dataset):
        n1_covs = []
        for x, y in dataset.take(5):
            n1cov = tf.tensordot(x[0,0], x[0,0], axes=[[0], [0]])
            n1_covs.append(n1cov)

        cov = tf.reduce_mean(tf.stack(n1_covs, axis=0), axis=0)

        return cov
    
    def get_component_relevances(self, X, y):
        model_weights = self.km.get_weights()
        base_loss, base_performance = self.km.evaluate(X, y, verbose=0)
        #if len(base_performance > 1):
        #    base_performance = bbase_performance[0]
        feature_relevances_loss = []
        n_out_t = self.out_weights.shape[0]
        n_out_y = self.out_weights.shape[-1]
        zeroweights = np.zeros((n_out_t,))
        for i in range(self.specs["n_latent"]):
            loss_per_class = []
            for jj in range(n_out_y):
                new_weights = self.out_weights.copy()
                new_bias = self.out_biases.copy()
                new_weights[:, i, jj] = zeroweights
                new_bias[jj] = 0
                new_weights = np.reshape(new_weights, self.out_w_flat.shape)
                model_weights[-2] = new_weights
                model_weights[-1] = new_bias
                self.km.set_weights(model_weights)
                loss = self.km.evaluate(X, y, verbose=0)[0]
                loss_per_class.append(base_loss - loss)
            feature_relevances_loss.append(np.array(loss_per_class))
            self.component_relevance_loss = np.array(feature_relevances_loss)
    
    
    def get_output_correlations(self, y_true):
        corr_to_output = []
        y_true = y_true.numpy()
        flat_feats = self.tc_out.reshape(self.tc_out.shape[0], -1)

        if self.dataset.h_params['target_type'] in ['float', 'signal']:
            for y_ in y_true.T:

                rfocs = np.array([spearmanr(y_, f)[0] for f in flat_feats.T])
                corr_to_output.append(rfocs.reshape(self.out_weights.shape[:-1]))

        elif self.dataset.h_params['target_type'] == 'int':
            y_true = y_true/np.linalg.norm(y_true, ord=1, axis=0)[None, :]
            flat_div = np.linalg.norm(flat_feats, 1, axis=0)[None, :]
            flat_feats = flat_feats/flat_div
            #print("ff:", flat_feats.shape)
            #print("y_true:", y_true.shape)
            for y_ in y_true.T:
                #print('y.T:', y_.shape)
                rfocs = 2. - np.sum(np.abs(flat_feats - y_[:, None]), 0)
                corr_to_output.append(rfocs.reshape(self.out_weights.shape[:-1]))

        corr_to_output = np.dstack(corr_to_output)

        if np.any(np.isnan(corr_to_output)):
            corr_to_output[np.isnan(corr_to_output)] = 0
        return corr_to_output
    
    
    def compute_patterns(self, data_path=None, output='patterns'):
        #vis_dict = None
        if not data_path: 
            print("Computing patterns: No path specified, using validation dataset (Default)")
            ds = self.dataset.val
        elif isinstance(data_path, str) or isinstance(data_path, (list, tuple)):
            ds = self.dataset._build_dataset(data_path, 
                                            split=False, 
                                            test_batch=None, 
                                            repeat=True)
        elif isinstance(data_path, mneflow.data.Dataset):
            if hasattr(data_path, 'test'):
                ds = data_path.test
            else:
                ds = data_path.val
        elif isinstance(data_path, tf.data.Dataset):
            ds = data_path
        else:
            raise AttributeError('Specify dataset or data path.')

        X, y = [row for row in ds.take(1)][0]

        self.fin_fc = self.design[-1]
        self.dmx_size = self.design[1].forward_layer.weights[1].shape[0]
        self.dmx = self.design[1]
        self.tconv = self.design[3]
        self.pool = self.design[4]
        
        self.out_w_flat = self.fin_fc.w.numpy()
        self.out_weights = np.reshape(self.out_w_flat, [-1, self.dmx_size,
                                            self.out_dim])
        self.out_biases = self.fin_fc.b.numpy()
        self.feature_relevances = self.get_component_relevances(X, y)
        
        #compute temporal convolution layer outputs for vis_dics
        # tc_out = self.pool(self.tconv(self.design[1](self.dmx(self.design[0](X)))).numpy())
        tc_out = ModelDesign(None, *model.design[:5])(X).numpy()

        #compute data covariance
        X = X - tf.reduce_mean(X, axis=-2, keepdims=True)
        X = tf.transpose(X, [3, 0, 1, 2])
        X = tf.reshape(X, [X.shape[0], -1])
        self.dcov = tf.matmul(X, tf.transpose(X))

        # get spatial extraction fiter weights
        
        fw, fa, fb = (w.numpy() for w in self.design[1].forward_layer.weights)
        bw, ba, bb = (w.numpy() for w in self.design[1].backward_layer.weights)

        # fwd + bcwd
        # (kernel weights + kernel biases) * recurrent weights
        demx = (((fw + fb)@fa.T) + ((bw + bb)@ba.T))
        self.lat_tcs = np.dot(demx.T, X)
        del X

        if 'patterns' in output:
            self.patterns = np.dot(self.dcov, demx)

        else:
            self.patterns = demx

        kern = self.tconv.filters.numpy()
        self.filters = np.squeeze(kern)
        self.tc_out = np.squeeze(tc_out)
        self.corr_to_output = self.get_output_correlations(y)


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
                savepath=savepath+'/',
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
                pooling=10,
                stride=10,
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
        # model.compute_patterns(meta['train_paths'])
        # nt = model.dataset.h_params['n_t']
        # time_courses = np.squeeze(model.lat_tcs.reshape([model.specs['n_latent'], -1, nt]))
        # times = (1/float(model.dataset.h_params['fs']))*np.arange(model.dataset.h_params['n_t'])
        # patterns = model.patterns.copy()
        # model.compute_patterns(meta['train_paths'], output='filters')
        # filters = model.patterns.copy()
        # franges, finputs, foutputs, fresponces = compute_temporal_parameters(model)
        
        # sp_path = os.path.join(network_out_path, 'Parameters')
        # check_path(sp_path)
        
        # induced = list()
        # for tc in time_courses:
        #     ls_induced = list()
        #     for lc in tc:
        #         widths = np.arange(1, 71)
        #         ls_induced.append(np.abs(sp.signal.cwt(lc, sp.signal.ricker, widths)))
        #     induced.append(np.array(ls_induced).mean(axis=0))
        # induced = np.array(induced)
        
        # save_parameters(
        #     WaveForms(time_courses.mean(1), induced, times, time_courses),
        #     os.path.join(sp_path, f'{classification_name_formatted}_waveforms.pkl'),
        #     'WaveForms'
        # )
        
        # save_parameters(
        #     SpatialParameters(patterns, filters),
        #     os.path.join(sp_path, f'{classification_name_formatted}_spatial.pkl'),
        #     'spatial'
        # )
        # save_parameters(
        #     TemporalParameters(franges, finputs, foutputs, fresponces),
        #     os.path.join(sp_path, f'{classification_name_formatted}_temporal.pkl'),
        #     'temporal'
        # )
        # get_order = lambda order, ts: order.ravel()
        # save_parameters(
        #     ComponentsOrder(
        #         get_order(*model._sorting('l2')),
        #         get_order(*model._sorting('compwise_loss')),
        #         get_order(*model._sorting('weight')),
        #         get_order(*model._sorting('output_corr')),
        #         get_order(*model._sorting('weight_corr')),
        #     ),
        #     os.path.join(sp_path, f'{classification_name_formatted}_sorting.pkl'),
        #     'sorting'
        # )
        
        # pics_path = os.path.join(os.path.dirname(subjects_dir), 'Pictures')
        # patterns_pics_path = os.path.join(pics_path, 'Patterns', classification_name_formatted)
        # filters_pics_path = os.path.join(pics_path, 'Filters', classification_name_formatted)
        # spectra_pics_path = os.path.join(pics_path, 'Spectra', classification_name_formatted)
        # wf_pics_path = os.path.join(pics_path, 'WaveForms', classification_name_formatted)
        # loss_pics_path = os.path.join(pics_path, 'Loss', classification_name_formatted)
        # acc_pics_path = os.path.join(pics_path, 'Accuracy', classification_name_formatted)
        
        # check_path(
        #     pics_path,
        #     os.path.join(pics_path, 'Patterns'),
        #     os.path.join(pics_path, 'Filters'),
        #     os.path.join(pics_path, 'Spectra'),
        #     os.path.join(pics_path, 'WaveForms'),
        #     os.path.join(pics_path, 'Loss'),
        #     os.path.join(pics_path, 'Accuracy'),
        #     patterns_pics_path,
        #     filters_pics_path,
        #     spectra_pics_path,
        #     wf_pics_path,
        #     loss_pics_path,
        #     acc_pics_path
        # )
        # plt.plot(model.t_hist.history['loss'])
        # plt.plot(model.t_hist.history['val_loss'])
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'validation'], loc='upper left')
        # plt.savefig(os.path.join(loss_pics_path, f'{subject_name}_{classification_name_formatted}.png'))
        # plt.close()
        # plt.plot(model.t_hist.history['cat_ACC'])
        # plt.plot(model.t_hist.history['val_cat_ACC'])
        # plt.title('model acc')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'validation'], loc='upper left')
        # plt.savefig(os.path.join(acc_pics_path, f'{subject_name}_{classification_name_formatted}.png'))
        # plt.close()
        
        # patterns_fig = plot_patterns(patterns, any_info)
        # patterns_fig.savefig(os.path.join(patterns_pics_path, f'{subject_name}_{classification_name_formatted}.png'))
        # plt.close(patterns_fig)
        # filters_fig = plot_patterns(filters, any_info)
        # filters_fig.savefig(os.path.join(filters_pics_path, f'{subject_name}_{classification_name_formatted}.png'))
        # plt.close(filters_fig)
        # spectra_fig = model.plot_spectra(sorting='weight_corr', class_names=class_names)
        # spectra_fig.savefig(os.path.join(spectra_pics_path, f'{subject_name}_{classification_name_formatted}.png'))
        # plt.close(spectra_fig)
        # wf_fig = plot_waveforms(model, class_names=class_names)
        # wf_fig.savefig(os.path.join(wf_pics_path, f'{subject_name}_{classification_name_formatted}.png'))
        # plt.close(wf_fig)
        
        # weights_path = os.path.join(subject_path, 'Weights')
        # models_path = os.path.join(subject_path, 'Models')
        # check_path(weights_path, models_path)
        # save_model_weights(
        #     model,
        #     os.path.join(
        #         weights_path,
        #         f'{classification_name_formatted}.h5'
        #     )
        # )
        # model.km.save(os.path.join(models_path, f'{classification_name_formatted}.h5'))
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
        