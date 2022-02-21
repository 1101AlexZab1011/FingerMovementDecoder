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


SpatialParameters = namedtuple('SpatialParameters', 'patterns filters')
TemporalParameters = namedtuple('TemporalParameters', 'franges finputs foutputs fresponces')
ComponentsOrder = namedtuple('ComponentsOrder', 'l2 compwise_loss weight output_corr weight_corr')
Predictions = namedtuple('Predictions', 'y_p y_true')

def compute_temporal_parameters(model, *, fs=None):
    
    if fs is None:
        
        if model.dataset.h_params['fs']:
            fs = model.dataset.h_params['fs']
        else:
            print('Sampling frequency not specified, setting to 1.')
            fs = 1.

    out_filters = model.filters
    _, psd = sl.welch(model.lat_tcs, fs=fs, nperseg=fs*2)
    finputs = psd[:, :-1]
    franges = None
    foutputs = list()
    fresponces = list()
    
    for i, flt in enumerate(out_filters.T):
        w, h = (lambda w, h: (w, np.abs(h)))(*sl.freqz(flt, 1, worN=fs))
        foutputs.append(np.abs(finputs[i, :]*h))
        
        if franges is None:
            franges = w/np.pi*fs/2
        fresponces.append(h)
        
    return franges, finputs, foutputs, fresponces


def save_parameters(content: Any, path: str, parameters_type: Optional[str] = '') -> NoReturn:
    
    parameters_type = parameters_type + ' ' if parameters_type else parameters_type
    print(f'Saving {parameters_type}parameters...')
    
    if path[-4:] != '.pkl':
        raise OSError(f'Pickle file must have extension ".pkl", but it has "{path[-4:]}"')
    
    pickle.dump(content, open(path, 'wb'))
    
    print('Successfully saved')


def save_model_weights(model: mf.models.BaseModel, path: str) -> NoReturn:
    
    print('Saving model weights')
    
    if path[-3:] != '.h5':
        raise OSError(f'File must have extension ".h5", but it has "{path[-3:]}"')
    
    model.km.save_weights(path, overwrite=True)
    
    print('Successfully saved')


def plot_waveforms(model, sorting='compwise_loss', tmin=0, class_names=None):
    
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
            model.waveforms = np.squeeze(model.lat_tcs.reshape([model.specs['n_latent'], -1, nt]).mean(1))
            tstep = 1/float(fs)
            times = tmin + tstep*np.arange(nt)
            scaling = 3*np.mean(np.std(model.waveforms, -1))
            [ax[0, 0].plot(times, wf + scaling*i) for i, wf in enumerate(model.waveforms) if i not in model.uorder]
            ax[0, 0].plot(times, model.waveforms[uo] + scaling*uo, 'k', linewidth=5.)
            ax[0, 0].set_title('Latent component waveforms')
            bias = model.tconv.b.numpy()[uo]
            ax[0, 1].stem(model.filters.T[uo], use_line_collection=True)
            ax[0, 1].hlines(bias, 0, len(model.filters.T[uo]), linestyle='--', label='Bias')
            ax[0, 1].legend()
            ax[0, 1].set_title('Filter coefficients')
            conv = np.convolve(model.filters.T[uo], model.waveforms[uo], mode='same')
            vmin = conv.min()
            vmax = conv.max()
            ax[1, 0].plot(times + 0.5*model.specs['filter_length']/float(fs), conv)
            tstep = float(model.specs['stride'])/fs
            strides = np.arange(times[0], times[-1] + tstep/2, tstep)[1:-1]
            pool_bins = np.arange(times[0], times[-1] + tstep, model.specs['pooling']/fs)[1:]
            ax[1, 0].vlines(strides, vmin, vmax, linestyle='--', color='c', label='Strides')
            ax[1, 0].vlines(pool_bins, vmin, vmax, linestyle='--', color='m', label='Pooling')
            ax[1, 0].set_xlim(times[0], times[-1])
            ax[1, 0].legend()
            ax[1, 0].set_title('Convolution output')
            strides1 = np.linspace(times[0], times[-1]+tstep/2, model.F.shape[1])
            ax[1, 1].pcolor(strides1, np.arange(model.specs['n_latent']), model.F)
            ax[1, 1].hlines(uo, strides1[0], strides1[-1], color='r')
            ax[1, 1].set_title('Feature relevance map')
            
            if class_names:
                comp_name = class_names[jj]
            else:
                comp_name = "Class " + str(jj)
                
            f.suptitle(comp_name, fontsize=16)
            
        return f

def plot_patterns(patterns, info, order=None, cmap='RdBu_r', sensors=True,
                colorbar=False, res=64,
                size=1, cbar_fmt='%3.1f', name_format='Latent\nSource %01d',
                show=True, show_names=False, title=None,
                outlines='head', contours=6,
                image_interp='bilinear'):
    
    if not title:
        title=f'All patterns'
        
    n_components = patterns.shape[1]
    info = copy.deepcopy(info)
    info['sfreq'] = 1.
    patterns = mne.EvokedArray(patterns, info, tmin=0)
    
    order = range(n_components) if order is None else order
    
    return patterns.plot_topomap(
        times=order,
        cmap=cmap, colorbar=colorbar, res=res,
        cbar_fmt=cbar_fmt, sensors=sensors, units=None, time_unit='s',
        time_format=name_format, size=size, show_names=show_names,
        title=title, outlines=outlines,
        contours=contours, image_interp=image_interp, show=show)


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
    # cases_to_combine = sorted(cases_to_combine, reverse=True)
    class_names = ['&'.join(sorted(cases_combination, reverse=True)) for cases_combination in cases_to_combine]
    
    if classification_name is None:
        classification_name = '_vs_'.join(class_names)
    
    perf_tables_path = os.path.join(os.path.dirname(subjects_dir), 'perf_tables')
    check_path(perf_tables_path)
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
                i += j
                cases_indices_to_combine[-1].append(i)
                cases_to_combine_list.append(epochs[case])
                
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
            )
        
        X, Y = combiner.X, combiner.Y
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=.8)
        X_test = np.transpose(np.expand_dims(X_test, axis = 1), (0, 1, 3, 2))
        meta = mf.produce_tfrecords((X_train, y_train), **import_opt)
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
        yp_path = os.path.join(subject_path, 'Predictions')
        y_true_train, y_pred_train = model.predict(model.dataset.train)
        y_true_val, y_pred_val = model.predict()
        
        print('test-set: ', subject_name, sklearn.metrics.accuracy_score(one_hot_decoder(y_true_train), one_hot_decoder(y_pred_train)))
        print('val-set: ', subject_name, sklearn.metrics.accuracy_score(one_hot_decoder(y_true_val), one_hot_decoder(y_pred_val)))
        
        check_path(yp_path)
        save_parameters(
            Predictions(
                np.concatenate([y_pred_train, y_pred_val], axis=0),
                np.concatenate([y_true_train, y_true_val], axis=0),
            ),
            os.path.join(yp_path, f'{classification_name_formatted}_pred.pkl'),
            'predictions'
        )
        train_loss_, train_acc_ = model.evaluate(meta['train_paths'])
        test_loss_, test_acc_ = model.evaluate(meta['test_paths'])
        model.compute_patterns(meta['train_paths'])
        patterns = model.patterns.copy()
        model.compute_patterns(meta['train_paths'], output='filters')
        filters = model.patterns.copy()
        franges, finputs, foutputs, fresponces = compute_temporal_parameters(model)
        
        sp_path = os.path.join(subject_path, 'Parameters')
        check_path(sp_path)
        
        save_parameters(
            SpatialParameters(patterns, filters),
            os.path.join(sp_path, f'{classification_name_formatted}_spatial.pkl'),
            'spatial'
        )
        save_parameters(
            TemporalParameters(franges, finputs, foutputs, fresponces),
            os.path.join(sp_path, f'{classification_name_formatted}_temporal.pkl'),
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
            os.path.join(sp_path, f'{classification_name_formatted}_sorting.pkl'),
            'sorting'
        )
        
        pics_path = os.path.join(os.path.dirname(subjects_dir), 'Pictures')
        patterns_pics_path = os.path.join(pics_path, 'Patterns', classification_name_formatted)
        filters_pics_path = os.path.join(pics_path, 'Filters', classification_name_formatted)
        spectra_pics_path = os.path.join(pics_path, 'Spectra', classification_name_formatted)
        wf_pics_path = os.path.join(pics_path, 'WaveForms', classification_name_formatted)
        loss_pics_path = os.path.join(pics_path, 'Loss', classification_name_formatted)
        acc_pics_path = os.path.join(pics_path, 'Accuracy', classification_name_formatted)
        
        check_path(
            pics_path,
            os.path.join(pics_path, 'Patterns'),
            os.path.join(pics_path, 'Filters'),
            os.path.join(pics_path, 'Spectra'),
            os.path.join(pics_path, 'WaveForms'),
            os.path.join(pics_path, 'Loss'),
            os.path.join(pics_path, 'Accuracy'),
            patterns_pics_path,
            filters_pics_path,
            spectra_pics_path,
            wf_pics_path,
            loss_pics_path,
            acc_pics_path
        )
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
        
        weights_path = os.path.join(subject_path, 'Weights')
        models_path = os.path.join(subject_path, 'Models')
        check_path(weights_path, models_path)
        save_model_weights(
            model,
            os.path.join(
                weights_path,
                f'{classification_name_formatted}.h5'
            )
        )
        model.km.save(os.path.join(models_path, f'{classification_name_formatted}.h5'))
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
        