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
from utils.combiners import EpochsCombiner
from utils.console import Silence
from utils.console.spinner import spinner
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
from time import perf_counter
import mneflow


SpatialParameters = namedtuple('SpatialParameters', 'patterns filters')
TemporalParameters = namedtuple('TemporalParameters', 'franges finputs foutputs fresponces')
ComponentsOrder = namedtuple('ComponentsOrder', 'l2 compwise_loss weight output_corr weight_corr')
Predictions = namedtuple('Predictions', 'y_p y_true')
WaveForms = namedtuple('WaveForms', 'evoked induced times tcs')


@spinner(prefix=lambda *args, **kwargs: f'Compute {kwargs.get("output", "patterns")}... ')
def compute_patterns(model: mf.models.BaseModel, data_path: str = None, *, output: str = 'patterns'):
    """
    Compute patterns and related data from a given model and dataset.

    This function computes patterns, spatial weights, and other related data from a trained model and dataset.

    Args:
        model (mf.models.BaseModel): The trained model from which to compute patterns.
        data_path (Optional[Union[str, list, tuple, mneflow.data.Dataset, tf.data.Dataset]], optional):
            The dataset or data path for computing patterns. If not provided, the validation dataset from the model
            will be used by default.
        output (str, optional): The type of output to compute. Options: 'patterns', 'patterns_old', or 'weights'.
            Default is 'patterns'.

    Raises:
        AttributeError: If an unsupported dataset or data path is specified.

    Returns:
        None: The function directly modifies the `model` object with computed data.
    """

    if not data_path:
        print("Computing patterns: No path specified, using validation dataset (Default)")
        ds = model.dataset.val
    elif isinstance(data_path, str) or isinstance(data_path, (list, tuple)):
        ds = model.dataset._build_dataset(
            data_path,
            split=False,
            test_batch=None,
            repeat=True
        )
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

    model.out_w_flat = model.fin_fc.w.numpy()
    model.out_weights = np.reshape(
        model.out_w_flat,
        [-1, model.dmx.size, model.out_dim]
    )
    model.out_biases = model.fin_fc.b.numpy()
    model.feature_relevances = model.compute_componentwise_loss(X, y)
    model.compwise_losses = model.feature_relevances
    # compute temporal convolution layer outputs for vis_dics
    # tc_out = model.pool(model.tconv(model.dmx(X)).numpy())
    model.lat_tcs_filt = model.tconv(model.dmx(X)).numpy()
    tc_out = model.pool(model.lat_tcs_filt)

    # compute data covariance
    X = X - tf.reduce_mean(X, axis=-2, keepdims=True)
    X = tf.transpose(X, [3, 0, 1, 2])
    X = tf.reshape(X, [X.shape[0], -1])
    model.dcov = tf.matmul(X, tf.transpose(X))

    # get spatial extraction fiter weights
    demx = model.dmx.w.numpy()
    model.lat_tcs = np.dot(demx.T, X)

    kern = np.squeeze(model.tconv.filters.numpy()).T

    X = X.numpy().T
    if 'patterns' in output:
        if 'old' in output:
            model.patterns = np.dot(model.dcov, demx)
        else:
            patterns = []
            X_filt = np.zeros_like(X)
            for i_comp in range(kern.shape[0]):
                for i_ch in range(X.shape[1]):
                    x = X[:, i_ch]
                    X_filt[:, i_ch] = np.convolve(x, kern[i_comp, :], mode="same")
                patterns.append(np.cov(X_filt.T) @ demx[:, i_comp])
            model.patterns = np.array(patterns).T
    else:
        model.patterns = demx

    del X

    #  Temporal conv stuff
    model.filters = kern.T
    model.tc_out = np.squeeze(tc_out)
    model.corr_to_output = model.get_output_correlations(y)


@spinner(prefix='Compute temporal parameters... ')
def compute_temporal_parameters(model: mf.models.BaseModel, *, fs: float = None):
    """
    This function computes temporal parameters such as frequency ranges, input powers, output powers, and filter responses
    for a given model. It utilizes the provided model's filters and latent time courses. Sampling frequency (`fs`)
    can be specified; otherwise, it is inferred from the model's dataset or set to 1 if not available.

    Args:
        model (mf.models.BaseModel): The model for which to compute temporal parameters.
        fs (Optional[float], optional): The sampling frequency of the data. If not provided, it is inferred from
            the model's dataset or set to 1 if not available.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]: A tuple containing:
            - franges (np.ndarray): The frequency ranges.
            - finputs (np.ndarray): Input powers (power spectral density).
            - foutputs (List[np.ndarray]): Output powers for each filter.
            - fresponces (List[np.ndarray]): Filter responses for each filter.
    """

    if fs is None:

        if model.dataset.h_params['fs']:
            fs = model.dataset.h_params['fs']
        else:
            print('Sampling frequency not specified, setting to 1.')
            fs = 1.

    out_filters = model.filters
    _, psd = sl.welch(model.lat_tcs, fs=fs, nperseg=fs * 2)
    finputs = psd[:, :-1]
    franges = None
    foutputs = list()
    fresponces = list()

    for i, flt in enumerate(out_filters.T):
        w, h = (lambda w, h: (w, np.abs(h)))(*sl.freqz(flt, 1, worN=fs))
        foutputs.append(np.abs(finputs[i, :] * h))

        if franges is None:
            franges = w / np.pi * fs / 2
        fresponces.append(h)

    return franges, finputs, foutputs, fresponces


def get_order(order: np.array, *args):
    if order is not None:
        return order.ravel()
    else:
        return (None, None)


def compute_morlet_cwt(
    sig: np.ndarray,
    t: np.ndarray,
    freqs: np.ndarray,
    omega_0: Optional[float] = 5,
    phase: Optional[bool] = False
) -> np.ndarray:
    """
    Compute the Continuous Wavelet Transform (CWT) using the Morlet wavelet.

    This function computes the Continuous Wavelet Transform (CWT) of a given signal using the Morlet wavelet.
    The CWT represents how the signal's frequency content evolves over time.

    Args:
        sig (np.ndarray): The input signal for which to compute the CWT.
        t (np.ndarray): The time points corresponding to the signal.
        freqs (np.ndarray): The frequencies at which to compute the CWT.
        omega_0 (Optional[float], optional): The angular frequency parameter for the Morlet wavelet.
            Defaults to 5.
        phase (Optional[bool], optional): If True, returns the complex CWT. If False, returns the power CWT.
            Defaults to False.

    Returns:
        np.ndarray: The computed CWT, either in complex form (if phase=True) or as power (real^2 + imag^2).
    """
    dt = t[1] - t[0]
    widths = omega_0 / (2 * np.pi * freqs * dt)
    cwtmatr = sl.cwt(sig, lambda M, s: sl.morlet2(M, s, w=omega_0), widths)
    if phase:
        return cwtmatr
    else:
        return np.real(cwtmatr)**2 + np.imag(cwtmatr)**2


@spinner(prefix='Compute spectral parameters... ')
def compute_waveforms(model: mf.models.BaseModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spectral parameters from the model's latent time courses.

    This function computes spectral parameters, including the time courses, induced power, and average power spectra,
    from the model's latent time courses using the Morlet wavelet transform.

    Args:
        model (mf.models.BaseModel): The machine learning model containing latent time courses.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - np.ndarray: The average induced power spectra for each latent source and frequency.
            - np.ndarray: The time points corresponding to the latent time courses.
            - np.ndarray: The latent time courses.
    """
    time_courses = np.squeeze(model.lat_tcs.reshape(
        [model.specs['n_latent'], -1, model.dataset.h_params['n_t']]
    ))
    times = (1 / float(model.dataset.h_params['fs'])) *\
        np.arange(model.dataset.h_params['n_t'])
    induced = list()

    for tc in time_courses:
        ls_induced = list()

        for lc in tc:
            freqs = np.arange(1, 71)
            ls_induced.append(np.abs(compute_morlet_cwt(lc, times, freqs)))

        induced.append(np.array(ls_induced).mean(axis=0))

    return np.array(induced), times, time_courses


def save_parameters(content: Any, path: str, parameters_type: Optional[str] = '') -> NoReturn:
    """
    Save content as parameters to a pickle file at the specified path.

    This function saves the provided content, which can be any Python object, as parameters to a pickle file
    at the specified path. It optionally allows specifying a type of parameters (e.g., 'model' or 'hyperparameters')
    for informational purposes.

    Args:
        content (Any): The Python object to be saved as parameters.
        path (str): The path to the pickle file where parameters will be saved.
        parameters_type (Optional[str]): An optional string specifying the type of parameters (e.g., 'model' or 'hyperparameters').
            This is for informational purposes and will be included in the print message.

    Raises:
        OSError: If the specified path does not have a '.pkl' extension.
    """
    parameters_type = parameters_type + ' ' if parameters_type else parameters_type
    print(f'Saving {parameters_type}parameters...')

    if path[-4:] != '.pkl':
        raise OSError(f'Pickle file must have extension ".pkl", but it has "{path[-4:]}"')

    pickle.dump(content, open(path, 'wb'))

    print('Successfully saved')


def save_model_weights(model: mf.models.BaseModel, path: str) -> NoReturn:
    """
    Save the weights of a machine learning model to an HDF5 file.

    This function saves the weights of a machine learning model to an HDF5 file at the specified path.

    Args:
        model (mf.models.BaseModel): The machine learning model whose weights will be saved.
        path (str): The path to the HDF5 file where model weights will be saved.

    Raises:
        OSError: If the specified path does not have a '.h5' extension.
    """
    print('Saving model weights')

    if path[-3:] != '.h5':
        raise OSError(f'File must have extension ".h5", but it has "{path[-3:]}"')

    model.km.save_weights(path, overwrite=True)

    print('Successfully saved')


if __name__ == '__main__':
    mpl.use('agg')
    parser = argparse.ArgumentParser(
        description='A script for applying the neural network "LFCNN" '
        'to the epoched data from gradiometers related to events for classification'
    )
    parser.add_argument('-es', '--exclude-sessions', type=str, nargs='+',
                        default=[], help='Sessions to exclude')
    parser.add_argument('-ep', '--exclude-participants', type=str, nargs='+',
                        default=[], help='IDs of subjects to exclude')
    parser.add_argument('-l', '--lock', type=str,
                        default='RespCor', help='Stimulus lock to consider')
    parser.add_argument('-c', '--cases', type=str, nargs='+',
                        default=['LI', 'LM', 'RI', 'RM'],
                        help='Cases to consider (must match epochs file names '
                        'for the respective classes)')
    parser.add_argument('-cmb', '--combine-cases', type=str, nargs='+',
                        default=None, help='Cases to consider (must be the number of '
                        'strings in which classes to combine are written separated by '
                        'a space, indices corresponds to order of "--cases" parameter)')
    parser.add_argument('-sd', '--subjects-dir', type=str,
                        default=os.path.join(os.getcwd(), 'Source', 'Subjects'),
                        help='Path to the subjects directory')
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
    parser.add_argument('-hp', '--high_pass', type=float,
                        default=None, help='High-pass filter (Hz)')
    parser.add_argument('--no-params', action='store_true', help='Do not compute parameters')
    parser.add_argument('-cf', '--crop-from', type=float, help='Crop epoch from time', default=None)
    parser.add_argument('-ct', '--crop-to', type=float, help='Crop epoch to time', default=None)

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
        project_name, \
        lfreq, \
        no_params, \
        crop_from, crop_to = vars(parser.parse_args()).values()

    if excluded_sessions:
        excluded_sessions = [
            sessions_name + session
            if sessions_name not in session
            else session
            for session in excluded_sessions
        ]

    cases_to_combine = [
        case.split(' ')
        for case in cases
    ] if cases_to_combine is None else [
        case.split(' ')
        for case in cases_to_combine
    ]
    cases = list(filter(lambda case: any([case in cmb for cmb in cases_to_combine]), cases))
    cases_to_combine = sorted(cases_to_combine, reverse=True)
    class_names = [
        '&'.join(sorted(cases_combination, reverse=True))
        for cases_combination in cases_to_combine
    ]

    if classification_name is None:
        classification_name = '_vs_'.join(class_names)

    classification_name_formatted = "_".join(list(filter(
        lambda s: s not in (None, ""),
        [
            classification_prefix,
            classification_name,
            classification_postfix
        ]
    )))

    perf_tables_path = os.path.join(os.path.dirname(subjects_dir), 'perf_tables')
    pics_path = os.path.join(os.path.dirname(subjects_dir), 'Pictures')
    check_path(perf_tables_path, pics_path)

    for subject_name in os.listdir(subjects_dir):

        if subject_name in excluded_subjects:
            continue

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

                if lfreq is not None:
                    epochs[case] = epochs[case].filter(lfreq, None)

                if crop_from is not None or crop_to is not None:
                    epochs[case] = epochs[case].crop(crop_from, crop_to)

                cases_to_combine_list.append(epochs[case])

            i += 1

        combiner = EpochsCombiner(*cases_to_combine_list).combine(*cases_indices_to_combine)
        n_classes, classes_samples = np.unique(combiner.Y, return_counts=True)
        n_classes = len(n_classes)
        classes_samples = classes_samples.tolist()
        combiner.shuffle()
        tfr_path = os.path.join(subject_path, 'TFR')
        check_path(tfr_path)
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
            crop_baseline=False,
            scale_interval=(0, 60),
            decimate=None,
            n_folds=5,
            overwrite=True,
            segment=False,
            test_set='holdout'
        )

        X, Y = combiner.X, combiner.Y
        meta = mf.produce_tfrecords((X, Y), **import_opt)
        dataset = mf.Dataset(meta, train_batch=100)
        lf_params = dict(
            n_latent=4,
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
        t1 = perf_counter()
        model.train(n_epochs=25, eval_step=100, early_stopping=5)
        print('#' * 100)
        runtime = perf_counter() - t1
        print(f'{classification_name_formatted}\nLFCNN\nruntime: {runtime}')
        print('#' * 100)
        network_out_path = os.path.join(subject_path, 'LFCNN')
        yp_path = os.path.join(network_out_path, 'Predictions')
        sp_path = os.path.join(network_out_path, 'Parameters')
        check_path(network_out_path, yp_path, sp_path)
        y_true_train, y_pred_train = model.predict(meta['train_paths'])
        y_true_test, y_pred_test = model.predict(meta['test_paths'])

        print(
            'train-set: ',
            subject_name,
            sklearn.metrics.accuracy_score(
                one_hot_decoder(
                    y_true_train
                ),
                one_hot_decoder(
                    y_pred_train
                )
            )
        )
        print(
            'test-set: ',
            subject_name,
            sklearn.metrics.accuracy_score(
                one_hot_decoder(
                    y_true_test
                ),
                one_hot_decoder(
                    y_pred_test
                )
            )
        )

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

        if not no_params:
            compute_patterns(model, meta['train_paths'], output='patterns old')
            old_patterns = model.patterns.copy()
            compute_patterns(model, meta['train_paths'])
            nt = model.dataset.h_params['n_t']
            time_courses = np.squeeze(model.lat_tcs.reshape([model.specs['n_latent'], -1, nt]))
            times = (1 / float(model.dataset.h_params['fs'])) *\
                np.arange(model.dataset.h_params['n_t'])
            patterns = model.patterns.copy()
            compute_patterns(model, meta['train_paths'], output='filters')
            filters = model.patterns.copy()
            franges, finputs, foutputs, fresponces = compute_temporal_parameters(model)
            induced, times, time_courses = compute_waveforms(model)

            # induced = list()
            # for tc in time_courses:
            #     ls_induced = list()
            #     for lc in tc:
            #         widths = np.arange(1, 71)
            #         ls_induced.append(np.abs(sp.signal.cwt(lc, sp.signal.ricker, widths)))
            #     induced.append(np.array(ls_induced).mean(axis=0))
            # induced = np.array(induced)

            save_parameters(
                WaveForms(time_courses.mean(1), induced, times, time_courses),
                os.path.join(sp_path, f'{classification_name_formatted}_waveforms.pkl'),
                'WaveForms'
            )
            save_parameters(
                SpatialParameters(old_patterns, filters),
                os.path.join(sp_path, f'{classification_name_formatted}_spatial_old.pkl'),
                'spatial'
            )
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

        perf_table_path = os.path.join(
            perf_tables_path,
            f'{classification_name_formatted}.csv'
        )
        processed_df = pd.Series(
            [
                n_classes,
                *classes_samples,
                sum(classes_samples),
                np.array(meta['test_fold'][0]).shape[0],
                train_acc_,
                train_loss_,
                test_acc_,
                test_loss_,
                model.v_metric,
                model.v_loss,
                runtime
            ],
            index=[
                'n_classes',
                *class_names,
                'total',
                'test_set',
                'train_acc',
                'train_loss',
                'test_acc',
                'test_loss',
                'val_acc',
                'val_loss',
                'runtime'
            ],
            name=subject_name
        ).to_frame().T

        if os.path.exists(perf_table_path):
            pd.concat([pd.read_csv(perf_table_path, index_col=0, header=0), processed_df], axis=0)\
                .to_csv(perf_table_path)
        else:
            processed_df.to_csv(perf_table_path)
