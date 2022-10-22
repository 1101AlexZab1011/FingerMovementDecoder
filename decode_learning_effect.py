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
from utils.storage_management import check_path
from utils.machine_learning import one_hot_decoder
from utils.machine_learning.designer import ModelDesign, LayerDesign
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
from models import SimpleNet, ZubarevNetFactory
from LFCNN_decoder import SpatialParameters,\
    TemporalParameters,\
    ComponentsOrder,\
    Predictions,\
    WaveForms,\
    save_parameters,\
    compute_patterns,\
    compute_waveforms,\
    compute_temporal_parameters,\
    get_order
from mneflow.layers import Dense, LFTConv, TempPooling, DeMixing
from tensorflow.keras.initializers import Constant
from tensorflow.keras import regularizers as k_reg


SessionInfo = namedtuple(
    'SessionInfo',
    'cases_cmb class_names classification_name classification_name_formatted model_name'
)

def make_session_info(
    cases: list[str],
    cases_to_combine: Optional[list[str]] = None,
    classification_name: Optional[str] = None,
    classification_prefix: Optional[str] = None,
    classification_postfix: Optional[str] = None,
    model_name: Optional[str] = 'unknown'
) -> SessionInfo:
    cases_to_combine = [case.split(' ') for case in cases] if cases_to_combine is None else [
        case.split(' ') for case in cases_to_combine
    ]
    cases = list(filter(lambda case: any([case in cmb for cmb in cases_to_combine]), cases))
    cases_to_combine = sorted(cases_to_combine, reverse=True)
    class_names = [
        '&'.join(sorted(cases_combination, reverse=True))
        for cases_combination in cases_to_combine
    ]
    if classification_name is None:
        classification_name = '_vs_'.join(class_names)

    classification_name_formatted = "_".join(
        list(
            filter(
                lambda s: s not in (None, ""),
                [classification_prefix, classification_name, classification_postfix]
            )
        )
    )

    return SessionInfo(
        cases_to_combine,
        class_names,
        classification_name,
        classification_name_formatted,
        model_name
    )


def select_design(design_name: str) -> ModelDesign:
    match design_name:
        case 'eegnet':
            return lambda dataset, specs: ModelDesign(
                None,
                LayerDesign(tf.transpose, [0, 3, 2, 1]),
                tf.keras.layers.Conv2D(
                    specs['n_latent'],
                    (2, specs['filter_length']),
                    padding=specs['padding'],
                    use_bias=False
                ),
                tf.keras.layers.BatchNormalization(axis=1),
                tf.keras.layers.DepthwiseConv2D(
                    (dataset.h_params['n_ch'], 1),
                    use_bias=False,
                    depth_multiplier=1,
                    depthwise_constraint=tf.keras.constraints.MaxNorm(1.)
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('elu'),
                tf.keras.layers.AveragePooling2D((1, specs['pooling'])),
                tf.keras.layers.Dropout(specs['dropout']),
                tf.keras.layers.SeparableConv2D(
                    specs['n_latent'],
                    (1, specs['filter_length'] // specs["pooling"]),
                    use_bias=False,
                    padding='same'
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('elu'),
                tf.keras.layers.AveragePooling2D((1, specs['pooling'] * 2)),
                tf.keras.layers.Dropout(specs['dropout']),
                Dense(size=np.prod(dataset.h_params['y_shape']))
            )
        case 'fbcsp':
            return lambda dataset, specs: ModelDesign(
                None,
                LayerDesign(tf.transpose, [0, 3, 2, 1]),
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=(1, specs['filter_length']),
                    depth_multiplier=specs['n_latent'],
                    strides=1,
                    padding="VALID",
                    activation=tf.identity,
                    kernel_initializer="he_uniform",
                    bias_initializer=Constant(0.1),
                    data_format="channels_last",
                    kernel_regularizer=k_reg.l2(specs['l2'])
                    # kernel_constraint="maxnorm"
                ),
                tf.keras.layers.Conv2D(
                    filters=specs['n_latent'],
                    kernel_size=(dataset.h_params['n_ch'], 1),
                    strides=1,
                    padding="VALID",
                    activation=tf.square,
                    kernel_initializer="he_uniform",
                    bias_initializer=Constant(0.1),
                    data_format="channels_last",
                    kernel_regularizer=k_reg.l2(specs['l2'])
                ),
                TempPooling(
                    pooling=specs['pooling'],
                    pool_type="avg",
                    stride=specs['stride'],
                    padding='SAME',
                ),
                Dense(size=np.prod(dataset.h_params['y_shape']), nonlin=tf.identity)
            )
        case 'deep4' | 'vgg19':
            return lambda dataset, specs: ModelDesign(
                None,
                LayerDesign(tf.transpose, [0, 3, 2, 1]),
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=(1, specs['filter_length']),
                    depth_multiplier=specs['n_latent'],
                    strides=1,
                    padding=specs['padding'],
                    activation=tf.identity,
                    kernel_initializer="he_uniform",
                    bias_initializer=Constant(0.1),
                    data_format="channels_last",
                    kernel_regularizer=k_reg.l2(specs['l2'])
                    # kernel_constraint="maxnorm"
                ),
                *[ModelDesign(
                    tf.keras.layers.Conv2D(
                        filters=specs['n_latent'],
                        kernel_size=(dataset.h_params['n_ch'], 1),
                        strides=1,
                        padding=specs['padding'],
                        activation=specs['nonlin'],
                        kernel_initializer="he_uniform",
                        bias_initializer=Constant(0.1),
                        data_format="channels_last",
                        # data_format="channels_first",
                        kernel_regularizer=k_reg.l2(specs['l2'])
                    ),
                    TempPooling(
                        pooling=specs['pooling'],
                        pool_type="avg",
                        stride=specs['stride'],
                        padding='SAME',
                    )
                ) for _ in range(4)],
                Dense(size=np.prod(dataset.h_params['y_shape']), nonlin=tf.nn.softmax)
            )
        case _:
            raise NotImplementedError(f'Model design {design_name} is not implemented')

def select_model(model_name: str) -> mneflow.models.BaseModel:
    match model_name:
        case 'LFCNN':
            return mneflow.models.LFCNN
        case 'simplenet':
            return SimpleNet
        case _:
            return ZubarevNetFactory(select_design(model_name))


class StorageManager:
    def __init__(self, subjects_dir: str, session_info: SessionInfo):
        self.subjects_dir = subjects_dir
        self.subject_dirs = os.listdir(subjects_dir)
        self.session_info = session_info
        self.perf_tables_path = os.path.join(os.path.dirname(self.subjects_dir), 'perf_tables')
        self.subject_path = None
        self.tfr_path = None
        self.classification_path = None
        self.network_path = None
        self.network_out_path = None
        self.predictions_path = None
        self.parameters_path = None
        self.__current_subject_index = 0

    def select_subject(self, subject_name: str):
        self.subject_path = os.path.join(self.subjects_dir, subject_name)
        self.epochs_path = os.path.join(self.subject_path, 'Epochs')

        if not len(os.listdir(self.epochs_path)):
            raise OSError(f'Epochs for {subject_name} not found')

        self.tfr_path = os.path.join(self.subject_path, 'TFR')
        self.classification_path = os.path.join(
            self.tfr_path,
            self.session_info.classification_name_formatted
        )
        self.network_path = os.path.join(
            self.subject_path,
            f'{self.session_info.model_name}',
        )
        self.network_out_path = os.path.join(
            self.network_path,
            self.session_info.classification_name_formatted
        )
        self.predictions_path = os.path.join(
            self.network_out_path,
            'Predictions'
        )
        self.parameters_path = os.path.join(
            self.network_out_path,
            'Parameters'
        )
        check_path(
            self.tfr_path,
            self.classification_path,
            self.network_path,
            self.network_out_path,
            self.predictions_path,
            self.parameters_path
        )

    def __iter__(self):
        self.__current_subject_index = 0
        return self

    def __next__(self):
        if self.__current_subject_index < len(self.subject_dirs):
            self.select_subject(self.subject_dirs[self.__current_subject_index])
            subject_name = self.subject_dirs[self.__current_subject_index]
            self.__current_subject_index += 1
            return subject_name
        else:
            raise StopIteration

    @property
    def subjects_dir(self):
        return self._subjects_dir
    @subjects_dir.setter
    def subjects_dir(self, value):
        self._subjects_dir = value
        check_path(self.subjects_dir)

    @property
    def perf_tables_path(self):
        return self._perf_tables_path
    @perf_tables_path.setter
    def perf_tables_path(self, value):
        self._perf_tables_path = value
        check_path(self.perf_tables_path)


def prepare_epochs(
    storage: StorageManager,
    lock: str,
    comb_ses: tuple[list[str], list[str]]
) -> dict[str, mne.Epochs | mne.EpochsArray]:
    if storage.subject_path is None:
        raise AttributeError('A subject is not selected in the storage')

    return {
        tuple(comb): mne.concatenate_epochs(list(
            map(
                lambda epoch_file: mne.read_epochs(
                    os.path.join(storage.epochs_path, epoch_file)
                ).resample(200),
                filter(
                    lambda epochs_file: any([
                        ses in epochs_file for ses in comb
                    ]) and lock in epochs_file,
                    os.listdir(storage.epochs_path)
                )
            )
        )) for comb in comb_ses
    }

if __name__ == '__main__':
    mpl.use('agg')
    parser = argparse.ArgumentParser(
        description='A script for applying the neural network "LFCNN" to the epoched data from '
        'gradiometers related to events for classification'
    )
    parser.add_argument('-cms', '--combine-sessions', type=str, nargs='+',
                        default=['1 2 3', '10 11 12'], help='Sessions to combine')
    parser.add_argument('-ep', '--exclude-participants', type=str, nargs='+',
                        default=[], help='IDs of subjects to exclude')
    parser.add_argument('-l', '--lock', type=str,
                        default='RespCor', help='Stimulus lock to consider')
    parser.add_argument('-c', '--cases', type=str, nargs='+',
                        default=['B1-3', 'B10-12'], help='Cases to consider (must match '
                        'epochs file names for the respective classes)')
    parser.add_argument('-cmc', '--combine-cases', type=str, nargs='+',
                        default=None, help='Cases to consider (must be the number of strings in '
                        'which classes to combine are written separated by a space, indices '
                        'corresponds to order of "--cases" parameter)')
    parser.add_argument('-sd', '--subjects-dir', type=str,
                        default=os.path.join(os.getcwd(), 'Source', 'Subjects'), help='Path to the '
                        'subjects directory')
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
    parser.add_argument('--use-train', action='store_true', help='Use train set from '
                        'separated dataset to test a model')
    parser.add_argument('--no-params', action='store_true', help='Do not compute parameters')
    parser.add_argument('--tmin', type=float, default=None,
                        help='Time to start (Where 0 is stimulus), defaults to start of epoch')
    parser.add_argument('--tmax', type=float, default=None,
                        help='Time to end (Where 0 is stimulus), defaults to end of epoch')

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
        no_params, \
        tmin, tmax = vars(parser.parse_args()).values()

    ses_info = make_session_info(
        cases,
        cases_to_combine,
        classification_name,
        classification_prefix,
        classification_postfix,
        model_name
    )

    classifier = select_model(model_name)
    storage = StorageManager(
        subjects_dir,
        ses_info
    )

    assert len(combined_sessions) == 2, 'Script is implemented for only two combinations of '\
        f'sessions, {len(combined_sessions)} is given'

    combined_sessions = sorted(tuple(map(lambda data: [f'_{sessions_name}{i}_' for i in data.split(' ')], combined_sessions)))

    for subject_name in storage:
        combiner = EpochsCombiner(*prepare_epochs(
                storage,
                'Resp',
                combined_sessions
            ).values()
        ).combine(0, 1)
        n_classes, classes_samples = np.unique(combiner.Y, return_counts=True)
        n_classes = len(n_classes)
        classes_samples = classes_samples.tolist()
        combiner.shuffle()
        import_opt = dict(
            savepath=storage.classification_path + '/',
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
        model = classifier(dataset, lf_params)
        model.build()
        t1 = perf_counter()
        model.train(n_epochs=25, eval_step=100, early_stopping=5)
        runtime = perf_counter() - t1
        y_true_train, y_pred_train = model.predict(meta['train_paths'])
        y_true_test, y_pred_test = model.predict(meta['test_paths'])
        save_parameters(
            Predictions(
                y_pred_test,
                y_true_test
            ),
            os.path.join(storage.predictions_path, f'{ses_info.classification_name_formatted}_pred.pkl'),
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
            save_parameters(
                WaveForms(time_courses.mean(1), induced, times, time_courses),
                os.path.join(storage.parameters_path, f'{ses_info.classification_name_formatted}_waveforms.pkl'),
                'WaveForms'
            )
            save_parameters(
                SpatialParameters(old_patterns, filters),
                os.path.join(storage.parameters_path, f'{ses_info.classification_name_formatted}_spatial_old.pkl'),
                'spatial'
            )
            save_parameters(
                SpatialParameters(patterns, filters),
                os.path.join(storage.parameters_path, f'{ses_info.classification_name_formatted}_spatial.pkl'),
                'spatial'
            )
            save_parameters(
                TemporalParameters(franges, finputs, foutputs, fresponces),
                os.path.join(storage.parameters_path, f'{ses_info.classification_name_formatted}_temporal.pkl'),
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
                os.path.join(storage.parameters_path, f'{ses_info.classification_name_formatted}_sorting.pkl'),
                'sorting'
            )

        perf_table_path = os.path.join(
            storage.perf_tables_path,
            f'{ses_info.classification_name_formatted}.csv'
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
                *ses_info.class_names,
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