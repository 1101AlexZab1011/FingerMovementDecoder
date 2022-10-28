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
from utils.machine_learning import one_hot_encoder


SessionInfo = namedtuple(
    'SessionInfo',
    'cases_cmb class_names classification_name classification_name_formatted model_name excluded_sessions excluded_subjects'
)

def make_session_info(
    cases: list[str],
    cases_to_combine: Optional[list[str]] = None,
    classification_name: Optional[str] = None,
    classification_prefix: Optional[str] = None,
    classification_postfix: Optional[str] = None,
    model_name: Optional[str] = 'unknown',
    excluded_sessions: Optional[list[str]] = None,
    excluded_subjects: Optional[list[str]] = None
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

    excluded_sessions = list() if excluded_sessions is None else excluded_sessions
    excluded_subjects = list() if excluded_subjects is None else excluded_subjects

    return SessionInfo(
        cases_to_combine,
        class_names,
        classification_name,
        classification_name_formatted,
        model_name,
        excluded_sessions,
        excluded_subjects
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
            if self.subject_dirs[self.__current_subject_index] in self.session_info.excluded_subjects:
                self.__current_subject_index += 1
                return next(self)
            else:
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
    cases: list[str],
    sessions_name: str,
    excluded_sessions: list[str]
) -> dict[str, mne.Epochs | mne.EpochsArray]:
    if storage.subject_path is None:
        raise AttributeError('A subject is not selected in the storage')

    epochs = {case: list() for case in cases}

    for epochs_file in os.listdir(storage.epochs_path):
        if lock not in epochs_file:
            continue

        session = re.findall(r'_{0}\d\d?'.format(sessions_name), epochs_file)[0][1:]

        if session in excluded_sessions:
            continue

        for case in cases:
            if case in epochs_file:
                with Silence(), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    epochs_ = mne.read_epochs(os.path.join(storage.epochs_path, epochs_file))
                    epochs_.resample(200)

                    epochs[case].append(epochs_)

    return dict(
        zip(
            epochs.keys(),
            map(
                mne.concatenate_epochs,
                list(epochs.values())
            )
        )
    )


class MaximizingLabelLoss(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true_inv = tf.cast(tf.math.logical_not(tf.cast(y_true, bool)), tf.dtypes.float32)
        y_true_val = tf.math.reduce_sum(y_true*y_pred)
        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)
        useful_diff = y_true_inv*y_pred - y_true_inv*y_true_val

        return tf.math.reduce_mean(useful_diff)/(1+ tf.math.reduce_std(useful_diff[:, :, 1:]))


class Encoder(tf.keras.Model):

    def __init__(
        self,
        decoder: mneflow.models.BaseModel,
        encoder_design: ModelDesign
    ):
        super().__init__()
        self.decoder = decoder
        self.encoder_design = encoder_design

    def call(self, inputs):
        return self.encoder_design(inputs)

    def train_step(self, data):
        y = data

        with tf.GradientTape() as tape:
            encoder_pred = self(y, training=True)
            decoder_pred = self.decoder.km(encoder_pred)
            decoder_pred = tf.expand_dims(tf.expand_dims(decoder_pred, 1), 1)
            loss = self.compiled_loss(y, decoder_pred, regularization_losses=self.losses)

        trainable_vars = [var for elem in self.encoder_design if hasattr(elem, 'trainable_variables') for var in elem.trainable_variables]
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, decoder_pred)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        y = data
        encoder_pred = self(y, training=False)
        decoder_pred = self.decoder.km(encoder_pred)
        decoder_pred = tf.expand_dims(tf.expand_dims(decoder_pred, 1), 1)
        self.compiled_loss(y, decoder_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, decoder_pred)

        return {m.name: m.result() for m in self.metrics}


if __name__ == '__main__':
    mpl.use('agg')
    parser = argparse.ArgumentParser(
        description='A script for applying the neural network "LFCNN" to the epoched data from '
        'gradiometers related to events for classification'
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
    parser.add_argument('-m', '--model', type=str,
                        default='LFCNN', help='Model to use')

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
        no_params,\
        model_name = vars(parser.parse_args()).values()

    if excluded_sessions:
        excluded_sessions = [
            sessions_name + session
            if sessions_name not in session
            else session
            for session in excluded_sessions
        ]


    ses_info = make_session_info(
        cases,
        cases_to_combine,
        classification_name,
        classification_prefix,
        classification_postfix,
        model_name,
        excluded_sessions,
        excluded_subjects
    )

    classifier = select_model(model_name)
    storage = StorageManager(
        subjects_dir,
        ses_info
    )


    for subject_name in storage:

        epochs = prepare_epochs(
            storage, lock, cases, sessions_name, excluded_sessions
        )

        i = 0
        cases_indices_to_combine = list()
        cases_to_combine_list = list()

        for combination in ses_info.cases_cmb:
            cases_indices_to_combine.append(list())

            for j, case in enumerate(combination):

                i += j
                cases_indices_to_combine[-1].append(i)
                if lfreq is None:
                    cases_to_combine_list.append(epochs[case])
                else:
                    cases_to_combine_list.append(epochs[case].filter(lfreq, None))

            i += 1

        combiner = EpochsCombiner(*cases_to_combine_list).combine(*cases_indices_to_combine)

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
        print(X.shape)
        raise OSError
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

        _, n_times, n_channels = model.input_shape
        n_classes = model.out_dim

        last_dim_x = n_times//4 + 3 if n_times%2 else n_times//4 + 2
        last_dim_y = n_channels//4 + 3 if n_channels%2 else n_channels//4 + 2

        encoder_design = ModelDesign(
            tf.keras.Input(shape=(1, 1, n_classes,), name='input_layer'),
            tf.keras.layers.Conv2DTranspose(10, (n_times//4, n_channels//4)),
            tf.keras.layers.Conv2DTranspose(10, (n_times//2, n_channels//2)),
            tf.keras.layers.Conv2DTranspose(1, (last_dim, last_dim_y)),
            LayerDesign(
                lambda X: tf.transpose(X, (0, 3, 1, 2))
            )
        )
        encoder = Encoder(model, encoder_design)
        encoder.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=MaximizingLabelLoss(),
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)],
            metrics=['acc']
        )
        Yoh = tf.expand_dims(tf.expand_dims(one_hot_encoder(Y), 1), 1)
        encoder.fit(
            X,
            Yoh,
            epochs=1,
            validation_split=0.2,
            shuffle=True,
        )
        generated_data = encoder(Yoh)
        save_parameters(
            Predictions(
                model.km(generated_data).numpy(),
                y_true_test
            ),
            os.path.join(storage.predictions_path, f'{ses_info.classification_name_formatted}_generated_pred.pkl'),
            'predictions'
        )
        import_opt = dict(
            savepath=storage.classification_path + '/',
            out_name=project_name + '_generated',
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
        meta = mf.produce_tfrecords(
            (
                tf.transpose(tf.squeeze(generated_data), (0, 2, 1)).numpy(),
                Y
            ),
                **import_opt
        )
        dataset = mf.Dataset(meta, train_batch=100)
        compute_patterns(model, dataset, output='patterns')
        patterns = model.patterns.copy()
        nt = model.dataset.h_params['n_t']
        time_courses = np.squeeze(model.lat_tcs.reshape([model.specs['n_latent'], -1, nt]))
        times = (1 / float(model.dataset.h_params['fs'])) *\
            np.arange(model.dataset.h_params['n_t'])
        compute_patterns(model, dataset, output='filters')
        filters = model.patterns.copy()
        franges, finputs, foutputs, fresponces = compute_temporal_parameters(model)
        induced, times, time_courses = compute_waveforms(model)
        save_parameters(
            WaveForms(time_courses.mean(1), induced, times, time_courses),
            os.path.join(storage.parameters_path, f'{ses_info.classification_name_formatted}_generated_waveforms.pkl'),
            'WaveForms'
        )
        save_parameters(
            SpatialParameters(patterns, filters),
            os.path.join(storage.parameters_path, f'{ses_info.classification_name_formatted}_generated_spatial.pkl'),
            'spatial'
        )
        save_parameters(
            TemporalParameters(franges, finputs, foutputs, fresponces),
            os.path.join(storage.parameters_path, f'{ses_info.classification_name_formatted}_generated_temporal.pkl'),
            'temporal'
        )
