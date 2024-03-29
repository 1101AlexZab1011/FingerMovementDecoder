import argparse
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
from utils.storage_management import check_path
from utils.machine_learning import one_hot_decoder
import matplotlib as mpl
import sklearn
from LFCNN_decoder import Predictions, save_parameters
# compute_temporal_parameters,
# , SpatialParameters, TemporalParameters,\
# ComponentsOrder, , WaveForms,\
from mneflow.models import BaseModel
from utils.machine_learning.designer import ModelDesign, LayerDesign
from mneflow.layers import Dense, LFTConv, TempPooling, DeMixing
from tensorflow.keras.initializers import Constant
from tensorflow.keras import regularizers as k_reg
from time import perf_counter


class MneflowNet(BaseModel):
    def __init__(self, Dataset, specs=dict(), design='eegnet'):
        self.scope = design
        specs.setdefault('filter_length', 7)
        specs.setdefault('n_latent', 32)
        specs.setdefault('pooling', 4)
        specs.setdefault('stride', 4)
        specs.setdefault('padding', 'SAME')
        specs.setdefault('pool_type', 'max')
        specs.setdefault('nonlin', tf.nn.relu)
        specs.setdefault('l1_lambda', 3e-4)
        specs.setdefault('l2_lambda', 0)
        specs.setdefault('l1_scope', ['fc', 'demix', 'lf_conv'])
        specs.setdefault('l2_scope', [])
        specs.setdefault('maxnorm_scope', [])

        super(MneflowNet, self).__init__(Dataset, specs)

    def build_graph(self):
        if self.scope == 'lfcnn':
            self.design = ModelDesign(
                self.inputs,
                DeMixing(
                    size=self.specs['n_latent'],
                    nonlin=tf.identity,
                    axis=3, specs=self.specs
                ),
                LFTConv(
                    size=self.specs['n_latent'],
                    nonlin=self.specs['nonlin'],
                    filter_length=self.specs['filter_length'],
                    padding=self.specs['padding'],
                    specs=self.specs
                ),
                TempPooling(
                    pooling=self.specs['pooling'],
                    pool_type=self.specs['pool_type'],
                    stride=self.specs['stride'],
                    padding=self.specs['padding'],
                ),
                tf.keras.layers.Dropout(self.specs['dropout'], noise_shape=None),
                Dense(size=self.out_dim, nonlin=tf.identity, specs=self.specs)
            )
        elif self.scope == 'deep4' or self.scope == 'vgg19':
            # deep4
            self.design = ModelDesign(
                self.inputs,
                LayerDesign(tf.transpose, [0, 3, 2, 1]),
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=(1, self.specs['filter_length']),
                    depth_multiplier=self.specs['n_latent'],
                    strides=1,
                    padding=self.specs['padding'],
                    activation=tf.identity,
                    kernel_initializer="he_uniform",
                    bias_initializer=Constant(0.1),
                    data_format="channels_last",
                    kernel_regularizer=k_reg.l2(self.specs['l2'])
                    # kernel_constraint="maxnorm"
                ),
                tf.keras.layers.Conv2D(
		        filters=self.specs['n_latent'],
		        kernel_size=(self.dataset.h_params['n_ch'], 1),
		        strides=1,
		        padding=self.specs['padding'],
		        activation=self.specs['nonlin'],
		        kernel_initializer="he_uniform",
		        bias_initializer=Constant(0.1),
		        data_format="channels_last",
		        # data_format="channels_first",
		        kernel_regularizer=k_reg.l2(self.specs['l2'])
		    ),
		TempPooling(
		        pooling=self.specs['pooling'],
		        pool_type="avg",
		        stride=self.specs['stride'],
		        padding='SAME',
		    ),
		tf.keras.layers.Conv2D(
		        filters=self.specs['n_latent'],
		        kernel_size=(self.dataset.h_params['n_ch'], 1),
		        strides=1,
		        padding=self.specs['padding'],
		        activation=self.specs['nonlin'],
		        kernel_initializer="he_uniform",
		        bias_initializer=Constant(0.1),
		        data_format="channels_last",
		        # data_format="channels_first",
		        kernel_regularizer=k_reg.l2(self.specs['l2'])
		    ),
		TempPooling(
		        pooling=self.specs['pooling'],
		        pool_type="avg",
		        stride=self.specs['stride'],
		        padding='SAME',
		    ),
		tf.keras.layers.Conv2D(
		        filters=self.specs['n_latent'],
		        kernel_size=(self.dataset.h_params['n_ch'], 1),
		        strides=1,
		        padding=self.specs['padding'],
		        activation=self.specs['nonlin'],
		        kernel_initializer="he_uniform",
		        bias_initializer=Constant(0.1),
		        data_format="channels_last",
		        # data_format="channels_first",
		        kernel_regularizer=k_reg.l2(self.specs['l2'])
		    ),
		TempPooling(
		        pooling=self.specs['pooling'],
		        pool_type="avg",
		        stride=self.specs['stride'],
		        padding='SAME',
		    ),
		 tf.keras.layers.Conv2D(
		        filters=self.specs['n_latent'],
		        kernel_size=(self.dataset.h_params['n_ch'], 1),
		        strides=1,
		        padding=self.specs['padding'],
		        activation=self.specs['nonlin'],
		        kernel_initializer="he_uniform",
		        bias_initializer=Constant(0.1),
		        data_format="channels_last",
		        # data_format="channels_first",
		        kernel_regularizer=k_reg.l2(self.specs['l2'])
		    ),
		TempPooling(
		        pooling=self.specs['pooling'],
		        pool_type="avg",
		        stride=self.specs['stride'],
		        padding='SAME',
		    ),
                Dense(size=self.out_dim, nonlin=tf.identity, specs=self.specs)
            )
        elif self.scope == 'fbcsp':
            # FBCSP_ShallowNet
            self.design = ModelDesign(
                self.inputs,
                LayerDesign(tf.transpose, [0, 3, 2, 1]),
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=(1, self.specs['filter_length']),
                    depth_multiplier=self.specs['n_latent'],
                    strides=1,
                    padding="VALID",
                    activation=tf.identity,
                    kernel_initializer="he_uniform",
                    bias_initializer=Constant(0.1),
                    data_format="channels_last",
                    kernel_regularizer=k_reg.l2(self.specs['l2'])
                    # kernel_constraint="maxnorm"
                ),
                tf.keras.layers.Conv2D(
                    filters=self.specs['n_latent'],
                    kernel_size=(self.dataset.h_params['n_ch'], 1),
                    strides=1,
                    padding="VALID",
                    activation=tf.square,
                    kernel_initializer="he_uniform",
                    bias_initializer=Constant(0.1),
                    data_format="channels_last",
                    # data_format="channels_first",
                    kernel_regularizer=k_reg.l2(self.specs['l2'])
                ),
                TempPooling(
                    pooling=self.specs['pooling'],
                    pool_type="avg",
                    stride=self.specs['stride'],
                    padding='SAME',
                ),
                Dense(size=self.out_dim, nonlin=tf.identity, specs=self.specs)
            )
        elif self.scope == 'eegnet':
            # EEGNet
            self.design = ModelDesign(
                self.inputs,
                LayerDesign(tf.transpose, [0, 3, 2, 1]),
                tf.keras.layers.Conv2D(
                    self.specs['n_latent'],
                    (2, self.specs['filter_length']),
                    padding=self.specs['padding'],
                    use_bias=False
                ),
                tf.keras.layers.BatchNormalization(axis=1),
                tf.keras.layers.DepthwiseConv2D(
                    (self.dataset.h_params['n_ch'], 1),
                    use_bias=False,
                    depth_multiplier=1,
                    depthwise_constraint=tf.keras.constraints.MaxNorm(1.)
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('elu'),
                tf.keras.layers.AveragePooling2D((1, self.specs['pooling'])),
                tf.keras.layers.Dropout(self.specs['dropout']),
                tf.keras.layers.SeparableConv2D(
                    self.specs['n_latent'],
                    (1, self.specs['filter_length'] // self.specs["pooling"]),
                    use_bias=False,
                    padding='same'
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('elu'),
                tf.keras.layers.AveragePooling2D((1, self.specs['pooling'] * 2)),
                tf.keras.layers.Dropout(self.specs['dropout']),
                Dense(size=self.out_dim, nonlin=tf.identity, specs=self.specs)
            )
        elif self.scope == 'simplenet':
            self.design = ModelDesign(
                self.inputs,
                DeMixing(
                    size=self.specs['n_latent'],
                    nonlin=tf.identity,
                    axis=3, specs=self.specs
                ),
                LFTConv(
                    size=self.specs['n_latent'],
                    nonlin=self.specs['nonlin'],
                    filter_length=self.specs['filter_length'],
                    padding=self.specs['padding'],
                    specs=self.specs
                ),
                LFTConv(
                    size=self.specs['n_latent'],
                    nonlin=self.specs['nonlin'],
                    filter_length=self.specs['filter_length'],
                    padding=self.specs['padding'],
                    specs=self.specs
                ),
                LayerDesign(
                    lambda X: X[:, :, ::self.specs['pooling']]
                ),
                tf.keras.layers.Dropout(self.specs['dropout'], noise_shape=None),
                Dense(size=self.out_dim, nonlin=tf.identity, specs=self.specs)
            )
        else:
            raise NotImplementedError(f'Model design {self.scope} is not implemented')

        return self.design()


if __name__ == '__main__':
    mpl.use('agg')
    parser = argparse.ArgumentParser(
        description='A script for applying neural models (LFCNN, VGG19, EEGNet, FBCSP ShallowNet)'
        ' to the epoched data from gradiometers related to events for classification'
    )
    parser.add_argument('-es', '--exclude-sessions', type=str, nargs='+',
                        default=[], help='Sessions to exclude')
    parser.add_argument('-ep', '--exclude-participants', type=str, nargs='+',
                        default=[], help='IDs of subjects to exclude')
    parser.add_argument('-l', '--lock', type=str,
                        default='RespCor', help='Stimulus lock to consider')
    parser.add_argument('-c', '--cases', type=str, nargs='+',
                        default=['LI', 'LM', 'RI', 'RM'], help='Cases to consider '
                        '(must match epochs file names for the respective classes)')
    parser.add_argument('-cmb', '--combine-cases', type=str, nargs='+',
                        default=None, help='Cases to consider (must be the number of '
                        'strings in which classes to combine are written separated by a '
                        'space, indices corresponds to order of "--cases" parameter)')
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
        model_name = vars(parser.parse_args()).values()

    if excluded_sessions:
        excluded_sessions = [
            sessions_name + session
            if sessions_name not in session
            else session
            for session in excluded_sessions
        ]

    cases_to_combine = [case.split(' ') for case in cases] \
        if cases_to_combine is None \
        else [case.split(' ') for case in cases_to_combine]
    cases = list(filter(lambda case: any([case in cmb for cmb in cases_to_combine]), cases))
    cases_to_combine = sorted(cases_to_combine, reverse=True)
    class_names = [
        '&'.join(sorted(cases_combination, reverse=True))
        for cases_combination in cases_to_combine
    ]

    if classification_name is None:
        classification_name = '_vs_'.join(class_names)

    classification_name_formatted = "_".join(
        list(filter(
            lambda s: s not in (None, ""),
            [classification_prefix, classification_name, classification_postfix]
        ))
    )

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
        print('#' * 100)
        print(f'{meta["test_size"] = }, {meta["val_size"] = }, {meta["class_ratio"] = }')
        print('#' * 100)
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

        model = MneflowNet(dataset, lf_params, model_name)
        print(tf.keras.Model(inputs=model.inputs, outputs=model.y_pred).summary())
        model.build()
        t1 = perf_counter()
        model.train(n_epochs=25, eval_step=100, early_stopping=5)
        print('#' * 100)
        runtime = perf_counter() - t1
        print(f'{classification_name_formatted}\n{model_name}\nruntime: {runtime}')
        print('#' * 100)
        network_out_path = os.path.join(subject_path, 'LFCNNm')
        check_path(network_out_path)
        yp_path = os.path.join(network_out_path, 'Predictions')
        y_true_train, y_pred_train = model.predict(meta['train_paths'])
        y_true_test, y_pred_test = model.predict(meta['test_paths'])

        print(
            'train-set: ',
            subject_name,
            sklearn.metrics.accuracy_score(
                one_hot_decoder(y_true_train),
                one_hot_decoder(y_pred_train)
            )
        )
        print(
            'test-set: ',
            subject_name,
            sklearn.metrics.accuracy_score(
                one_hot_decoder(y_true_test),
                one_hot_decoder(y_pred_test)
            )
        )

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
            f'{model_name}_{classification_name_formatted}.csv'
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
