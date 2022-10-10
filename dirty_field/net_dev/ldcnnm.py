import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, os.path.dirname(parentdir))
sys.path.insert(0, parentdir)

from utils.structures import Pipeline, Deploy
from utils.data_management import dict2str
from utils.machine_learning import one_hot_encoder, one_hot_decoder
from typing import *
import tensorflow as tf
from sklearn.datasets import make_classification
import mne
from combiners import EpochsCombiner
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from utils.machine_learning.designer import ModelDesign, ParallelDesign, LayerDesign
from utils.machine_learning.analyzer import ModelAnalyzer, LFCNNAnalyzer
from mne.datasets import multimodal
import sklearn
import mneflow as mf
import tensorflow as tf
from mneflow.layers import DeMixing, LFTConv, TempPooling, Dense
from mneflow.models import BaseModel
import mneflow
import logging
from time import perf_counter
import wandb

class WanbCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        model,
        meta,
        *args, **kwargs):
        self.model = model
        self.meta = meta
        self.start_time = perf_counter()
        wandb.init(*args, **kwargs)
    def on_epoch_end(self, epoch, logs=None):
        wandb.log(logs)
    def on_train_end(self, logs=None):
        train_runtime = perf_counter() - self.start_time
        wandb.log(dict(
            train_runtime=train_runtime
        ))

wlogger= logging.getLogger('wandb')
wlogger.setLevel(logging.CRITICAL)
logger= logging.getLogger(__name__)
logging.root.handlers = []
logger.setLevel(logging.NOTSET)
logging.basicConfig(
    format='%(asctime)s, %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler('./history.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class Deconw(tf.keras.layers.Layer):
    def __init__(
        self,
        units=32,
        kernel_size=(4, 10),
        strides=(1, 1),
        padding='valid',
        output_padding=None,
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__()
        self.units = units
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.output_padding = output_padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.kwargs = kwargs

    def build(self, input_shape):
        self.n_channels = input_shape[-1]
        self.deconws = self.deconv_constructor(self.n_channels)

    def __call__(self, inputs):
        self.build(inputs.shape)
        outputs = []
        for i in range(self.n_channels):
            input_ = tf.expand_dims(inputs[:, :, :, i], axis=3)
            outputs.append(self.deconws[i](input_))
        return tf.transpose(tf.stack(outputs), (1, 0, 2, 3, 4))

    def deconv_constructor(self, n_channels):
        return [
            tf.keras.layers.Conv2DTranspose(
                filters=1,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding=self.padding,
                output_padding=self.output_padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate,
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                activity_regularizer=self.activity_regularizer,
                kernel_constraint=self.kernel_constraint,
                bias_constraint=self.bias_constraint,
                **self.kwargs
            )
        for _ in range(n_channels)
    ]


class ZubarevBaseNet(BaseModel):
    def __init__(self, Dataset, specs=dict()):
        super().__init__(Dataset, specs)

    def train(
        self,
        n_epochs,
        eval_step=None,
        min_delta=1e-6,
        early_stopping=3,
        mode='single_fold',
        *,
        callbacks=None
    ):
        callbacks = [] if callbacks is None else callbacks

        stop_early = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=min_delta,
            patience=early_stopping,
            restore_best_weights=True
        )
        if not eval_step:
            train_size = self.dataset.h_params['train_size']
            eval_step = train_size // self.dataset.h_params['train_batch'] + 1

        self.train_params = [n_epochs, eval_step, early_stopping, mode]

        if mode == 'single_fold':
            self.t_hist = self.km.fit(
                self.dataset.train,
                validation_data=self.dataset.val,
                epochs=n_epochs, steps_per_epoch=eval_step,
                shuffle=True,
                validation_steps=self.dataset.validation_steps,
                callbacks=[stop_early, *callbacks], verbose=2
            )
            self.v_loss, self.v_metric = self.evaluate(self.dataset.val)
            self.v_loss_sd = 0
            self.v_metric_sd = 0
            print("Training complete: loss: {}, Metric: {}".format(self.v_loss, self.v_metric))
            self.update_log()
        elif mode == 'cv':
            n_folds = len(self.dataset.h_params['folds'][0])
            print("Running cross-validation with {} folds".format(n_folds))
            metrics = []
            losses = []
            for jj in range(n_folds):
                print("fold:", jj)
                train, val = self.dataset._build_dataset(
                    self.dataset.h_params['train_paths'],
                    train_batch=self.dataset.training_batch,
                    test_batch=self.dataset.validation_batch,
                    split=True, val_fold_ind=jj
                )
                self.t_hist = self.km.fit(
                    train,
                    validation_data=val,
                    epochs=n_epochs, steps_per_epoch=eval_step,
                    shuffle=True,
                    validation_steps=self.dataset.validation_steps,
                    callbacks=[stop_early, *callbacks], verbose=2
                )
                loss, metric = self.evaluate(val)
                losses.append(loss)
                metrics.append(metric)

                if jj < n_folds -1:
                    self.shuffle_weights()
                else:
                    "Not shuffling the weights for the last fold"


                print("Fold: {} Loss: {:.4f}, Metric: {:.4f}".format(jj, loss, metric))
            self.cv_losses = losses
            self.cv_metrics = metrics
            self.v_loss = np.mean(losses)
            self.v_metric = np.mean(metrics)
            self.v_loss_sd = np.std(losses)
            self.v_metric_sd = np.std(metrics)
            print("{} with {} folds completed. Loss: {:.4f} +/- {:.4f}. Metric: {:.4f} +/- {:.4f}".format(mode, n_folds, np.mean(losses), np.std(losses), np.mean(metrics), np.std(metrics)))
            self.update_log()
            return self.cv_losses, self.cv_metrics

        elif mode == "loso":
            n_folds = len(self.dataset.h_params['test_paths'])
            print("Running leave-one-subject-out CV with {} subject".format(n_folds))
            metrics = []
            losses = []
            for jj in range(n_folds):
                print("fold:", jj)

                test_subj = self.dataset.h_params['test_paths'][jj]
                train_subjs = self.dataset.h_params['train_paths'].copy()
                train_subjs.pop(jj)

                train, val = self.dataset._build_dataset(
                    train_subjs,
                    train_batch=self.dataset.training_batch,
                    test_batch=self.dataset.validation_batch,
                    split=True, val_fold_ind=0
                )
                self.t_hist = self.km.fit(
                    train,
                    validation_data=val,
                    epochs=n_epochs, steps_per_epoch=eval_step,
                    shuffle=True,
                    validation_steps=self.dataset.validation_steps,
                    callbacks=[stop_early, *callbacks], verbose=2
                )
                test = self.dataset._build_dataset(
                    test_subj,
                    test_batch=None,
                    split=False
                )

                loss, metric = self.evaluate(test)
                losses.append(loss)
                metrics.append(metric)

                if jj < n_folds -1:
                    self.shuffle_weights()
                else:
                    "Not shuffling the weights for the last fold"

            self.cv_losses = losses
            self.cv_metrics = metrics
            self.v_loss = np.mean(losses)
            self.v_metric = np.mean(metrics)
            self.v_loss_sd = np.std(losses)
            self.v_metric_sd = np.std(metrics)
            self.update_log()
            print("{} with {} folds completed. Loss: {:.4f} +/- {:.4f}. Metric: {:.4f} +/- {:.4f}".format(mode, n_folds, np.mean(losses), np.std(losses), np.mean(metrics), np.std(metrics)))
            return self.cv_losses, self.cv_metrics


class ZubarevNet(ZubarevBaseNet):
    def __init__(self, Dataset, specs=dict(), design=None, design_name='design'):
        assert design is not None, 'design is not specified'
        self.scope = design_name
        self.design = design
        specs.setdefault('filter_length', 7)
        specs.setdefault('n_latent', 4)
        specs.setdefault('pooling', 4)
        specs.setdefault('stride', 4)
        specs.setdefault('padding', 'SAME')
        specs.setdefault('pool_type', 'max')
        specs.setdefault('nonlin', tf.nn.relu)
        specs.setdefault('l1', 3e-4)
        specs.setdefault('l2', 0)
        specs.setdefault('l1_scope', ['fc', 'demix', 'lf_conv'])
        specs.setdefault('l2_scope', [])
        specs.setdefault('maxnorm_scope', [])

        super().__init__(Dataset, specs)

    def build_graph(self):
        return self.design(self.inputs)

    def set_design(self, design: ModelDesign):
        self.design = design

mne.set_log_level(verbose='CRITICAL')
fname_raw = os.path.join(multimodal.data_path(), 'multimodal_raw.fif')
raw = mne.io.read_raw_fif(fname_raw)
cond = raw.acqparser.get_condition(raw, None)
condition_names = [k for c in cond for k,v in c['event_id'].items()]
epochs_list = [mne.Epochs(raw, **c) for c in cond]
epochs = mne.concatenate_epochs(epochs_list)
epochs = epochs.pick_types(meg='grad')
X = np.array([])
Y = list()
for i, epochs in enumerate(epochs_list):
    data = epochs.get_data()
    if i == 0:
        X = data.copy()
    else:
        X = np.append(X, data, axis=0)
    Y += [i for _ in range(data.shape[0])]

Y = np.array(Y)
X = np.array([X[i, epochs._channel_type_idx['grad'], :] for i, _ in enumerate(X)])
original_X = X.copy()
original_Y = Y.copy()

import_opt = dict(savepath='../tfr/',
    out_name='mne_sample_epochs',
    fs=600,
    input_type='trials',
    target_type='int',
    picks={'meg':'grad'},
    scale=True,  # apply baseline_scaling
    crop_baseline=True,  # remove baseline interval after scaling
    decimate=None,
    scale_interval=(0, 60),  # indices in time axis corresponding to baseline interval
    n_folds=5,
    overwrite=True,
    segment=False,
)

specs = dict()
specs.setdefault('filter_length', 15)
specs.setdefault('n_latent', 4)
specs.setdefault('pooling', 10)
specs.setdefault('stride', 2)
specs.setdefault('padding', 'SAME')
specs.setdefault('pool_type', 'max')
specs.setdefault('nonlin', tf.nn.relu)
specs.setdefault('l1', 3e-4)
specs.setdefault('l2', 0)
specs.setdefault('l1_scope', ['fc', 'dmx', 'tconv', 'fc'])
specs.setdefault('l2_scope', [])
specs.setdefault('maxnorm_scope', [])
specs.setdefault('dropout', .5)

specs['filter_length'] = 17
specs['pooling'] = 5
specs['stride'] = 5
specs['l1'] = 3e-3
out_dim = len(np.unique(original_Y))

lfcnnd = ModelDesign(
    None,
    DeMixing(
        size=specs['n_latent'],
        nonlin=tf.identity,
        axis=3, specs=specs
    ),
    LFTConv(
        size=specs['n_latent'],
        nonlin=specs['nonlin'],
        filter_length=specs['filter_length'],
        padding=specs['padding'],
        specs=specs
    ),
    TempPooling(
        pooling=specs['pooling'],
        pool_type=specs['pool_type'],
        stride=specs['stride'],
        padding=specs['padding'],
    ),
    tf.keras.layers.Dropout(specs['dropout'], noise_shape=None),
    Dense(size=out_dim, nonlin=tf.identity, specs=specs)
)

simplenetd = ModelDesign(
    None,
    DeMixing(
        size=specs['n_latent'],
        nonlin=tf.identity,
        axis=3, specs=specs
    ),
    LFTConv(
        size=specs['n_latent'],
        nonlin=specs['nonlin'],
        filter_length=specs['filter_length'],
        padding=specs['padding'],
        specs=specs
    ),
    LFTConv(
        size=specs['n_latent'],
        nonlin=specs['nonlin'],
        filter_length=specs['filter_length'],
        padding=specs['padding'],
        specs=specs
    ),
    LayerDesign(
        lambda X: X[:, :, ::2, :]
    ),
    tf.keras.layers.Dropout(specs['dropout'], noise_shape=None),
    Dense(size=out_dim, nonlin=tf.identity, specs=specs)
)

lfrnnd = ModelDesign(
    None,
    LayerDesign(tf.squeeze, axis=1),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            specs['n_latent'],
            bias_regularizer='l1',
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.L1(.01),
            recurrent_regularizer=tf.keras.regularizers.L1(.01),
            dropout=0.4,
            recurrent_dropout=0.4,
        ),
        merge_mode='sum'
    ),
    LayerDesign(tf.expand_dims, axis=1),
    LFTConv(
        size=specs['n_latent'],
        nonlin=specs['nonlin'],
        filter_length=specs['filter_length'],
        padding=specs['padding'],
        specs=specs
    ),
    TempPooling(
        pooling=specs['pooling'],
        pool_type=specs['pool_type'],
        stride=specs['stride'],
        padding=specs['padding'],
    ),
    tf.keras.layers.Dropout(specs['dropout'], noise_shape=None),
    Dense(size=out_dim, nonlin=tf.identity, specs=specs)
)

Y = original_Y.copy()
Y = one_hot_encoder(Y)
X = original_X.copy()
X = np.transpose(np.expand_dims(X, axis = 1), (0, 1, 3, 2))
print(X.shape)
n_samples, _, n_times, n_channels = X.shape
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(original_X, original_Y, train_size=.85)

newnetd = ModelDesign(
    tf.keras.Input(shape=(1, n_times, n_channels)),
    Deconw(kernel_size=(specs['n_latent'], specs['filter_length']), activation='relu', kernel_regularizer='l1'),
    LayerDesign(
        lambda X: tf.transpose(tf.squeeze(X, axis=-1), (0, 1, 3, 2))
    ),
    tf.keras.layers.DepthwiseConv2D((1, specs['filter_length']), activation='relu', depthwise_regularizer='l1'),
    tf.keras.layers.DepthwiseConv2D((n_channels, 1), name='demixing'),
    LayerDesign(
        lambda X: X[:, :, ::specs['pooling'], :]
    ),
    tf.keras.layers.Dropout(specs['dropout'], noise_shape=None),
    Dense(size=out_dim, nonlin=tf.identity, specs=specs)
)

newnettd = ModelDesign(
    tf.keras.Input(shape=(1, n_times, n_channels)),
    Deconw(kernel_size=(specs['n_latent'], specs['filter_length']), activation='relu', kernel_regularizer='l1'),
    LayerDesign(
        lambda X: tf.transpose(tf.squeeze(X, axis=-1), (0, 1, 3, 2))
    ),
    tf.keras.layers.DepthwiseConv2D((n_channels, 1), name='demixing'),
    tf.keras.layers.DepthwiseConv2D((1, specs['filter_length']), activation='relu', depthwise_regularizer='l1'),
    LayerDesign(
        lambda X: X[:, :, ::specs['pooling'], :]
    ),
    tf.keras.layers.Dropout(specs['dropout'], noise_shape=None),
    Dense(size=out_dim, nonlin=tf.identity, specs=specs)
)

newnetfd = ModelDesign(
    tf.keras.Input(shape=(1, n_times, n_channels)),
    Deconw(kernel_size=(specs['n_latent'], specs['filter_length']), activation='relu', kernel_regularizer='l1'),
    LayerDesign(
        lambda X: tf.transpose(tf.squeeze(X, axis=-1), (0, 1, 3, 2))
    ),
    tf.keras.layers.DepthwiseConv2D((1, specs['filter_length']), activation='relu'),
    tf.keras.layers.DepthwiseConv2D((n_channels, 1), name='demixing'),
    LFTConv(
        size=specs['n_latent'],
        nonlin=specs['nonlin'],
        filter_length=specs['filter_length'],
        padding=specs['padding'],
        specs=specs
    ),
    LayerDesign(
        lambda X: X[:, :, ::specs['pooping'], :]
    ),
    tf.keras.layers.Dropout(specs['dropout'], noise_shape=None),
    Dense(size=out_dim, nonlin=tf.identity, specs=specs)
)

newnetfpd = ModelDesign(
    tf.keras.Input(shape=(1, n_times, n_channels)),
    Deconw(kernel_size=(specs['n_latent'], specs['filter_length']), activation='relu', kernel_regularizer='l1'),
    LayerDesign(
        lambda X: tf.transpose(tf.squeeze(X, axis=-1), (0, 1, 3, 2))
    ),
    tf.keras.layers.DepthwiseConv2D((1, specs['filter_length']), activation='relu'),
    tf.keras.layers.DepthwiseConv2D((n_channels, 1), name='demixing'),
    LFTConv(
        size=specs['n_latent'],
        nonlin=specs['nonlin'],
        filter_length=specs['filter_length'],
        padding=specs['padding'],
        specs=specs
    ),
    TempPooling(
        pooling=specs['pooling'],
        pool_type=specs['pool_type'],
        stride=specs['stride'],
        padding=specs['padding'],
    ),
    tf.keras.layers.Dropout(specs['dropout'], noise_shape=None),
    Dense(size=out_dim, nonlin=tf.identity, specs=specs)
)


newnetad = ModelDesign(
    tf.keras.Input(shape=(1, n_times, n_channels)),
    Deconw(kernel_size=(specs['n_latent'], specs['filter_length']), activation='relu', kernel_regularizer='l1'),
    tf.keras.layers.Conv2D(1, (1, specs['filter_length']), activation='relu'),
    LayerDesign(
        lambda X: tf.transpose(tf.squeeze(X, axis=-1), (0, 1, 3, 2))
    ),
    tf.keras.layers.DepthwiseConv2D((204, 1), kernel_regularizer='l1'),
    LayerDesign(
        lambda X: X[:, :, ::specs['pooling'], :]
    ),
    tf.keras.layers.Dropout(specs['dropout'], noise_shape=None),
    Dense(size=out_dim, nonlin=tf.identity, specs=specs)
)

newnetsd = ModelDesign(
    tf.keras.Input(shape=(1, n_times, n_channels)),
    Deconw(kernel_size=(specs['n_latent'], 1), activation='relu', kernel_regularizer='l1'),
    LayerDesign(
        lambda X: tf.transpose(tf.squeeze(X, axis=-1), (0, 1, 3, 2))
    ),
    tf.keras.layers.DepthwiseConv2D((204, 1)),
    LayerDesign(
        lambda X: X[:, :, ::specs['pooling'], :]
    ),
    tf.keras.layers.Dropout(specs['dropout'], noise_shape=None),
    Dense(size=out_dim, nonlin=tf.identity, specs=specs)
)

newnetsfd = ModelDesign(
    tf.keras.Input(shape=(1, n_times, n_channels)),
    Deconw(kernel_size=(specs['n_latent'], 1), activation='relu', kernel_regularizer='l1'),
    LayerDesign(
        lambda X: tf.transpose(tf.squeeze(X, axis=-1), (0, 1, 3, 2))
    ),
    tf.keras.layers.DepthwiseConv2D((204, 1)),
    LFTConv(
        size=specs['n_latent'],
        nonlin=specs['nonlin'],
        filter_length=specs['filter_length'],
        padding=specs['padding'],
        specs=specs
    ),
    LayerDesign(
        lambda X: X[:, :, ::specs['pooling'], :]
    ),
    tf.keras.layers.Dropout(specs['dropout'], noise_shape=None),
    Dense(size=out_dim, nonlin=tf.identity, specs=specs)
)


newnetsfpd = ModelDesign(
    tf.keras.Input(shape=(1, n_times, n_channels)),
    Deconw(kernel_size=(specs['n_latent'], 1), activation='relu', kernel_regularizer='l1'),
    LayerDesign(
        lambda X: tf.transpose(tf.squeeze(X, axis=-1), (0, 1, 3, 2))
    ),
    tf.keras.layers.DepthwiseConv2D((204, 1), kernel_regularizer='l1'),
    LFTConv(
        size=specs['n_latent'],
        nonlin=specs['nonlin'],
        filter_length=specs['filter_length'],
        padding=specs['padding'],
        specs=specs
    ),
    TempPooling(
        pooling=specs['pooling'],
        pool_type=specs['pool_type'],
        stride=specs['stride'],
        padding=specs['padding'],
    ),
    tf.keras.layers.Dropout(specs['dropout'], noise_shape=None),
    Dense(size=out_dim, nonlin=tf.identity, specs=specs)
)

newnetsimpled = ModelDesign(
    tf.keras.Input(shape=(1, n_times, n_channels)),
    Deconw(kernel_size=(specs['n_latent'], 1), activation='relu', kernel_regularizer='l1'),
    LayerDesign(
        lambda X: tf.transpose(tf.squeeze(X, axis=-1), (0, 1, 3, 2))
    ),
    tf.keras.layers.DepthwiseConv2D((204, 1), kernel_regularizer='l1'),
    LFTConv(
        size=specs['n_latent'],
        nonlin=specs['nonlin'],
        filter_length=specs['filter_length'],
        padding=specs['padding'],
        specs=specs
    ),
    LFTConv(
        size=specs['n_latent'],
        nonlin=specs['nonlin'],
        filter_length=specs['filter_length'],
        padding=specs['padding'],
        specs=specs
    ),
    LayerDesign(
        lambda X: X[:, :, ::specs['pooling'], :]
    ),
    tf.keras.layers.Dropout(specs['dropout'], noise_shape=None),
    Dense(size=out_dim, nonlin=tf.identity, specs=specs)
)

designs = {
    'lfcnn': lfcnnd,
    'lfrnn': lfrnnd,
    'simplenet': simplenetd,
    'newnet': newnetd,
    'newnet_t': newnettd,
    'newnet_alt': newnetad,
    'newnet_f': newnetfd,
    'newnet_fp': newnetfpd,
    'newnet_s': newnetsd,
    'newnet_sf': newnetsfd,
    'newnet_sfp': newnetsfpd,
    'newnet_simple': newnetsimpled
}

# write TFRecord files and metadata file to disk
meta = mneflow.produce_tfrecords((original_X, original_Y), **import_opt)
dataset = mneflow.Dataset(meta, train_batch=100)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='lfcnn')
parser.add_argument('--postfix', type=str, default='')
model_name,\
    postfix = vars(parser.parse_args()).values()
model = ZubarevNet(dataset, specs, designs[model_name], model_name)
model.build()
t1 = perf_counter()
model.train(n_epochs=25, eval_step=100, early_stopping=5,
            callbacks=[
                WanbCallback(
                    model, meta,
                    project='comp_decs',
                    config=specs,
                    name=model_name + postfix
                )
            ]
        )
y_true_train, y_pred_train = model.predict(meta['train_paths'])
t1 = perf_counter()
y_true_test, y_pred_test = model.predict(meta['test_paths'])
runtime=perf_counter()-t1
logging.info(
    f'{model.scope} performance:\n'
    f'\truntime: {runtime : .4f}\n'
    f'\ttrain-set: {sklearn.metrics.accuracy_score(one_hot_decoder(y_true_train), one_hot_decoder(y_pred_train))}\n'
    f'\ttest-set: {sklearn.metrics.accuracy_score(one_hot_decoder(y_true_test), one_hot_decoder(y_pred_test))}'
)

wandb.log(dict(
    test_runtime=runtime,
    train_acc=sklearn.metrics.accuracy_score(one_hot_decoder(y_true_train), one_hot_decoder(y_pred_train)),
    test_acc=sklearn.metrics.accuracy_score(one_hot_decoder(y_true_test), one_hot_decoder(y_pred_test))
))
