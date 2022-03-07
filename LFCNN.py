import matplotlib
import sys
import os
current_dir = os.path.dirname(os.path.abspath('./'))
if not current_dir in sys.path:
    sys.path.append(current_dir)
    
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
import matplotlib
matplotlib.use('agg')


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

Y = original_Y.copy()
Y = one_hot_encoder(Y)

X = original_X.copy()
X = np.transpose(np.expand_dims(X, axis = 1), (0, 1, 3, 2))

specs = dict()
specs.setdefault('filter_length', 7)
specs.setdefault('n_latent', 32)
specs.setdefault('pooling', 2)
specs.setdefault('stride', 2)
specs.setdefault('padding', 'SAME')
specs.setdefault('pool_type', 'max')
specs.setdefault('nonlin', tf.nn.relu)
specs.setdefault('l1', 3e-4)
specs.setdefault('l2', 0)
# specs.setdefault('l1_scope', ['fc', 'demix', 'lf_conv'])
specs.setdefault('l1_scope', ['fc', 'dmx', 'tconv', 'fc'])
specs.setdefault('l2_scope', [])
specs.setdefault('maxnorm_scope', [])
specs.setdefault('dropout', .5)

# n_latent=32,
# filter_length=17,
# nonlin = tf.nn.relu,
# padding = 'SAME',
# pooling = 5,
# stride = 5,
# pool_type='max',
# dropout = .5,
# l1_scope = ["weights"],
# l1=3e-3

specs['filter_length'] = 17
specs['pooling'] = 5
specs['stride'] = 5
specs['l1'] = 3e-3


out_dim = len(np.unique(original_Y))

n_samples, _, n_times, n_channels = X.shape

inputs = tf.keras.Input(shape=(1, n_times, n_channels))

kmd = ModelDesign(
    tf.keras.Input(shape=(1, n_times, n_channels)),
    DeMixing(size=specs['n_latent'], nonlin=tf.identity, axis=3, specs=specs),
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

print(kmd().shape)


km = kmd.build()

optimizer="adam"
learn_rate=3e-4
params = {"optimizer": tf.optimizers.get(optimizer).from_config({"learning_rate":learn_rate})}
params.setdefault("loss", tf.nn.softmax_cross_entropy_with_logits)
params.setdefault("metrics", tf.keras.metrics.CategoricalAccuracy(name="cat_ACC"))

km.compile(optimizer=params["optimizer"],
                loss=params["loss"],
                metrics=params["metrics"])

import mneflow
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
                #   n_folds=5,  # validation set size set to 20% of all data
                  n_folds=5,
                  overwrite=True,
                  segment=False,
                #   test_set='holdout'
)


#write TFRecord files and metadata file to disk
meta = mneflow.produce_tfrecords((original_X, original_Y), **import_opt)  
dataset = mneflow.Dataset(meta, train_batch=100)

train_size = dataset.h_params['train_size']
eval_step = train_size // dataset.h_params['train_batch'] + 1
min_delta=1e-6
early_stopping=3
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      min_delta=min_delta,
                                                      patience=early_stopping,
                                                      restore_best_weights=True)

km.fit(dataset.train,
      validation_data=dataset.val,
      epochs=300, steps_per_epoch=eval_step,
      shuffle=True, 
      validation_steps=dataset.validation_steps,
      callbacks=[stop_early], verbose=2)

ma = ModelAnalyzer(km)
fig = ma.plot_metrics(['cat_ACC', 'val_cat_ACC'], show=False)
fig.savefig('/mnt/Local_data/Alexey_Zabolotniy/ACC.png')
fig.close()
ma.plot_metrics(['loss', 'val_loss'], show=False)
fig.savefig('/mnt/Local_data/Alexey_Zabolotniy/LOSS.png')
fig.close()