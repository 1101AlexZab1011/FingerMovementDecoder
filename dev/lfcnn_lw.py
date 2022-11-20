import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from dataclasses import dataclass
from utils.storage_management import check_path
from combiners import EpochsCombiner
from typing import *
import mne
import tensorflow as tf
import mneflow as mf
import matplotlib.pyplot as plt
import numpy as np
mne.set_log_level('CRITICAL')

content_root = './'
subjects_folder_path = os.path.join(content_root, 'Source/Subjects')
subject_path = os.path.join(subjects_folder_path, 'Ga_Fed_06')
info_path = os.path.join(subject_path, 'Info',
                        'ML_Subject05_P1_tsss_mc_trans_info.pkl')
resp_lock_lm_B1_epochs_path = os.path.join(
    subject_path, 'Epochs', 'RespCor_LM_B1_epochs.fif')
resp_lock_rm_B1_epochs_path = os.path.join(
    subject_path, 'Epochs', 'RespCor_RM_B1_epochs.fif')
resp_lock_li_B1_epochs_path = os.path.join(
    subject_path, 'Epochs', 'RespCor_LI_B1_epochs.fif')
resp_lock_ri_B1_epochs_path = os.path.join(
    subject_path, 'Epochs', 'RespCor_RI_B1_epochs.fif')
resp_lock_lm_B1_epochs = mne.read_epochs(resp_lock_lm_B1_epochs_path)
resp_lock_li_B1_epochs = mne.read_epochs(resp_lock_li_B1_epochs_path)
resp_lock_rm_B1_epochs = mne.read_epochs(resp_lock_rm_B1_epochs_path)
resp_lock_ri_B1_epochs = mne.read_epochs(resp_lock_ri_B1_epochs_path)

resp_lock_li_B1_epochs.resample(200)
resp_lock_lm_B1_epochs.resample(200)
resp_lock_ri_B1_epochs.resample(200)
resp_lock_rm_B1_epochs.resample(200)

combiner = EpochsCombiner(
    resp_lock_lm_B1_epochs.copy(),
    resp_lock_li_B1_epochs.copy(),
    resp_lock_rm_B1_epochs.copy(),
    resp_lock_ri_B1_epochs.copy()
)
first_class_indices = (0, 1)
second_class_indices = (2, 3)
combiner.combine(first_class_indices, second_class_indices, shuffle=True)
# combiner = EpochsCombiner(resp_lock_li_B1_epochs, resp_lock_lm_B1_epochs)
# combiner.combine(0, 1, shuffle=True)




savepath = './Source/tmp/TFR/'
check_path(savepath)
project_name = 'fake_name'

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
meta = mf.produce_tfrecords((combiner.X, combiner.Y), **import_opt)
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

weights_path = os.path.join(subject_path, 'Weights', 'LM&LI_vs_RM&RI_B1-B8.h5')
model.km.load_weights(weights_path)

X = combiner.X.copy()
X = np.transpose(np.expand_dims(X, axis = 1), (0, 1, 3, 2))

print(model.km(X))