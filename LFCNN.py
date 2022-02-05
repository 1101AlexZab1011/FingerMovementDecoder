import sys
import os
current_dir = os.path.dirname(os.path.abspath('./'))
if not current_dir in sys.path:
    sys.path.append(current_dir)
from combiners import EpochsCombiner
from typing import *
import mne
import pandas as pd
import tensorflow as tf
import mneflow as mf
import matplotlib.pyplot as plt
import numpy as np
from utils.storage_management import check_path


case_indices = dict(
    left_vs_right=[(0, 1), (2, 3)],
    lm_vs_li = [0, 1],
    rm_vs_ri = [2, 3],
    
)

content_root = './'
subjects_folder_path = os.path.join(content_root, 'Source/Subjects')
perf_tables_path = os.path.join('Source', 'perf_tables')
check_path(perf_tables_path)
train_acc, train_loss, val_acc, val_loss, test_acc, test_loss = (list() for _ in range(6))

indices = list()
for session in ['B1', 'B10']:
    for case, class_indices in case_indices.items():
        first_class_indices = class_indices[0]
        second_class_indices = class_indices[1]
        for subject in os.listdir(subjects_folder_path):
            indices.append(subject)
            subject_path = os.path.join(subjects_folder_path, subject)
            resp_lock_lm_B1_epochs_path = os.path.join(
                subject_path, 'Epochs', f'RespCor_LM_{session}_epochs.fif')
            resp_lock_rm_B1_epochs_path = os.path.join(
                subject_path, 'Epochs', f'RespCor_RM_{session}_epochs.fif')
            resp_lock_li_B1_epochs_path = os.path.join(
                subject_path, 'Epochs', f'RespCor_LI_{session}_epochs.fif')
            resp_lock_ri_B1_epochs_path = os.path.join(
                subject_path, 'Epochs', f'RespCor_RI_{session}_epochs.fif')
            resp_lock_lm_B1_epochs = mne.read_epochs(resp_lock_lm_B1_epochs_path)
            resp_lock_rm_B1_epochs = mne.read_epochs(resp_lock_rm_B1_epochs_path)
            resp_lock_li_B1_epochs = mne.read_epochs(resp_lock_li_B1_epochs_path)
            resp_lock_ri_B1_epochs = mne.read_epochs(resp_lock_ri_B1_epochs_path)
            combiner = EpochsCombiner(resp_lock_lm_B1_epochs, resp_lock_li_B1_epochs, resp_lock_rm_B1_epochs, resp_lock_ri_B1_epochs)
            combiner.combine(first_class_indices, second_class_indices, shuffle=True)

            check_path(os.path.join(subject_path, 'TFR'))
            
            # Specify import options
            import_opt = dict(
                # path where TFR files will be saved
                savepath=f'./Source/Subjects/{subject}/TFR/{case}/',
                out_name='mne_sample_epochs',  # name of TFRecords files
                fs=200,
                input_type='trials',
                target_type='int',
                picks={'meg': 'grad'},
                scale=True,  # apply baseline_scaling
                crop_baseline=True,  # remove baseline interval after scaling
                decimate=None,
                # indices in time axis corresponding to baseline interval
                scale_interval=(0, 60),
                n_folds=5,  # validation set size set to 20% of all data
                overwrite=True,
                segment=False,
                test_set='holdout'
            )
            meta = mf.produce_tfrecords((combiner.X, combiner.Y), **import_opt)
            # Ivan had 120 Hz
            dataset = mf.Dataset(meta, train_batch=100)
            lf_params = dict(
                n_latent=32,  # number of latent factors ~ optimal
                # convolutional filter length in time samples ~ increase (its 17 amopng 1000)
                filter_length=50,
                nonlin=tf.keras.activations.elu,
                padding='SAME',
                pooling=10,  # pooling factor (5 - 10)
                stride=10,  # stride parameter for pooling layer
                pool_type='max',
                model_path=import_opt['savepath'],
                dropout=.4,
                l2_scope=["weights"],
                l2=1e-6  # decrease it
            )

            model = mf.models.LFCNN(dataset, lf_params)
            model.build()

            model.train(n_epochs=25, eval_step=100, early_stopping=5)
            train_loss_, train_acc_ = model.evaluate(meta['train_paths'])
            test_loss_, test_acc_ = model.evaluate(meta['test_paths'])
            train_acc.append(train_acc_) 
            train_loss.append(train_loss_)
            val_acc.append(model.v_metric)
            val_loss.append(model.v_loss)
            test_acc.append(test_acc_)
            test_loss.append(test_loss_)

            # model.plot_hist()

        columns = ['train accuracy', 'train loss', 'val accuracy', 'val loss', 'test accuracy', 'test loss']

        df = pd.DataFrame(list(zip(train_acc, train_loss, val_acc, val_loss, test_acc, test_loss)), columns=columns, index=indices)
        df.to_csv(os.path.join(perf_tables_path, f'{case}_{session}_perf.csv'))

