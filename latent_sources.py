
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import mne
from mne.decoding import cross_val_multiscore
from mne import Epochs, create_info, events_from_annotations
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.time_frequency import AverageTFR

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from typing import Union
from utils.beep import Beeper
from sklearn.base import BaseEstimator
from abc import ABC, abstractmethod
from typing import Optional, Iterable, Generator, Callable
from mne import EpochsArray
from collections import UserDict, UserList
from combiners import EpochsCombiner
from mne.decoding import SlidingEstimator
from sklearn.metrics import accuracy_score
from sklearn.base import ClassifierMixin, RegressorMixin
from utils.machine_learning import AbstractTransformer
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from utils.machine_learning.plotting import binary_dicision_boundary
from utils.data_management import dict2str
from sklearn.model_selection import train_test_split
from typing import Any
import pickle


def read_pkl(path: str) -> Any:
    content = pickle.load(
        open(
            path,
            'rb'
        )
    )
    return content

# speakers
# sad_beep = Beeper(duration=[.1, .15, .25], frequency=[280, 240, 190], repeat=3)
# happy_beep = Beeper(duration=[.1, .1, .15, .25], frequency=[400, 370, 470, 500], repeat=4)
sad_beep = lambda: print('sad_beep')
happy_beep = lambda: print('happy_beep')

# paths
content_root = './'
subjects_folder_path = os.path.join(content_root, 'Source/Subjects')
subject_path = os.path.join(subjects_folder_path, 'Az_Mar_05')
info_path = os.path.join(subject_path, 'Info', 'ML_Subject05_P1_tsss_mc_trans_info.pkl')
resp_lock_lm_B1_epochs_path = os.path.join(subject_path, 'Epochs', 'RespCor_LM_B1_epochs.fif')
resp_lock_rm_B1_epochs_path = os.path.join(subject_path, 'Epochs', 'RespCor_RM_B1_epochs.fif')
resp_lock_li_B1_epochs_path = os.path.join(subject_path, 'Epochs', 'RespCor_LI_B1_epochs.fif')
resp_lock_ri_B1_epochs_path = os.path.join(subject_path, 'Epochs', 'RespCor_RI_B1_epochs.fif')

# readers
original_info = read_pkl(info_path)
resp_lock_lm_B1_epochs = mne.read_epochs(resp_lock_lm_B1_epochs_path)
resp_lock_rm_B1_epochs = mne.read_epochs(resp_lock_rm_B1_epochs_path)
resp_lock_li_B1_epochs = mne.read_epochs(resp_lock_li_B1_epochs_path)
resp_lock_ri_B1_epochs = mne.read_epochs(resp_lock_ri_B1_epochs_path)

# combiner

#classes
first_class_indices = (0, 1)
second_class_indices = (2, 3)

# combiner = EpochsCombiner(
#     resp_lock_lm_B1_epochs,
#     resp_lock_li_B1_epochs,
#     resp_lock_rm_B1_epochs,
#     resp_lock_ri_B1_epochs
# ).filter(l_freq=8, h_freq=13).combine((0, 1), (2, 3))

combiner = EpochsCombiner(
    resp_lock_li_B1_epochs,
    resp_lock_ri_B1_epochs
).filter(l_freq=15, h_freq=30).combine(0, 1)

print(combiner.X.shape)

csp = CSP(n_components=4, reg='shrinkage', rank='full', component_order='alternate')
clf = LogisticRegression()

X = combiner.X
Y = combiner.Y
x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle=True)
X_t = csp.fit_transform(X, Y)

print(X.shape)
n_components = 4
filters = csp.filters_[:n_components]
X_t = np.array([filters@epoch for epoch in X])
# X_t = X_t**2
print(X_t.shape)
plt.show()

ch_names = [f'L{i+1}' for i in range(n_components)]
ch_types = ['mag'] * n_components
sampling_freq = 1000
info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
# csp.plot_patterns(resp_lock_li_B1_epochs.info, size=5, res=128)
# plt.show()
latent_epochs = mne.EpochsArray(X_t, info)
# latent_epochs.plot(scalings='auto')
# plt.show()
latent_epochs.plot_image()
plt.show()
# latent_evo = latent_epochs.average()
# latent_evo.plot()
# plt.show()
# latent_epochs = mne.EpochsArray(X, resp_lock_li_B1_epochs.info)
# latent_epochs.plot()