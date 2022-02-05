import os
import pickle

import mne
from matplotlib import pyplot as plt

from cross_runs_TF_planes import CrossRunsTFScorer

#%%

from utils.console.colored import warn
from cross_runs_TF_planes import CrossRunsTFScorer
import pickle
from typing import Any
import re
import matplotlib
from mne.time_frequency import tfr_morlet
import matplotlib.pyplot as plt
import os
import mne
import numpy as np

INCLUDED_SESSIONS = ['B1_', 'B10']
INCLUDED_TRIALS = ['RespCor']
INCLUDED_CASES = ['LI', 'LM', 'RI', 'RM']
epochs = dict()

root = './Source/Subjects'

subjects_dir = './Source/Subjects'

for subject in os.listdir(root):
    for epoch_file in os.listdir(os.path.join(root, subject, 'Epochs')):
        for session in INCLUDED_SESSIONS:
            if session in epoch_file:
                for trial in INCLUDED_TRIALS:
                    if trial in epoch_file:
                        for case in INCLUDED_CASES:
                            if case in epoch_file:
                                if subject not in epochs:
                                    epochs.update({subject: dict()})
                                if session not in epochs[subject]:
                                    epochs[subject].update({session: dict()})
                                if trial not in epochs[subject][session]:
                                    epochs[subject][session].update({trial: dict()})
                                if case not in epochs[subject][session][trial]:
                                    warn(os.path.join(root, subject, 'Epochs', epoch_file))
                                    epochs[subject][session][trial].update({
                                        case: mne.read_epochs(
                                            os.path.join(root, subject, 'Epochs', epoch_file)
                                        )
                                    })

                            else:
                                continue
                    else:
                        continue
            else:
                continue

for subject, subject_content in epochs.items():
    for session, session_content in subject_content.items():
        for trial, trial_content in session_content.items():
            for case, epoch in trial_content.items():
                print(subject, session, trial, case)
                plt.rcParams["figure.figsize"] = (15, 10)
                epoch.plot(
                    title=f'{subject}, {session}, {trial}, {case}'
                )
                plt.show()
                a = input()
                if str(a) == 'q':
                    os._exit(0)
                else:
                    print(a)

# path = './Source/Subjects/Ga_Fed_06/TF_planes/B1/RespCor/lm_vs_li.pkl'
# tf_planes = pickle.load(open(path, 'rb'))
# csp = tf_planes.csp[
#         list(tf_planes.csp.keys())[0]
#     ][
#         list(tf_planes.csp[
#                  list(tf_planes.csp.keys())[0]
#              ].keys())[0]
#     ]
# content_root = './'
# subjects_folder_path = os.path.join(content_root, 'Source/Subjects')
# subject_path = os.path.join(subjects_folder_path, 'Ga_Fed_06')
# raw_path = os.path.join(subject_path, 'Raw', 'ML_Subject05_P1_tsss_mc_trans.fif')
# resp_lock_lm_B1_epochs_path = os.path.join(subject_path, 'Epochs', 'RespCor_LM_B1_epochs.fif')
# resp_lock_lm_B1_epochs = mne.read_epochs(resp_lock_lm_B1_epochs_path)
# csp.plot_patterns(resp_lock_lm_B1_epochs.info)
# plt.show()