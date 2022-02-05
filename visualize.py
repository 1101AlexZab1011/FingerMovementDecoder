import numpy as np
import matplotlib.pyplot as plt
import os
from mne import create_info
from mne.time_frequency import AverageTFR
from cross_runs_TF_planes import CrossRunsTFScorer
import pickle
from utils.storage_management import check_path

tmin, tmax = -.500, .500
n_cycles = 14
min_freq = 5.
max_freq = 70.
n_freqs = 7
freqs_range = np.logspace(np.log10(min_freq), np.log10(max_freq), n_freqs)
window_spacing = (n_cycles / np.max(freqs_range) / 2.)
centered_w_times = np.arange(tmin, tmax, window_spacing)[1:]
n_windows = len(centered_w_times)

root = './'
subjects_dir = os.path.join(root, 'Source/Subjects')
pics_dir = os.path.join(root, 'Source/Pictures')
check_path(pics_dir)
tf_planes_pics_dir = os.path.join(pics_dir, 'TF_planes')
check_path(tf_planes_pics_dir)

for subject_name in os.listdir(subjects_dir):
    tf_planes_path = os.path.join(subjects_dir, subject_name, 'TF_planes')
    for session in os.listdir(tf_planes_path):
        session_path = os.path.join(tf_planes_path, session)
        for lock in os.listdir(session_path):
            lock_path = os.path.join(session_path, lock)
            for cross_runs_TF_scorer_file in os.listdir(lock_path):
                cross_runs_TF_scorer = pickle.load(
                    open(
                        os.path.join(lock_path, cross_runs_TF_scorer_file), 'rb')
                )
                av_tfr = AverageTFR(create_info(['freq'], 1000), cross_runs_TF_scorer.mean()[np.newaxis, :],
                                centered_w_times, freqs_range[1:], 1)
                chance = .5
                fig = av_tfr.plot(
                    [0],
                    vmin=chance,
                    title=f"{subject_name}, {session}, {lock}, {cross_runs_TF_scorer_file[:-4]}",
                    cmap=plt.cm.Reds,
                    show=False
                )
                plt.savefig(
                    os.path.join(
                        tf_planes_pics_dir, f'{subject_name}_{session}_{lock}_{cross_runs_TF_scorer_file[:-4]}.png'
                    ),
                )
