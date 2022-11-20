# %%

import os
import pickle
from typing import Optional
from typing import Union

import mne
import numpy as np
from mne import EpochsArray
from mne.datasets.sample import sample
from mne.decoding import LinearModel, get_coef
from mne.decoding import SlidingEstimator
from mne.decoding import cross_val_multiscore
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from combiners import EpochsCombiner
from utils.beep import Beeper
from utils.storage_management import read_or_write


@read_or_write(
    'fif',
    mne.read_source_spaces,
    mne.write_source_spaces,
    '../Source/Subjects/Az_Mar_05/Generated',
    'src'
)
def generate_src(subject_name: str, subjects_dir: str):
    return mne.setup_source_space(subject_name, subjects_dir=subjects_dir)


@read_or_write(
    'fif',
    mne.read_bem_solution,
    mne.write_bem_solution,
    '../Source/Subjects/Az_Mar_05/Generated',
    'bem'
)
def generate_bem(
        subject_name: str,
        subjects_dir: str,
        conductivity: Optional[tuple] = (.3,),
        ico: Optional[int] = 4
):
    return mne.make_bem_solution(
        mne.make_bem_model(
            subject=subject_name,
            ico=ico,
            conductivity=conductivity,
            subjects_dir=subjects_dir
        )
    )


@read_or_write(
    'fif',
    mne.read_forward_solution,
    mne.write_forward_solution,
    '../Source/Subjects/Az_Mar_05/Generated',
    'fwd'
)
def generate_forward(
        info: Union[str, mne.Info],
        trans: Union[str, dict, mne.Transform],
        src: Union[str, mne.SourceSpaces],
        bem: Union[str, dict],
        *args, **kwargs
):
    return mne.make_forward_solution(
        info,
        *args,
        trans=trans,
        src=src,
        bem=bem,
        **kwargs
    )


@read_or_write(
    'fif',
    mne.read_cov,
    mne.write_cov,
    '../Source/Subjects/Az_Mar_05/Generated',
    'cov'
)
def combine_covariance(*args: EpochsArray, tmin: Optional[float] = -.5, tmax: Optional[float] = .5, **kwargs):
    covs = [
        mne.compute_covariance(epochs, tmin=tmin, tmax=tmax, **kwargs)
        for epochs in args
    ]
    out = None
    for i in range(len(covs)):
        if i == 0:
            out = covs[i]
        else:
            out += covs[i]
    return out


@read_or_write(
    'fif',
    mne.minimum_norm.read_inverse_operator,
    mne.minimum_norm.write_inverse_operator,
    '../Source/Subjects/Az_Mar_05/Generated',
    'inv'
)
def compute_inverse(info: Union[str, mne.Info], fwd: Union[str, mne.Forward], cov: mne.Covariance, **kwargs):
    return mne.minimum_norm.make_inverse_operator(
        info, fwd, cov, **kwargs)


@read_or_write(
    'pkl',
    lambda path: pickle.load(open(path, 'rb')),
    lambda path, stc: pickle.dump(stc, open(path, 'wb')),
    '../Source/Subjects/Az_Mar_05/Generated',
    'stc'
)
def compute_stc(evoked: mne.EvokedArray,
                inv: mne.minimum_norm.InverseOperator,
                lambda2: float, method: str,
                *args, **kwargs):
    return mne.minimum_norm.apply_inverse(evoked, inv, lambda2, method, *args, **kwargs)


if __name__ == '__main__':
    # speakers
    sad_beep = Beeper(duration=[.1, .15, .25], frequency=[280, 240, 190], repeat=3)
    happy_beep = Beeper(duration=[.1, .1, .15, .25], frequency=[400, 370, 470, 500], repeat=4)

    # paths
    content_root = '../'
    subjects_folder_path = os.path.join(content_root, 'Source/Subjects')
    subject_path = os.path.join(subjects_folder_path, 'Az_Mar_05')
    raw_file_path = os.path.join(subject_path, 'Raw', 'ML_Subject05_P1_tsss_mc_trans.fif')
    raw_path = os.path.join(subject_path, 'Raw', 'ML_Subject05_P1_tsss_mc_trans.fif')
    resp_lock_lm_B1_epochs_path = os.path.join(subject_path, 'Epochs_old', 'resp_lock_lm_B1_epochs.fif')
    resp_lock_rm_B1_epochs_path = os.path.join(subject_path, 'Epochs_old', 'resp_lock_rm_B1_epochs.fif')
    resp_lock_li_B1_epochs_path = os.path.join(subject_path, 'Epochs_old', 'resp_lock_li_B1_epochs.fif')
    resp_lock_ri_B1_epochs_path = os.path.join(subject_path, 'Epochs_old', 'resp_lock_ri_B1_epochs.fif')

    # readers
    original_data = mne.io.read_raw_fif(raw_file_path)
    original_info = original_data.info
    resp_lock_lm_B1_epochs = mne.read_epochs(resp_lock_lm_B1_epochs_path)
    resp_lock_rm_B1_epochs = mne.read_epochs(resp_lock_rm_B1_epochs_path)
    resp_lock_li_B1_epochs = mne.read_epochs(resp_lock_li_B1_epochs_path)
    resp_lock_ri_B1_epochs = mne.read_epochs(resp_lock_ri_B1_epochs_path)

    data_path = sample.data_path()
    subjects_dir = data_path + '/subjects'
    subject = 'sample'
    clf = make_pipeline(StandardScaler(),
                        LinearModel(LogisticRegression(solver='lbfgs')))

    combiner = EpochsCombiner(resp_lock_lm_B1_epochs, resp_lock_li_B1_epochs, resp_lock_rm_B1_epochs,
                              resp_lock_ri_B1_epochs)
    combiner.combine((0, 1), (2, 3), shuffle=True)

    # time-decoder

    time_decod = SlidingEstimator(clf, n_jobs=1, scoring='roc_auc', verbose=True)

    scores = np.mean(
        cross_val_multiscore(time_decod, combiner.X, combiner.Y, cv=5, n_jobs=1),
        axis=0
    )

    times = np.linspace(-.5, .5, scores.shape[0])

    time_decod.fit(combiner.X, combiner.Y)

    coef = get_coef(time_decod, 'patterns_', inverse_transform=True)
    evoked_time_gen = mne.EvokedArray(coef, resp_lock_rm_B1_epochs.info, tmin=times[0])
    joint_kwargs = dict(ts_args=dict(time_unit='s'),
                        topomap_args=dict(time_unit='s'))
    evoked_time_gen.plot_joint(times=np.arange(0., .500, .100), title='patterns',
                               **joint_kwargs)

    src = generate_src(subject, subjects_dir=subjects_dir)
    conductivity = (0.3, 0.006, 0.3)
    bem = generate_bem(subject, subjects_dir, conductivity)

    fwd = generate_forward(
        resp_lock_lm_B1_epochs.info,
        trans='fsaverage',
        src=src, bem=bem,
        meg=True, eeg=False,
        mindist=5.0,
        n_jobs=1,
        verbose=True
    )

    cov = combine_covariance(
        resp_lock_lm_B1_epochs,
        resp_lock_li_B1_epochs,
        resp_lock_rm_B1_epochs,
        resp_lock_ri_B1_epochs,
    )

    inv = compute_inverse(evoked_time_gen.info, fwd, cov, loose=0.)

    stc = compute_stc(evoked_time_gen, inv, 1. / 9., 'dSPM')
    surfer_kwargs = dict(
        hemi='lh', subjects_dir=subjects_dir,
        clim=dict(kind='steps', lims=[8, 12, 15]), views='lateral',
        initial_time=0.09, time_unit='s', size=(800, 800),
        smoothing_steps=5)
    brain = stc.plot(hemi='split', views=('lat', 'med'), initial_time=0.1,
                     subjects_dir=subjects_dir)
