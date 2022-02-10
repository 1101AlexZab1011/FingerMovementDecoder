import logging
import os
import pickle
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Union, Optional
import argparse
import mne
import numpy as np
from mne.decoding import CSP
from sklearn.base import ClassifierMixin, RegressorMixin, BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection._split import _BaseKFold
from sklearn.pipeline import make_pipeline

from combiners import EpochsCombiner
from cross_runs_TF_planes import CrossRunsTFScorer
from utils.console import Silence
from utils.console.asynchrony import Handler, async_generator, closed_async
from utils.console.colored import success, warn, alarm, ColoredText
from utils.console.progress_bar import run_spinner, Spinner, ProgressBar, SpinnerRunner, Progress
from utils.storage_management import check_path
import re

from utils.structures import Deploy


def score_planes(
        combiner: EpochsCombiner,
        first_class_indices: tuple[int, ...],
        second_class_indices: tuple[int, ...],
        n_freqs: int,
        n_windows: int,
        n_cycles: int,
        freq_ranges: list[tuple[float, float]],
        centered_w_times: np.ndarray,
        tmin: float,
        tmax: float,
        csp_cache: Optional[dict] = None,
        tf_acc_cache: Optional[dict] = None,
        clf: Optional[Union[BaseEstimator, ClassifierMixin, RegressorMixin]] = LogisticRegression(),
        cv: Optional[_BaseKFold] = StratifiedKFold(n_splits=5, shuffle=True),
        identifier: Optional[str] = 'Current task'
) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if csp_cache is None:
            csp_cache = dict()
        if tf_acc_cache is None:
            tf_acc_cache = dict()

        tf_scores = np.zeros((n_freqs - 1, n_windows))

        for freq, (fmin, fmax) in enumerate(freq_ranges):

            w_size = n_cycles / ((fmax + fmin) / 2.)

            combiner \
                .switch_data('original') \
                .filter(l_freq=fmin, h_freq=fmax, skip_by_annotation='edge') \
                .combine(first_class_indices, second_class_indices, shuffle=True)

            for t, w_time in enumerate(centered_w_times):
                w_tmin = w_time - w_size / 2.
                w_tmax = w_time + w_size / 2.

                if w_tmin < tmin:
                    w_tmin = tmin
                if w_tmax > tmax:
                    w_tmax = tmax

                if w_tmin > w_tmax:
                    raise ValueError(f'w_tmin is greater than w_tmax: {w_tmin=}, {w_tmax=}')

                # print(f'{identifier}: {w_tmin}-{w_tmax}ms at {fmin}-{fmax}Hz')
                
                combiner \
                    .switch_data('filtered') \
                    .crop(tmin=w_tmin, tmax=w_tmax) \
                    .combine(first_class_indices, second_class_indices, shuffle=True)

                if not f'{fmin}-{fmax}' in csp_cache:
                    csp_cache.update({f'{fmin}-{fmax}': dict()})
                if not f'{w_tmin}-{w_tmax}' in csp_cache[f'{fmin}-{fmax}']:
                    csp = CSP(n_components=5, reg='shrinkage', rank='full')

                    fitted = False
                    while not fitted:
                        try:
                            csp.fit(combiner.X, combiner.Y)
                            fitted = True
                        except Exception:
                            continue

                    csp_cache.update({f'{fmin}-{fmax}': {f'{w_tmin}-{w_tmax}': csp}})

                if not f'{fmin}-{fmax}' in tf_acc_cache:
                    tf_acc_cache.update({f'{fmin}-{fmax}': dict()})
                if not f'{w_tmin}-{w_tmax}' in tf_acc_cache[f'{fmin}-{fmax}']:
                    tf_acc_cache[f'{fmin}-{fmax}'].update({f'{w_tmin}-{w_tmax}': list()})

                scored = False
                while not scored:
                    tf_scores[freq, t] = np.mean(
                        cross_val_score(
                            estimator=clf,
                            X=combiner.X,
                            y=combiner.Y,
                            scoring='roc_auc',
                            cv=cv,
                            n_jobs=1,
                        ),
                        axis=0
                    )
                    if not np.isnan(tf_scores[freq, t]):
                        tf_acc_cache[f'{fmin}-{fmax}'][f'{w_tmin}-{w_tmax}'].append(
                            tf_scores[freq, t])
                        scored = True

        return tf_scores

if __name__ == '__main__':
    logging.getLogger(__name__).setLevel(logging.CRITICAL)
    logging.getLogger('mne').setLevel(logging.CRITICAL)
    bar = ProgressBar()
    EXCLUDED_SESSIONS = ['B1', 'B10']
    EXCLUDED_SUBJECTS = [] #['Az_Mar_05', 'Ga_Fed_06']
    EXCLUDED_LOCKS = ['StimCor']
    
    parser = argparse.ArgumentParser(
        description='A script for applying the neural network "LFCNN" to the epoched data from gradiometers related to events for classification'
    )
    parser.add_argument('-es', '--exclude-sessions', type=str, nargs='+',
                        default=[], help='Sessions to exclude')
    parser.add_argument('-ep', '--exclude-participants', type=str, nargs='+',
                        default=[], help='IDs of subjects to exclude')
    parser.add_argument('-el', '--exclude-locks', type=str, nargs='+',
                        default=[], help='IDs of subjects to exclude')
    
    
    EXCLUDED_SESSIONS, \
    EXCLUDED_SUBJECTS, \
    EXCLUDED_LOCKS = vars(parser.parse_args()).values()

    content_root = './'
    subjects_folder_path = os.path.join(content_root, 'Source', 'Subjects')

    for subject_name in os.listdir(subjects_folder_path):
        if subject_name in EXCLUDED_SUBJECTS:
            warn(f'Skip subject {subject_name}')
            continue
        subject_path = os.path.join(subjects_folder_path, subject_name)

        subject_epochs = os.path.join(subject_path, 'Epochs')
        info_path = os.path.join(subject_path, 'Info')
        for address, dirs, files in os.walk(info_path):
            if len(files) != 1:
                raise OSError(
                    f'Several ({len(files)}) info files detected at {info_path}'
                )
            subject_info = pickle.load(
                open(
                    os.path.join(info_path, files[0]),
                    'rb'
                )
            )
            epochs = dict()
            for epoch in os.listdir(subject_epochs):
                session = re.findall(r'(_B\d\d?)', epoch)[0][1:]
                if session in EXCLUDED_SESSIONS:
                    continue
                if not session in epochs:
                    epochs.update({
                        session: {}
                    })
                current_lock = None
                for lock in ['RespCor', 'StimCor']:
                    if lock in epoch:
                        current_lock = lock
                if current_lock in EXCLUDED_LOCKS:
                    continue
                if current_lock not in epochs[session] and current_lock is not None:
                    epochs[session].update({current_lock: {}})
                elif current_lock is None:
                    raise ValueError('This lock is not RespCor nor StimCor')
                for case in ['LI', 'LM', 'RI', 'RM']:
                    if case in epoch:
                        if case not in epochs[session][current_lock]:
                            print(f'Reading {session} {current_lock} {case} epochs...', end='')
                            with Silence(), warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                epochs[session][current_lock].update(
                                    {case: mne.read_epochs(os.path.join(subject_epochs, epoch))})
                            success(' OK')
                        else:
                            alarm(f'\nThe case \"{case}\" already in epoch {epoch}, skipping via the conflict\n')

            for session in epochs:
                for lock in epochs[session]:
                    resp_lock_li_epochs = epochs[session][lock]['LI']
                    resp_lock_lm_epochs = epochs[session][lock]['LM']
                    resp_lock_ri_epochs = epochs[session][lock]['RI']
                    resp_lock_rm_epochs = epochs[session][lock]['RM']

                    clf = make_pipeline(CSP(n_components=4, reg='shrinkage', rank='full'),
                                        LogisticRegression(penalty='l1', solver='saga'))

                    n_splits = 5
                    cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
                    tmin, tmax = -.500, .500
                    n_cycles = 14
                    n_iters = 25
                    min_freq = 5.
                    max_freq = 70.
                    n_freqs = 7
                    freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), n_freqs)
                    freq_ranges = list(zip(freqs[:-1], freqs[1:]))
                    window_spacing = (n_cycles / np.max(freqs) / 2.)
                    centered_w_times = np.arange(tmin, tmax, window_spacing)[1:]
                    n_windows = len(centered_w_times)

                    for combiner, first_class_indices, second_class_indices, name in \
                            zip(
                                (
                                        # left vs right
                                        EpochsCombiner(resp_lock_lm_epochs, resp_lock_li_epochs, resp_lock_rm_epochs,
                                                        resp_lock_ri_epochs),
                                        # one finger two sides
                                        EpochsCombiner(resp_lock_lm_epochs, resp_lock_li_epochs),
                                        EpochsCombiner(resp_lock_rm_epochs, resp_lock_ri_epochs)
                                ),
                                (
                                        # left vs right
                                        (0, 1),
                                        # one finger two sides
                                        0,
                                        0
                                ),
                                (
                                        # left vs right
                                        (2, 3),
                                        # one finger two sides
                                        1,
                                        1,
                                ),
                                (
                                        # left vs right
                                        'left_vs_right',
                                        # one finger two sides
                                        'lm_vs_li',
                                        'rm_vs_ri'
                                )

                            ):
                        tf_acc_cache = dict()
                        csp_cache = dict()
                        cross_tf_scores = list()
                        tf_planes_path = os.path.join(subject_path, 'TF_planes')
                        check_path(tf_planes_path)
                        session_tf_planes_path = os.path.join(tf_planes_path, session)
                        check_path(session_tf_planes_path)
                        lock_tf_planes_path = os.path.join(session_tf_planes_path, lock)
                        check_path(lock_tf_planes_path)
                        out_path = os.path.join(lock_tf_planes_path, f'{name}.pkl')
                        if os.path.exists(out_path):
                            warn(f'The file {out_path} already exists, skipping...')
                            continue
                        bar = ProgressBar()
                        processes = [
                            Deploy(
                                run_spinner,
                                Deploy(
                                    score_planes,
                                    combiner,
                                    first_class_indices,
                                    second_class_indices,
                                    n_freqs,
                                    n_windows,
                                    n_cycles,
                                    freq_ranges,
                                    centered_w_times,
                                    tmin,
                                    tmax,
                                    csp_cache,
                                    tf_acc_cache,
                                    identifier=f'Iteration {i}'
                                ),
                                Spinner(prefix=f'Iteration {i}:', report_message=f'Spinner {i}: Done'),
                                delete_final=True,
                                bar=bar
                            )
                            for i in range(n_iters)
                        ]
                        n_progress = bar.add_progress(
                            Progress(n_iters, prefix=f'Processing {name} case for subject {subject_name}: ', length=25),
                            return_index=True,
                        )
                        handler = Handler(processes, 5)
                        try:
                            for i, tasks in enumerate(async_generator(handler=handler)):
                                success(f'Iteration {i} processed')
                                for task in list(tasks):
                                    cross_tf_scores.append(
                                        task.result()
                                    )
                                bar(n_progress)
                                
                        except KeyboardInterrupt:
                            print()
                            os._exit(0)

                        crtfs = CrossRunsTFScorer(np.array(cross_tf_scores), tf_acc_cache, csp_cache)
                        with open(out_path, 'wb') as f:
                            pickle.dump(crtfs, f)
