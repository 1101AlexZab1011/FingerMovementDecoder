import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import os
import pickle
from dataclasses import dataclass
from typing import Dict, List

import mne
import numpy as np
from mne.decoding import CSP
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

from combiners import EpochsCombiner
from utils.storage_management import check_path
import re

from mne.decoding import  CSP
from typing import Any

@dataclass
class CrossRunsTFScorer:
    tf_scores: np.ndarray
    accuracy_cache: Dict[str, Dict[str, List[float]]]
    csp: Dict[str, Dict[str, List[CSP]]]

    def mean(self):
        return np.mean(self.tf_scores, axis=0)

    def std(self):
        return np.std(self.tf_scores, axis=0)

    def tf_windows_mean(self):
        return {
            freq: {
                time: np.mean(np.array(self.accuracy_cache[freq][time]))
                for time in self.accuracy_cache[freq]
            }
            for freq in self.accuracy_cache
        }

    def tf_windows_std(self):
        return {
            freq: {
                time: np.std(np.array(self.accuracy_cache[freq][time]))
                for time in self.accuracy_cache[freq]
            }
            for freq in self.accuracy_cache
        }


if __name__ == '__main__':
    INCLUDED_SESSIONS = [f'B{i+1}' for i in range(10)]
    EXCLUDED_SUBJECTS = []  # ['Az_Mar_05', 'Fe_To_08', 'Ga_Fed_06']
    EXCLUDED_LOCKS = ['StimCor']

    content_root = './'
    subjects_folder_path = os.path.join(content_root, 'Source/Subjects')

    for subject_name in os.listdir(subjects_folder_path):
        if subject_name in EXCLUDED_SUBJECTS:
            print(f'Skip subject {subject_name}')
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
                if session not in INCLUDED_SESSIONS:
                    continue
                if session not in epochs:
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
                            epochs[session][current_lock].update(
                                {case: mne.read_epochs(os.path.join(subject_epochs, epoch))})
                        else:
                            print(
                                f'\nThe case \"{case}\" already in '
                                f'epoch {epoch}, skipping via the conflict\n'
                            )

            for session in epochs:
                for lock in epochs[session]:
                    resp_lock_li_epochs = epochs[session][lock]['LI']
                    resp_lock_lm_epochs = epochs[session][lock]['LM']
                    resp_lock_ri_epochs = epochs[session][lock]['RI']
                    resp_lock_rm_epochs = epochs[session][lock]['RM']

                    clf = make_pipeline(
                        CSP(n_components=4, reg='shrinkage', rank='full'),
                        LogisticRegression(penalty='l1', solver='saga')
                    )

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
                                EpochsCombiner(
                                    resp_lock_lm_epochs,
                                    resp_lock_li_epochs,
                                    resp_lock_rm_epochs,
                                    resp_lock_ri_epochs
                                ),
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
                            print(f'The file {out_path} already exists, skipping...')
                            continue
                        for i in range(n_iters):
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

                                    combiner \
                                        .switch_data('filtered') \
                                        .crop(tmin=w_tmin, tmax=w_tmax) \
                                        .combine(first_class_indices, second_class_indices, shuffle=True)

                                    if not f'{fmin}-{fmax}' in csp_cache:
                                        csp_cache.update({f'{fmin}-{fmax}': dict()})
                                    if not f'{w_tmin}-{w_tmax}' in csp_cache[f'{fmin}-{fmax}']:
                                        csp = CSP(n_components=4, reg='shrinkage', rank='full')

                                        fitted = False
                                        while not fitted:
                                            try:
                                                csp.fit(combiner.X, combiner.Y)
                                                fitted = True
                                            except Exception:
                                                continue

                                        csp_cache[f'{fmin}-{fmax}'].update({f'{w_tmin}-{w_tmax}': csp})

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

                            cross_tf_scores.append(tf_scores)

                        cross_tf_scores_np = np.array(cross_tf_scores)

                        crtfs = CrossRunsTFScorer(np.array(cross_tf_scores), tf_acc_cache, csp_cache)
                        with open(out_path, 'wb') as f:
                            pickle.dump(crtfs, f)
