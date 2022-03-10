import os
import pickle
from dataclasses import dataclass
from typing import Dict, List
import warnings
import mne
import numpy as np
from mne.decoding import CSP
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from utils.console import Silence
from combiners import EpochsCombiner
from utils.storage_management import check_path
import re
import argparse
from mne.decoding import  CSP
from typing import Any
from cross_runs_TF_planes import CrossRunsTFScorer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='A script for applying the neural network "LFCNN" to the epoched data from gradiometers related to events for classification'
    )
    parser.add_argument('-cs', '--combine-sessions', type=str,
                        default='1-8', help='Sessions to combine in a format [from]-[to]')
    parser.add_argument('-ep', '--exclude-participants', type=str, nargs='+',
                        default=[], help='IDs of subjects to exclude')
    parser.add_argument('-l', '--lock', type=str,
                        default='RespCor', help='Stimulus lock to consider')
    parser.add_argument('-c', '--cases', type=str, nargs='+',
                        default=['LI', 'LM', 'RI', 'RM'], help='Cases to consider (must match epochs file names for the respective classes)')
    parser.add_argument('-sd', '--subjects-dir', type=str,
                        default=os.path.join(os.getcwd(), 'Source', 'Subjects'), help='Path to the subjects directory')
    parser.add_argument('--trials-name', type=str,
                        default='B', help='Name of trials')
    
    combined_sessions, \
    excluded_subjects, \
    lock, \
    cases, \
    subjects_dir, \
    sessions_name = vars(parser.parse_args()).values()
    
    combined_sessions = combined_sessions.split('-')
    from_, to = (
        int(re.findall(r'\d+', session)[-1])
        for session in combined_sessions
    )
    included_sessions = [f'{sessions_name}{i}' for i in range(from_, to + 1)]
    combined_sessions = f'{sessions_name}{included_sessions[0]}-{sessions_name}{included_sessions[-1]}'
    
    for subject_name in os.listdir(subjects_dir):
        
        if subject_name in excluded_subjects:
            continue
        
        train_acc, train_loss, val_acc, val_loss, test_acc, test_loss = (list() for _ in range(6))
        subject_path = os.path.join(subjects_dir, subject_name)
        epochs_path = os.path.join(subject_path, 'Epochs')
        epochs = {case: list() for case in cases}
        any_info = None
        
        for epochs_file in os.listdir(epochs_path):
            if lock not in epochs_file:
                continue
            
            session = re.findall(r'_{0}\d\d?'.format(sessions_name), epochs_file)[0][1:]
            
            if session not in included_sessions:
                continue
            
            for case in cases:
                if case in epochs_file:
                    with Silence(), warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        epochs_ = mne.read_epochs(os.path.join(epochs_path, epochs_file))
                        epochs_.resample(200)
                        
                        if any_info is None:
                            any_info = epochs_.info
                        
                        epochs[case].append(epochs_)
        
        epochs = dict(
                    zip(
                        epochs.keys(),
                        map(
                            mne.concatenate_epochs,
                            list(epochs.values())
                        )
                    )
                )
        resp_lock_li_epochs = epochs['LI']
        resp_lock_lm_epochs = epochs['LM']
        resp_lock_ri_epochs = epochs['RI']
        resp_lock_rm_epochs = epochs['RM']

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
            session_tf_planes_path = os.path.join(tf_planes_path, combined_sessions)
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
