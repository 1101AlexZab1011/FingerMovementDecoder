import os
import argparse
import pickle
import re
import warnings
from shutil import rmtree
from typing import NoReturn, Union, Any
import mne
from utils.console import Silence
from utils.console.colored import ColoredText, bold, success, alarm, warn
from utils.console.spinner import spinner
from utils.storage_management import check_path


@spinner(prefix=lambda path: f'Reading {path}... ')
def read_pkl(path: str) -> Any:
    with open(
        path,
        'rb'
    ) as file:
        content = pickle.load(
            file
        )
    return content


@spinner(prefix=lambda _, path: f'Saving file into {path}... ')
def save_pkl(content: Any, path: str) -> NoReturn:
    if path[-4:] != '.pkl':
        raise OSError(f'Pickle file must have extension ".pkl", but it has "{path[-4:]}"')
    pickle.dump(content, open(path, 'wb'))


@spinner(
    prefix=lambda path, info=None:
    f'Reading '
    f'"{path.split("/")[-2]}/'
    f'{path.split("/")[-1]}" file for '
    f'{path.split("/")[-3]}'  # [-5]}'
    f'... '
)
def read_mat_epochs(path, *, info):
    with Silence():
        return mne.read_epochs_fieldtrip(path, info=info)


@spinner(
    prefix=lambda path, _:
    f'Saving '
    f'"{path.split("/")[-1]}" file for '
    f'{path.split("/")[-3]}'
    f'... '
)
def save_epochs(path, epochs: Union[mne.Epochs, mne.EpochsArray]) -> None:
    with warnings.catch_warnings(), Silence():
        warnings.simplefilter("ignore")
        epochs.save(path, overwrite=True)


def y_n_answer(ans: str) -> bool:
    if ans in ['Y', 'y']:
        return True
    elif ans in ['N', 'n']:
        return False
    else:
        raise ValueError(f'Incorrect answer: {ans}')


def y_n_question(msg: str) -> bool:
    bold(f'{msg} [Y/n] ', end='')
    ans = input()
    return y_n_answer(ans)


if __name__ == "__main__":

    warning = ColoredText().color('y').style('b')

    parser = argparse.ArgumentParser(
        description='Script to read epochs in ".mat" format and save them in ".fif" format. '
                    'It is assumed that the files lie in directories that hierarchically declare:'
                    ' subject id, stimulus lock, event recorded in epochs and recording session.'
    )
    parser.add_argument('-el', '--exclude-locks', type=str, nargs='+',
                        default=[], help='Locks names to exclude')
    parser.add_argument('-es', '--exclude-sessions', type=str, nargs='+',
                        default=[], help='Sessions to exclude')
    parser.add_argument('-ep', '--exclude-participants', type=str, nargs='+',
                        default=[], help='IDs of subjects to exclude')
    parser.add_argument('-l', '--locks', type=str, nargs='+',
                        default=['RespLock', 'StimLock'], help='Locks to consider')
    parser.add_argument('-c', '--cases', type=str, nargs='+',
                        default=['LI', 'LM', 'RI', 'RM'], help='Cases to consider')
    parser.add_argument('-sd', '--subjects-dir', type=str,
                        default=os.path.join(os.getcwd(), 'Source', 'Subjects'),
                        help='Path to the subjects directory')
    parser.add_argument('-t', '--trials', type=int,
                        default=10, help='Number of trials')
    parser.add_argument('-m', '--manually', help='if true, ask how to perform each step',
                        action='store_true')
    parser.add_argument('--trials-names', type=str,
                        default='B', help='Names of trials')
    parser.add_argument('--delete-unnecessary', help='Delete all files that are no longer needed',
                        action='store_true')
    excluded_locks, \
        excluded_sessions, \
        excluded_subjects, \
        all_locks, \
        cases, \
        subjects_dir, \
        n_trials, \
        manually, \
        sessions_name, \
        delete_unnecessary = vars(parser.parse_args()).values()

    sessions = [
        f'{sessions_name}{i + 1}'
        for i in range(n_trials) if f'B{i}' not in excluded_sessions
    ]

    locks = [lock for lock in all_locks if lock not in excluded_locks]

    for subject in os.listdir(subjects_dir):

        if subject in excluded_subjects:
            print(f'Skipping subject {subject}...')
            continue

        subject_path = os.path.join(subjects_dir, subject)
        info_path = os.path.join(subject_path, 'Info')
        subject_info = None
        epochs_path = os.path.join(subject_path, 'Epochs')
        info_loaded = False

        if os.path.exists(info_path):
            info_files = os.listdir(info_path)

            if info_files:
                print(f'Info for {subject} are found')

                if manually:
                    read_info = y_n_question(f'Read Info for {subject}?')

                else:
                    read_info = True

                if read_info:

                    if len(info_files) > 1:
                        alarm(f'Several Info files found for {subject}:', True)

                        for i, info_file in enumerate(info_files):
                            print(f'\t{i + 1}: {info_file}')
                        bold('Enter a number of the file you want to use: ', end='')
                        n_file = int(input()) - 1

                    else:
                        n_file = 0

                    subject_info = read_pkl(os.path.join(info_path, info_files[n_file]))

                    if isinstance(subject_info, list) and len(subject_info) == 1:
                        subject_info = subject_info[0]

                    info_loaded = True
                    success('Info file successfully loaded')
                    del read_info, info_files

        if not info_loaded:
            raw_dir = os.path.join(subject_path, 'Raw')
            raw_files = os.listdir(raw_dir)

            if not raw_files:
                raise OSError('No Raw files found')
            elif len(raw_files) > 1:
                alarm(f'Several Raw files found for {subject}:', True)

                for i, raw_file in enumerate(raw_files):
                    print(f'\t{i + 1}: {raw_file}')

                bold('Enter a number of the file you want to use: ', end='')
                n_file = int(input()) - 1
                raw_file_name = raw_files[n_file]
                raw = mne.io.read_raw_fif(os.path.join(raw_dir, raw_file_name))
            else:
                raw_file_name = raw_files[0]
                raw = mne.io.read_raw_fif(os.path.join(raw_dir, raw_file_name))

            subject_info = raw.info

            if manually:
                save_info = y_n_question(f'Save Info of {subject} at {info_path}?')
            else:
                save_info = True

            if save_info:
                check_path(info_path)
                save_pkl(subject_info, os.path.join(info_path, f'{raw_file_name[:-4]}_info.pkl'))
                success('Info file successfully saved')
                info_saved = True
                del save_info, raw, info_path
            else:
                info_saved = False

            if info_saved:

                if manually:
                    delete_raw = y_n_question(
                        f'Remove Raw-file of {subject} '
                        f'at {os.path.join(raw_dir, raw_file_name)}?'
                    )
                elif delete_unnecessary:
                    delete_raw = True
                else:
                    delete_raw = False

                if delete_raw:
                    rmtree(raw_dir)
                    warn(f'Raw file of {subject} has been removed', True)

                del info_saved, raw_dir, raw_file_name

        epoch_path_exists = check_path(epochs_path)

        for lock in locks:
            if lock in excluded_locks:
                print(f'Skipping {lock}...')

            subject_lock = os.path.join(subject_path, lock)

            for case in os.listdir(subject_lock):
                if case not in cases:
                    warn(f'Unexpected case found in {subject_lock}: {case}', True)
                    warn('Skipping this case...')
                    continue

                subject_case = os.path.join(subject_lock, case)

                for cor_case_session in sorted(
                        os.listdir(subject_case),
                        key=lambda item: int(
                            re.findall(r'(_B\d\d?)', item)[0][2:]
                        )
                ):
                    session = re.findall(r'(_B\d\d?)', cor_case_session)[0][1:]
                    if session in excluded_sessions:
                        print(f'Skipping session {session}...')
                        continue
                    if os.path.exists(os.path.join(
                            epochs_path, f'{cor_case_session}_epochs.fif'
                    )):
                        alarm(f'Epochs for {subject} {lock} {case} {session} already exist')

                        if manually:
                            transform_epoch = y_n_question(
                                f'Overwrite epoch "{cor_case_session}_epochs.fif"?'
                            )
                        else:
                            transform_epoch = False
                    else:
                        transform_epoch = True

                    if transform_epoch:
                        dat_mat_path = os.path.join(subject_case, cor_case_session, 'dat_ft.mat')
                        try:
                            epochs = read_mat_epochs(dat_mat_path, info=subject_info)
                            if isinstance(epochs, list) and len(epochs) == 1:
                                epochs = epochs[0]
                        except ValueError as err:
                            alarm(
                                f'Epochs for {subject} {lock} {case} {session} can not be read '
                                f'due to {type(err)}\n{err}\n',
                                True
                            )
                            continue
                        success('Epochs successfully read:')
                        print(epochs)
                        check_path(epochs_path)
                        epochs_to_save = os.path.join(
                            epochs_path, f'{cor_case_session}_epochs.fif'
                        )
                        save_epochs(
                            epochs_to_save,
                            epochs
                        )
                        success(f'Epochs successfully saved at {epochs_to_save}')

                        if manually:
                            delete_mat_epochs = y_n_question(
                                f'Remove ".mat" epochs file for '
                                f'{subject} {lock} {case} {session} at {dat_mat_path}?'
                            )
                        elif delete_unnecessary:
                            delete_mat_epochs = True
                        else:
                            delete_mat_epochs = False

                        if delete_mat_epochs:
                            rmtree(os.path.join(subject_case, cor_case_session))
                            warn(
                                f'Epochs file in ".mat" format for {subject} '
                                f'{lock} {case} {session} has been removed',
                                True
                            )
