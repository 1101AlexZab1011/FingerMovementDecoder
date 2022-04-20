import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from typing import Any
import pickle
import mne
from utils.storage_management import check_path
import argparse
from mat_to_fif import read_mat_epochs, read_pkl, save_epochs
from utils.console.colored import success, alarm, warn

def read_info(path: str) -> mne.Info:
    ext = path.split('.')[-1]
    if ext == 'pkl':
        info = read_pkl(path)
    elif ext == 'fif':
        info = mne.io.read_info(path)
    else:
        raise ValueError(f'Wrong extension for Info file: {ext}')
    
    if isinstance(info, mne.Info):
        return info
    elif isinstance(info, (list, tuple)):
        assert len(info) == 1, f'There are several ({len(info)}) Info objects at {path}'
        return info[0]
    else:
        raise ValueError(f'Info has wrong type: {type(info)}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', '--exclude-participants', type=str, nargs='+',
                        default=[], help='IDs of subjects to exclude')
    parser.add_argument('-sd', '--source-dir', type=str,
                        default=os.path.join(os.getcwd(), 'Source', 'Data'), help='Path to the data directory')
    parser.add_argument('-sp', '--savepath', type=str,
                        default=os.path.join(os.getcwd(), 'Source', 'MemoryTaskSubjects'), help='Path to the subjects directory')
    parser.add_argument('-tf', '--target-file', type=str,
                        default='dat_ft.mat', help='Filename to transform')
    parser.add_argument('-if', '--info-file', type=str,
                        default=None, help='path to any suitable info file')
    parser.add_argument('--name', type=str,
                        default='', help='Name of saved file')
    parser.add_argument('--postfix', type=str,
                        default='', help='String to append to saved file name')
    parser.add_argument('--prefix', type=str,
                        default='', help='String to set in the start of saved file name')
    
    excluded_subjects, \
    source_dir, \
    subjects_dir, \
    target_file, \
    info_path, \
    filename, \
    postfix, \
    prefix = vars(parser.parse_args()).values()
    
    if info_path is None or not os.path.exists(info_path):
        raise OSError(f'Info file by path {info_path} not found')
    
    check_path(subjects_dir)
    n_subjects = len(os.listdir(source_dir))
    
    if n_subjects == 0:
        raise OSError(f'There is not a single subject directory found by the path {source_dir}')
    
    if os.path.isdir(info_path):
        n_info_files = len(os.listdir(info_path))
        if n_info_files != n_subjects:
            raise ValueError(f'Number of info files ({n_info_files}) and number of subjects ({n_subjects}) are inconsistent')
        info = [read_info(info_file) for info_file in os.listdir(info_path)]
    else:
        info = read_info(info_path)
        info = [info for _ in range(n_subjects)]
    
    for subject, subject_info in zip(os.listdir(source_dir), info):
        
        if subject in excluded_subjects:
            warn(f'Skipping subject {subject}...')
            continue
        
        subject_dir = os.path.join(source_dir, subject)
        subject_save_dir = os.path.join(subjects_dir, subject)
        
        if not os.path.isdir(subject_dir):
            continue
        
        epochs_dir = os.path.join(subject_save_dir, 'Epochs')
        check_path(subject_save_dir, epochs_dir)
        
        for stimulus in os.listdir(subject_dir):
            mat_file = os.path.join(subject_dir, stimulus, target_file)
            
            try:
                epochs = read_mat_epochs(mat_file, info=subject_info)
                if isinstance(epochs, list) and len(epochs) == 1:
                    epochs = epochs[0]
            except ValueError as err:
                alarm(f'Epochs for {subject} {stimulus} can not be read '
                        f'due to {type(err)}\n{err}\n', True)
                continue
            
            # success(f'{subject}, {stimulus}: Epochs successfully read')
            savepath = os.path.join(epochs_dir, '_'.join(
                list(
                    filter(
                        lambda str_: str_ != '',
                        [prefix, filename, stimulus, f'{postfix}.fif']
                    )
                ),
            ))
            save_epochs(savepath, epochs)
            success(f'{subject}, {stimulus}: Epochs successfully saved')
