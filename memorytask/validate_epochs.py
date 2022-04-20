import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import mne
from utils.console import Silence

if __name__ == '__main__':
    subjects_dir = './Source/MemoryTaskSubjects'
    for subject in os.listdir(subjects_dir):
        epochs_dir = os.path.join(subjects_dir, subject, 'Epochs')
        for epochs_file in os.listdir(epochs_dir):
            with Silence():
                print(f'{epochs_file}: {mne.read_epochs(os.path.join(epochs_dir, epochs_file)).get_data().shape}')