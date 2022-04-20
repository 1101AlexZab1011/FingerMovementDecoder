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
        out = dict()
        for epochs_file in os.listdir(epochs_dir):
            with Silence():
                shape = mne.read_epochs(os.path.join(epochs_dir, epochs_file)).get_data().shape
            out[epochs_file[:9]] = shape[0]
        for sti, samp in out.items():
            print(f'{sti}: {samp}', end=';\t')
        print()