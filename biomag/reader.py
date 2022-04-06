import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import mne
from typing import *
from combiners import EpochsCombiner
from utils.storage_management import check_path
import tensorflow as tf
import mneflow as mf
from LFCNN_decoder import save_parameters, Predictions, WaveForms, SpatialParameters, TemporalParameters, ComponentsOrder, compute_temporal_parameters
import numpy as np
import scipy as sp
import pandas as pd
from LFRNN_decoder import LFRNN


class_names = ['S', 'T']
biomag_home = os.path.join(os.path.dirname(parentdir), 'BIOMAG')
biomag_data = os.path.join(biomag_home, 'BIOMAG_DATA')
classification_name = 'biomag'

for epochs_file in os.listdir(biomag_data):
    epochs = mne.read_epochs(os.path.join(biomag_data, epochs_file))
    epochs.pick_types(meg=True)
    print(epochs.info)
    print(epochs.info['ch_names'])
    print(len(epochs.info['ch_names']))