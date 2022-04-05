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
class EpochsCollector(object): 
    def __init__(self):
        self._data = dict()
    
    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, content: tuple[str, str, mne.Epochs]):
        
        subject_name, event_type, epochs = content
        
        if subject_name in self.data:
            
            if event_type in self.data[subject_name]:
                self._data[subject_name][event_type].append(epochs)
            else:
                self._data[subject_name][event_type] = [epochs]
        
        else:
            self._data[subject_name] = {
                event_type: [epochs]
            }
    
    def concatenate(self) -> dict[str, dict[str, mne.Epochs]]:
        return {
            subject_name: {
                event_type: mne.concatenate_epochs(epochs_list)
                for event_type, epochs_list in enumerate(subject_content)
            }
            for subject_name, subject_content in enumerate(self.data)
        }
    
    def map(self, fun: Callable):
        self._data = {
            subject_name: {
                event_type: map(fun, epochs_list)
                for event_type, epochs_list in enumerate(subject_content)
            }
            for subject_name, subject_content in enumerate(self.data)
        }

class_names = ['S', 'T']
collector = EpochsCollector()
biomag_home = os.path.join(os.path.dirname(parentdir), 'BIOMAG')
biomag_data = os.path.join(biomag_home, 'BIOMAG_DATA')
classification_name = 'biomag'

for epochs_file in os.listdir(biomag_data):
    epochs = mne.read_epochs(os.path.join(biomag_data, epochs_file))
    subject_name, event_type, *_ = epochs_file.split('_')
    collector.data = subject_name, event_type, mne.read_epochs(os.path.join(biomag_home, epochs_file))

all_epochs = collector.concatenate()

for subject_name, subject_content in all_epochs:
    subject_path = os.path.join(biomag_home, subject_name)
    combiner = EpochsCombiner(*(epochs for epochs in subject_content.values())).combine(0, 1)
    n_classes, classes_samples = np.unique(combiner.Y, return_counts=True)
    n_classes = len(n_classes)
    classes_samples = classes_samples.tolist()
    combiner.shuffle()
    savepath = os.path.join(subject_path, 'TFR')
    network_out_path = os.path.join(subject_path, 'LFCNN')
    yp_path = os.path.join(network_out_path, 'Predictions')
    sp_path = os.path.join(network_out_path, 'Parameters')
    perf_tables_path  = os.path.join(network_out_path, 'perf_tables')
    check_path(subject_path, savepath, network_out_path, yp_path, sp_path, perf_tables_path)
    
    import_opt = dict(
        savepath=savepath+'/',
        out_name=classification_name,
        fs=200,
        input_type='trials',
        target_type='int',
        picks={'meg': 'grad'},
        scale=True,
        crop_baseline=True,
        decimate=None,
        scale_interval=(0, 60),
        n_folds=5,
        overwrite=True,
        segment=False,
        test_set='holdout'
    )
    X, Y = combiner.X, combiner.Y
    meta = mf.produce_tfrecords((X, Y), **import_opt)
    dataset = mf.Dataset(meta, train_batch=100)
    lf_params = dict(
        n_latent=32,
        filter_length=50,
        nonlin=tf.keras.activations.elu,
        padding='SAME',
        pooling=10,
        stride=10,
        pool_type='max',
        model_path=import_opt['savepath'],
        dropout=.4,
        l2_scope=["weights"],
        l2=1e-6
    )
    model = mf.models.LFCNN(dataset, lf_params)
    model.build()
    model.train(n_epochs=25, eval_step=100, early_stopping=5)
    y_true_train, y_pred_train = model.predict(meta['train_paths'])
    y_true_test, y_pred_test = model.predict(meta['test_paths'])
    save_parameters(
        Predictions(
            y_pred_test,
            y_true_test
        ),
        os.path.join(yp_path, f'{classification_name}_pred.pkl'),
        'predictions'
    )
    train_loss_, train_acc_ = model.evaluate(meta['train_paths'])
    test_loss_, test_acc_ = model.evaluate(meta['test_paths'])
    model.compute_patterns(meta['train_paths'])
    nt = model.dataset.h_params['n_t']
    time_courses = np.squeeze(model.lat_tcs.reshape([model.specs['n_latent'], -1, nt]))
    times = (1/float(model.dataset.h_params['fs']))*np.arange(model.dataset.h_params['n_t'])
    patterns = model.patterns.copy()
    model.compute_patterns(meta['train_paths'], output='filters')
    filters = model.patterns.copy()
    franges, finputs, foutputs, fresponces = compute_temporal_parameters(model)
    induced = list()
    for tc in time_courses:
        ls_induced = list()
        for lc in tc:
            widths = np.arange(1, 71)
            ls_induced.append(np.abs(sp.signal.cwt(lc, sp.signal.ricker, widths)))
        induced.append(np.array(ls_induced).mean(axis=0))
    induced = np.array(induced)
    
    save_parameters(
        WaveForms(time_courses.mean(1), induced, times, time_courses),
        os.path.join(sp_path, f'{classification_name}_waveforms.pkl'),
        'WaveForms'
    )
    
    save_parameters(
        SpatialParameters(patterns, filters),
        os.path.join(sp_path, f'{classification_name}_spatial.pkl'),
        'spatial'
    )
    save_parameters(
        TemporalParameters(franges, finputs, foutputs, fresponces),
        os.path.join(sp_path, f'{classification_name}_temporal.pkl'),
        'temporal'
    )
    get_order = lambda order, ts: order.ravel()
    save_parameters(
        ComponentsOrder(
            get_order(*model._sorting('l2')),
            get_order(*model._sorting('compwise_loss')),
            get_order(*model._sorting('weight')),
            get_order(*model._sorting('output_corr')),
            get_order(*model._sorting('weight_corr')),
        ),
        os.path.join(sp_path, f'{classification_name}_sorting.pkl'),
        'sorting'
    )
    perf_table_path = os.path.join(
        perf_tables_path,
        f'{classification_name}.csv'
    )
    processed_df = pd.Series(
        [
            n_classes,
            *classes_samples,
            sum(classes_samples),
            np.array(meta['test_fold'][0]).shape,
            train_acc_,
            train_loss_,
            test_acc_,
            test_loss_,
            model.v_metric,
            model.v_loss,
        ],
        index=['n_classes', *class_names, 'total', 'test_set', 'train_acc', 'train_loss', 'test_acc', 'test_loss', 'val_acc', 'val_loss'],
        name=subject_name
    ).to_frame().T
    
    if os.path.exists(perf_table_path):
        pd.concat([pd.read_csv(perf_table_path, index_col=0, header=0), processed_df], axis=0)\
            .to_csv(perf_table_path)
    else:
        processed_df\
        .to_csv(perf_table_path)
