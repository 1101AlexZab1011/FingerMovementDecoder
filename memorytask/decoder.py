import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import argparse
import collections
from utils.storage_management import check_path
from utils.console import Silence
import warnings
import mne
from combiners import EpochsCombiner
import numpy as np
import mneflow as mf
import tensorflow as tf
import sklearn
from utils.machine_learning import one_hot_decoder
from LFRNN_decoder import LFRNN
from LFCNN_decoder import compute_temporal_parameters, compute_waveforms, \
    save_parameters, get_order, \
    Predictions, TemporalParameters, SpatialParameters, WaveForms, ComponentsOrder
import pandas as pd


def remove_repeated_members(arr: list) -> list:
    counter = collections.Counter(arr)
    return list(
        filter(lambda x: counter[x] == 1, counter.keys())
    )


def remove_single_members(arr: list) -> list:
    counter = collections.Counter(arr)
    return list(
        filter(lambda x: counter[x] > 1, counter.keys())
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', '--exclude-participants', type=str, nargs='+',
                        default=[],
                        help='IDs of subjects to exclude')
    parser.add_argument('-c', '--cases', type=str, nargs='+',
                        default=['neg_hit', 'neg_miss', 'neu_hit', 'neu_miss'],
                        help='Cases to consider (must match epochs file names '
                        'for the respective classes)')
    parser.add_argument('-cmc', '--combine-cases', type=str, nargs='+',
                        default=None,
                        help='Cases to consider (must be the number of strings in which classes '
                        'to combine are written separated by a space, indices corresponds '
                        'to order of "--cases" parameter)')
    parser.add_argument('-sd', '--subjects-dir', type=str,
                        default=os.path.join(os.getcwd(), 'Source', 'MemoryTaskSubjects'),
                        help='Path to the subjects directory')
    parser.add_argument('--name', type=str,
                        default=None,
                        help='Name of a task')
    parser.add_argument('--postfix', type=str,
                        default='',
                        help='String to append to a task name')
    parser.add_argument('--prefix', type=str,
                        default='',
                        help='String to set in the start of a task name')
    parser.add_argument('--project_name', type=str,
                        default='memory_task_epochs',
                        help='Name of a project')
    parser.add_argument('-m', '--model', type=str,
                        default='LFCNN',
                        help='Model to use')
    parser.add_argument('--no-params', action='store_true', help='Do not compute parameters')

    excluded_subjects, \
        cases, \
        cases_to_combine, \
        subjects_dir, \
        classification_name,\
        classification_postfix,\
        classification_prefix, \
        project_name, \
        model_name, \
        no_params = vars(parser.parse_args()).values()

    try:
        classifier = {
            'LFCNN': mf.models.LFCNN,
            'LFRNN': LFRNN
        }[model_name]
    except KeyError:
        raise NotImplementedError(f'This model is not implemented: {model_name}')

    cases_to_combine = sorted([[case] for case in cases] if cases_to_combine is None else [
        case.split(' ') for case in cases_to_combine
    ], reverse=True)
    cases = list(filter(lambda case: any([case in cmb for cmb in cases_to_combine]), cases))

    class_names = [
        list(dict.fromkeys([
            class_member for class_name in comb for class_member in class_name.split('_')
        ]))
        for comb in cases_to_combine
    ]

    for class_member in class_names[0]:
        if all([class_member in class_name for class_name in class_names[1:]]):
            for class_name in class_names:
                class_name.remove(class_member)

    classification_name = '_vs_'.join(['_&_'.join(class_name) for class_name in class_names])
    classification_name_formatted = "_".join(
        list(filter(lambda s: s not in (None, ""), [
            classification_prefix, classification_name, classification_postfix
        ]))
    )
    del class_names, classification_name

    perf_tables_path = os.path.join(os.path.dirname(subjects_dir), 'mem_task_perf_tables')
    check_path(perf_tables_path)

    for subject_name in os.listdir(subjects_dir):

        if subject_name in excluded_subjects:
            continue

        subject_path = os.path.join(subjects_dir, subject_name)
        epochs_path = os.path.join(subject_path, 'Epochs')
        epochs = {case: None for case in cases}
        any_info = None

        for epochs_file in os.listdir(epochs_path):

            for case in cases:
                if case in epochs_file:
                    with Silence(), warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        epochs_ = mne.read_epochs(os.path.join(epochs_path, epochs_file))
                        epochs_.resample(200)

                        if any_info is None:
                            any_info = epochs_.info

                        if epochs[case] is not None:
                            raise ValueError(f'Epochs for {case} are readed twice')

                        epochs[case] = epochs_
        for case in cases:
            print(f'{case}: {epochs[case].get_data().shape}')
        # i = 0
        # cases_indices_to_combine = list()
        # cases_to_combine_list = list()

        # for combination in cases_to_combine:
        #     cases_indices_to_combine.append(list())

        #     for j, case in enumerate(combination):

        #         i += j
        #         cases_indices_to_combine[-1].append(i)
        #         cases_to_combine_list.append(epochs[case])

        #     i += 1

        # combiner = EpochsCombiner(*cases_to_combine_list).combine(*cases_indices_to_combine)

        # n_classes, classes_samples = np.unique(combiner.Y, return_counts=True)
        # n_classes = len(n_classes)
        # classes_samples = classes_samples.tolist()
        # combiner.shuffle()
        # tfr_path = os.path.join(subject_path, 'TFR')
        # check_path(tfr_path)
        # savepath = os.path.join(
        #     tfr_path,
        #     classification_name_formatted
        # )
        # import_opt = dict(
        #     savepath=savepath + '/',
        #     out_name=project_name,
        #     fs=200,
        #     input_type='trials',
        #     target_type='int',
        #     picks={'meg': 'grad'},
        #     scale=True,
        #     crop_baseline=True,
        #     decimate=None,
        #     scale_interval=(0, 60),
        #     n_folds=5,
        #     overwrite=True,
        #     segment=False,
        #     test_set='holdout'
        # )

        # X, Y = combiner.X, combiner.Y
        # meta = mf.produce_tfrecords((X, Y), **import_opt)
        # dataset = mf.Dataset(meta, train_batch=100)
        # lf_params = dict(
        #     n_latent=32,
        #     filter_length=50,
        #     nonlin=tf.keras.activations.elu,
        #     padding='SAME',
        #     pooling=10,
        #     stride=10,
        #     pool_type='max',
        #     model_path=import_opt['savepath'],
        #     dropout=.4,
        #     l2_scope=["weights"],
        #     l2=1e-6
        # )

        # model = classifier(dataset, lf_params)
        # model.build()
        # model.train(n_epochs=25, eval_step=100, early_stopping=5)

        # network_out_path = os.path.join(subject_path, model_name)
        # yp_path = os.path.join(network_out_path, 'Predictions')
        # sp_path = os.path.join(network_out_path, 'Parameters')
        # check_path(network_out_path, yp_path, sp_path)
        # y_true_train, y_pred_train = model.predict(meta['train_paths'])
        # y_true_test, y_pred_test = model.predict(meta['test_paths'])

        # print('train-set: ', subject_name, sklearn.metrics.accuracy_score(
        #     one_hot_decoder(y_true_train), one_hot_decoder(y_pred_train)
        # ))
        # print('test-set: ', subject_name, sklearn.metrics.accuracy_score(
        #     one_hot_decoder(y_true_test), one_hot_decoder(y_pred_test)
        # ))

        # train_loss_, train_acc_ = model.evaluate(meta['train_paths'])
        # test_loss_, test_acc_ = model.evaluate(meta['test_paths'])

        # if not no_params:
        #     model.compute_patterns(meta['train_paths'])
        #     nt = model.dataset.h_params['n_t']
        #     time_courses = np.squeeze(model.lat_tcs.reshape([model.specs['n_latent'], -1, nt]))
        #     times = (1 / float(model.dataset.h_params['fs'])) *\
        #         np.arange(model.dataset.h_params['n_t'])
        #     patterns = model.patterns.copy()
        #     model.compute_patterns(meta['train_paths'], output='filters')
        #     filters = model.patterns.copy()
        #     franges, finputs, foutputs, fresponces = compute_temporal_parameters(model)
        #     induced, times, time_courses = compute_waveforms(model)

        #     save_parameters(
        #         Predictions(
        #             y_pred_test,
        #             y_true_test
        #         ),
        #         os.path.join(yp_path, f'{classification_name_formatted}_pred.pkl'),
        #         'predictions'
        #     )

        #     save_parameters(
        #         WaveForms(time_courses.mean(1), induced, times, time_courses),
        #         os.path.join(sp_path, f'{classification_name_formatted}_waveforms.pkl'),
        #         'WaveForms'
        #     )

        #     save_parameters(
        #         SpatialParameters(patterns, filters),
        #         os.path.join(sp_path, f'{classification_name_formatted}_spatial.pkl'),
        #         'spatial'
        #     )

        #     save_parameters(
        #         TemporalParameters(franges, finputs, foutputs, fresponces),
        #         os.path.join(sp_path, f'{classification_name_formatted}_temporal.pkl'),
        #         'temporal'
        #     )

        #     save_parameters(
        #         ComponentsOrder(
        #             get_order(*model._sorting('l2')),
        #             get_order(*model._sorting('compwise_loss')),
        #             get_order(*model._sorting('weight')),
        #             get_order(*model._sorting('output_corr')),
        #             get_order(*model._sorting('weight_corr')),
        #         ),
        #         os.path.join(sp_path, f'{classification_name_formatted}_sorting.pkl'),
        #         'sorting'
        #     )

        # perf_table_path = os.path.join(
        #     perf_tables_path,
        #     f'{classification_name_formatted}.csv'
        # )
        # processed_df = pd.Series(
        #     [
        #         n_classes,
        #         *classes_samples,
        #         sum(classes_samples),
        #         np.array(meta['test_fold'][0]).shape[0],
        #         train_acc_,
        #         test_acc_,
        #         model.v_metric,
        #     ],
        #     index=[
        #         'n_classes',
        #         *class_names,
        #         'total',
        #         'test_set',
        #         'train_acc',
        #         'test_acc',
        #         'val_acc',
        #     ],
        #     name=subject_name
        # ).to_frame().T

        # if os.path.exists(perf_table_path):
        #     pd.concat([pd.read_csv(perf_table_path, index_col=0, header=0), processed_df], axis=0)\
        #         .to_csv(perf_table_path)
        # else:
        #     processed_df.to_csv(perf_table_path)
