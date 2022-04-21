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
    parser.add_argument('--model', type=str,
                        default='LFCNN',
                        help='Model to use')

    excluded_subjects, \
        cases, \
        cases_to_combine, \
        subjects_dir, \
        classification_name,\
        classification_postfix,\
        classification_prefix, \
        project_name, \
        model = vars(parser.parse_args()).values()

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
    cases_to_combine_fornmatted = ['_'.join(class_name) for class_name in class_names]

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

    perf_tables_path = os.path.join(os.path.dirname(subjects_dir), 'mem_task_perf_tables')
    check_path(perf_tables_path)

    for subject_name in os.listdir(subjects_dir):

        if subject_name in excluded_subjects:
            continue

        subject_path = os.path.join(subjects_dir, subject_name)
        epochs_path = os.path.join(subject_path, 'Epochs')
        epochs = {case: list() for case in cases}
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

                        epochs[case].append(epochs_)
        
        i = 0
        cases_indices_to_combine = list()
        cases_to_combine_list = list()
        
        for combination in cases_to_combine:
            cases_indices_to_combine.append(list())
            
            for j, case in enumerate(combination):
                
                i += j
                cases_indices_to_combine[-1].append(i)
                cases_to_combine_list.append(epochs[case])
                
            i += 1
        
        print(cases_indices_to_combine, cases)
