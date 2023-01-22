import matplotlib as mpl
import argparse
import os
from lfcnn_vis import plot_spatial_weights, read_pkl
from LFCNN_decoder import SpatialParameters, TemporalParameters, WaveForms
from typing import  Optional
import mne

def info_pick_channels(info: mne.Info, ch_names: list[str], ordered: Optional[bool] = False) -> mne.Info:
    sel = mne.pick_channels(info.ch_names, ch_names)
    return mne.pick_info(info, sel, copy=False, verbose=False)


if __name__ == '__main__':
    mpl.use('agg')
    parser = argparse.ArgumentParser(
        description='A script for applying the neural network "LFCNN" to the epoched '
        'data from gradiometers related to events for classification'
    )
    parser.add_argument('-s', '--sort', type=str,
                        default='sum', help='Method to sort components')
    parser.add_argument('--subject', type=str,
                        default=None, help='Subject ID')
    parser.add_argument('-cmb', '--combine-cases', type=str, nargs='+',
                        default=['LI', 'LM', 'RI', 'RM'],
                        help='Cases to consider (must be the number of strings in '
                        'which classes to combine are written separated by a space, '
                        'indices corresponds to order of "--cases" parameter)')
    parser.add_argument('-sd', '--subjects-dir', type=str,
                        default=os.path.join(os.getcwd(), 'Source', 'Subjects'),
                        help='Path to the subjects directory')
    parser.add_argument('--name', type=str,
                        default=None, help='Name of a task')
    parser.add_argument('--postfix', type=str,
                        default='', help='String to append to a task name')
    parser.add_argument('--prefix', type=str,
                        default='', help='String to set in the start of a task name')
    parser.add_argument('-m', '--model', type=str,
                        default='LFCNN', help='Name of the model that was used')
    parser.add_argument('--log', action='store_true', help='Apply logaritmic scale')

    sort, \
        subject_name, \
        cases_to_combine, \
        subjects_dir, \
        classification_name,\
        classification_postfix,\
        classification_prefix, \
        model_name, \
        logscale = vars(parser.parse_args()).values()

    cases_to_combine = [case.split(' ') for case in cases_to_combine]

    cases_to_combine = sorted(cases_to_combine, reverse=True)
    class_names = ['&'.join(sorted(
        cases_combination, reverse=True
    )) for cases_combination in cases_to_combine]

    if classification_name is None:
        classification_name = '_vs_'.join(class_names)

    classification_name_formatted = "_".join(list(filter(
        lambda s: s not in (None, ""),
        [classification_prefix, classification_name, classification_postfix]
    )))

    subject_info_path = os.path.join(subjects_dir, subject_name, 'Info')
    subject_parameters_path = os.path.join(subjects_dir, subject_name, model_name, classification_name_formatted, 'Parameters')

    spatial_parameters = read_pkl(os.path.join(
        subject_parameters_path,
        f'{classification_name_formatted}_spatial.pkl'
    ))
    temporal_parameters = read_pkl(os.path.join(
        subject_parameters_path,
        f'{classification_name_formatted}_temporal.pkl'
    ))
    waveforms = read_pkl(os.path.join(
        subject_parameters_path,
        f'{classification_name_formatted}_waveforms.pkl'
    ))
    compressions = read_pkl(os.path.join(
        subject_parameters_path,
        f'{classification_name_formatted}_compression.pkl'
    ))
    info = read_pkl(os.path.join(
        subject_info_path,
        os.listdir(subject_info_path)[0]
    ))

    info_pick_channels(
        info,
        list(
            filter(
                lambda ch_name: (
                    ch_name[-1] == '2'
                    or ch_name[-1] == '3'
                ) and 'meg' in ch_name.lower(),
                info['ch_names']
            )
        )
    )
    # print(f'weight {orders.weight + 1}')
    # print(f'weight_corr {orders.weight_corr + 1 }')
    # print(f'output_corr {orders.output_corr + 1 }')
    # print(f'compwise_loss {orders.compwise_loss + 1 }')
    plot_spatial_weights(
        spatial_parameters,
        temporal_parameters,
        waveforms,
        info,
        summarize=sort,
        logscale=logscale
    )
