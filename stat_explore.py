import os
import pickle
from mne import create_info
from mne.time_frequency import AverageTFR
from cross_runs_TF_planes import CrossRunsTFScorer
from matplotlib import pyplot as plt
from typing import Union, Generator, Optional, NoReturn
import numpy as np

from utils.console import Silence
from utils.storage_management import check_path


def convolutional_generator(
        matrix: np.ndarray,
        filter_size: Union[int, tuple[int, int]],
        stride: Optional[Union[int, tuple[int, int]]] = (1, 1),
        padding: Optional[Union[int, tuple[int, int]]] = (0, 0),
        x_labels: Optional[list[Union[int, str]]] = None,
        y_labels: Optional[list[Union[int, str]]] = None
) -> Generator[tuple[Union[int, str], Union[int, str], np.ndarray], None, NoReturn]:
    if x_labels is not None and matrix.shape[1] != len(x_labels):
        raise ValueError(f'Shape and labels does not match: shape[1]: {matrix.shape[1]}, x_labels: {len(x_labels)}')
    if y_labels is not None and matrix.shape[0] != len(y_labels):
        raise ValueError(f'Shape and labels does not match: shape[0]: {matrix.shape[0]}, x_labels: {len(y_labels)}')
    if isinstance(filter_size, int):
        filter_size = (filter_size, filter_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if matrix.ndim == 2:
        matrix = np.pad(matrix, ((padding[0], padding[0]), (padding[1], padding[1])))
    elif matrix.ndim == 3:
        slices = [matrix[:, :, i] for i in range(matrix.shape[-1])]
        for i in range(len(slices)):
            slices[i] = np.pad(slices[i], ((padding[0], padding[0]), (padding[1], padding[1]))).T
        matrix = np.array(slices).T
    else:
        raise ValueError(f'Wrong number of dimensions: {matrix.ndim}. Only ndim=2 and ndim=3 are supported')
    i = 0
    j = 0
    for _0 in range(matrix.shape[0]):
        for _1 in range(matrix.shape[1]):
            if i <= matrix.shape[0] and j <= matrix.shape[1]:
                if i < matrix.shape[0]:
                    if i + filter_size[0] <= matrix.shape[0]:
                        i_edge = i + filter_size[0]
                    else:
                        i_edge = i + matrix.shape[0] - filter_size[0]
                else:
                    i_edge = matrix.shape[0] - 1
                if j < matrix.shape[1]:
                    if j + filter_size[1] <= matrix.shape[1]:
                        j_edge = filter_size[1]
                    else:
                        j_edge = matrix.shape[1] - j
                else:
                    j_edge = matrix.shape[1] - 1
                if matrix.ndim == 2:
                    out = matrix[i:i_edge, j:j + j_edge]
                else:
                    out = matrix[i:i_edge, j:j + j_edge, :]
                y_edges = (i, i_edge - 1) if x_labels is None else (y_labels[i], y_labels[i_edge - 1])
                x_edges = (j, j + j_edge - 1) if x_labels is None else (x_labels[j], x_labels[j + j_edge - 1])
                yield x_edges, y_edges, out
                j += stride[1]
            else:
                break
        i += stride[0]
        j = 0


def ssd_pool(
        matrix: np.ndarray,
        filter_size: Union[int, tuple[int, int]],
        stride: Optional[Union[int, tuple[int, int]]] = (1, 1),
        padding: Optional[Union[int, tuple[int, int]]] = (0, 0),
        x_labels: Optional[list[Union[int, str]]] = None,
        y_labels: Optional[list[Union[int, str]]] = None
) -> tuple[list[Union[int, str]], list[Union[int, str]], np.ndarray]:
    y_prev = None
    A = list()
    row = list()
    x_labels_out = list()
    y_labels_out = list()
    initial = True
    for i, shape in enumerate(convolutional_generator(matrix, filter_size, stride, padding, x_labels, y_labels)):
        x, y, shape = shape
        x_labels_out.append(x)
        y_labels_out.append(y)
        if initial:
            y_prev = y
            initial = False
        ssds = list()
        for j in range(shape.shape[-1]):
            for k in range(shape.shape[-1]):
                # ssds.append(np.sum((shape[:, :, j] - shape[:, :, k]))**2)
                ssds.append(np.linalg.norm((shape[:, :, j] - shape[:, :, k])) ** 2)
        res = sum(ssds) / len(ssds)
        if y == y_prev:
            row.append(res)
        else:
            A.append(np.array(row))
            row = [res]
        y_prev = y
    A.append(np.array(row))
    return x_labels_out, y_labels_out, np.array(A)


def zero_one_sigmoid(x: Union[float, np.ndarray], threshold: float, infinitesimal=1e-5) -> np.array:
    k = threshold / np.log(infinitesimal ** -1)
    return 1 / (1 + np.exp((x - threshold) / k))

import matplotlib

tmin, tmax = -.500, .500
n_cycles = 14
min_freq = 5.
max_freq = 70.
n_freqs = 7
freqs_range = np.logspace(np.log10(min_freq), np.log10(max_freq), n_freqs)
window_spacing = (n_cycles / np.max(freqs_range) / 2.)
centered_w_times = np.arange(tmin, tmax, window_spacing)[1:]

subjects_dir = os.path.join('./', 'Source', 'Subjects')
pics_dir = os.path.join('./', 'Source/Pictures')
check_path(pics_dir)
tf_planes_pics_dir = os.path.join(pics_dir, 'Stat_Explore')
check_path(tf_planes_pics_dir)

tf_scorers = dict()
for subject in os.listdir(subjects_dir):
    if subject is 'Ga_Fed_06':
        continue
    tf_planes_path = os.path.join(subjects_dir, subject, 'TF_planes')
    for session in os.listdir(tf_planes_path):
        cor_path = os.path.join(tf_planes_path, session, 'RespCor')
        if session not in tf_scorers:
            tf_scorers.update({session: dict()})
        for case in os.listdir(cor_path):
            case_name = case[:-4]
            if case_name in tf_scorers[session]:
                tf_scorers[session][case_name].append(
                    pickle.load(open(
                        os.path.join(cor_path, case),
                        'rb'
                    ))
                )
            else:
                tf_scorers[session].update({
                    case_name: [
                        pickle.load(open(
                            os.path.join(cor_path, case),
                            'rb'
                        ))
                    ]
                })

tf_planes = {
    session: {
        case: np.array([tfs.tf_scores.mean(axis=0).T for tfs in tf_scorers[session][case]]).T
        for case in tf_scorers[session]
    }
    for session in tf_scorers
}

for session, cases in tf_planes.items():
    for case, planes in cases.items():
        m, n = (1, 1), (1, 1)
        x, y, A = ssd_pool(planes, m, n)
        # print(planes.shape)
        print(f'{session} {case}: {ssd_pool(planes, 6, 11)[2][0][0]}')
        freqs = [0.0] + list(freqs_range[1:])
        new_freqs = sorted([np.round(
            (freqs[i%len(freqs)] + freqs[j%len(freqs)])/2, 2
        ) for i, j in set(y)])
        new_times = sorted([np.round(
            (centered_w_times[i%len(centered_w_times)] + centered_w_times[j%len(centered_w_times)])/2, 2
        ) for i, j in set(x)])
        thr = A.mean().mean()*0.5
        with Silence():
            A_corr = zero_one_sigmoid(A, thr)
            av_tfr = AverageTFR(create_info(['freq'], 1000), planes.mean(axis=2)[np.newaxis, :],
                        centered_w_times, freqs_range[1:], 1)
            av_tfr.plot([0], vmin=.5, title=f'Decoding Scores: Average, {session}, {case}',
                cmap=plt.cm.Reds, show=False)
            plt.savefig(
                os.path.join(
                    tf_planes_pics_dir, f'dec_scores_ave_{session}_{case}.png'
                ),
            )
            av_tfr = AverageTFR(create_info(['freq'], 1000), planes.std(axis=2)[np.newaxis, :],
                        centered_w_times, freqs_range[1:], 1)
            av_tfr.plot([0], vmin=0e0, vmax=.1, title=f'Decoding Scores: Standard Deviation, {session}, {case}',
                cmap=matplotlib.cm.get_cmap('Blues_r'), show=False)
            plt.savefig(
                os.path.join(
                    tf_planes_pics_dir, f'dec_scores_std_{session}_{case}.png'
                ),
            )
            av_tfr = AverageTFR(create_info(['freq'], 1000), A_corr[np.newaxis, :],
                        new_times, new_freqs, 1)
            av_tfr.plot([0], vmin=thr, title=f'Decoding Scores: Pairwise Similarity, 1x1, {session}, {case}',
                cmap=plt.cm.Greens, show=False)
            plt.savefig(
                os.path.join(
                    tf_planes_pics_dir, f'dec_scores_sim_{session}_{case}.png'
                ),
            )
