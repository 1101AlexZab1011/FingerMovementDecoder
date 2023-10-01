import mne
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from LFCNN_decoder import SpatialParameters, TemporalParameters, WaveForms, Predictions, ComponentsOrder
import matplotlib.gridspec as gridspec
from utils import info_pick_channels
from utils.storage_management import read_pkl
from utils.viz import plot_patterns
import matplotlib as mpl
from matplotlib.patches import Rectangle, ConnectionPatch
import math
import itertools
import matplotlib.cm as cm
import matplotlib.colors as mc


def validate_arrays(array1, array2):
    if array1.shape[0] > array2.shape[0]:
        min_len = min(array1.shape[0], array2.shape[0])
        array1 = array1[:min_len]
        array2 = array2[:min_len]
    if array1.shape[1] > array2.shape[1]:
        min_len = min(array1.shape[1], array2.shape[1])
        array1 = array1[:, :min_len]
        array2 = array2[:, :min_len]
    if array1.shape[2] > array2.shape[2]:
        min_len = min(array1.shape[2], array2.shape[2])
        array1 = array1[:, :, :min_len]
        array2 = array2[:, :, :min_len]
    return array1, array2


def process_subject(subjects_dir, model_name, classification_name, sessions, subject):
    subject_path = os.path.join(subjects_dir, subject)
    info_path = os.path.join(subject_path, 'Info')
    info = read_pkl(os.path.join(info_path, os.listdir(info_path)[0]))
    info = info_pick_channels(
        info,
        list(
            filter(
                lambda ch_name: (ch_name[-1] == '2' or ch_name[-1] == '3') and 'meg' in ch_name.lower(),
                info['ch_names']
            )
        )
    )
    train_session = f'train_{sessions[0]}'
    test_sessions = [f'test_{session}' for session in sessions]
    model_data_path = os.path.join(subject_path, f'{model_name}_{train_session}')
    parameters_path = os.path.join(model_data_path, 'Parameters')
    datapaths_pair = [
        f'{classification_name}_{train_session}_{test_session}'
        for test_session in test_sessions
    ]
    waves_data_pair = {
        session: read_pkl(os.path.join(parameters_path, f'{path}_waveforms.pkl'))
        for path, session in zip(datapaths_pair, test_sessions)
    }
    patterns_data_pair = {
        session: read_pkl(os.path.join(parameters_path, f'{path}_spatial.pkl'))
        for path, session in zip(datapaths_pair, test_sessions)
    }

    return info, waves_data_pair, patterns_data_pair


def lcm_list(numbers):
    # Ensure the list is not empty
    if not numbers:
        return None

    # Initialize the result as the first number in the list
    result = numbers[0]

    # Iterate through the list and calculate the LCM of each number with the result
    for num in numbers[1:]:
        result = (result * num) // math.gcd(result, num)

    return result


def generate_slices(numbers):
    lcm = lcm_list(numbers)
    slices = list()

    for number in numbers:
        slices.append(range(0, lcm + 1, lcm // number))

    slices = list(
        map(
            lambda slice_: [slice(start, end) for start, end in zip(slice_[:-1], slice_[1:])],
            slices
        )
    )

    return slices


def count_nested_lists(lst):

    nested_count = 0
    for elem in lst:
        if isinstance(elem, list):
            nested_count += 1
    return nested_count


def get_grid(template: list[tuple[int, int]] = 4, figsize: tuple[int, int] = (16, 8)) -> tuple[plt.Figure, list[plt.Axes], list[plt.Axes]]:
    if isinstance(template, int):
        template = [template]

    rows = [row for row, _ in template ]
    cols = [col for _, col in template]
    n_rows = sum(rows)
    n_cols = lcm_list(cols)

    rows_slices = list()
    start = 0
    for row in rows:
        rows_slices.append(slice(start, start + row))
        start += row

    cols_slices = generate_slices(cols)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_rows, n_cols)
    axes = list()

    for row, col in zip(rows_slices, cols_slices):
        axes.append([fig.add_subplot(gs[row, col]) for col in col])

    return fig, axes


def plot_sescomp(
    t_obs,
    times,
    cluster_ranges,
    patterns_data_pair,
    waves_data_pair,
    info,
    best_patterns,
    best_patterns_time,
    session1,
    session2,
    subject,
    spatial_prob,
    temporal_prob,
):

    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(8, 80)

    upper_row = [
        fig.add_subplot(gs[0:2, 0:8]),
        fig.add_subplot(gs[0:2, 8:78]),
        fig.add_subplot(gs[0:2, 79:]),
    ]

    middle_row = [
        fig.add_subplot(gs[3:5, 3:22]),
        fig.add_subplot(gs[3:5, 22:42]),
        fig.add_subplot(gs[3:5, 42:60]),
        fig.add_subplot(gs[3:5, 60:79]),
        fig.add_subplot(gs[3:5, -1:]),
    ]
    scalar_product_axes = [
        fig.add_subplot(gs[3:5, 3:42]),
        fig.add_subplot(gs[3:5, 42:79]),
    ]

    bottom_row = [
        fig.add_subplot(gs[6:, 0:8]),
        fig.add_subplot(gs[6:, 8:40]),
        fig.add_subplot(gs[6:, 40:48]),
        fig.add_subplot(gs[6:, 48:]),
]

    upper_row[0].text(0.25, .85, 'A', fontsize=26)
    upper_row[0].axis('off')
    bottom_row[0].text(0.25, .85, 'B', fontsize=26)
    bottom_row[0].axis('off')
    bottom_row[2].text(0.25, .85, 'C', fontsize=26)
    bottom_row[2].axis('off')
    axes = [
        upper_row[1:2],
        middle_row[:-1],
        bottom_row[1::2]
    ]
    upper_cax = upper_row[-1]
    middle_cax = middle_row[-1]

    ax2 = axes[0][0]

    if session1 == 'test_B10-B12':
        session1, session2 = session2, session1

    session_to_name = {
        'test_B1-B3': 'Learning',
        'test_B10-B12': 'Learned'
    }

    condition1 = waves_data_pair[session1]
    condition2 = waves_data_pair[session2]
    condition1, condition2 = validate_arrays(condition1, condition2)
    diff = condition1.mean(axis=0) - condition2.mean(axis=0)

    if best_patterns_time[0][-1] > best_patterns_time[1][-1]:
        best_patterns_time[0], best_patterns_time[1] = best_patterns_time[1], best_patterns_time[0]
        best_patterns[0], best_patterns[1] = best_patterns[1], best_patterns[0]

    cb = None
    for i, (best_pattern, best_pattern_time) in enumerate(zip(best_patterns, best_patterns_time)):

        if best_pattern is None:
            continue

        ratio = np.corrcoef(
            patterns_data_pair[session1].patterns[:, best_pattern],
            patterns_data_pair[session2].patterns[:, best_pattern]
        )[0, 1]
        scalar_product_axes[i].text(.425, .5, f'{ratio : .3f}', fontsize=16)
        scalar_product_axes[i].axis('off')

        vlim = (
            0,
            1.1*patterns_data_pair[session1].patterns[:, best_pattern].max(),
        )

        img = ax2.imshow(
            diff,
            aspect='auto',
            cmap='RdBu_r',
            origin='lower',
            extent=[times[0], times[-1], 0, t_obs.shape[0]],
        )

        if cb is None:
            fig.colorbar(img, cax=upper_cax)

        ax2.set_ylabel('Component')
        ax2.set_xlabel('Time (s)')
        cluster_ranges[cluster_ranges == 1] = np.nan
        ax2.imshow(cluster_ranges, aspect='auto', cmap='binary', vmin=0, origin='lower', extent=[times[0], times[-1], 0, t_obs.shape[0]], alpha=.8)
        ax3 = axes[1][i*2]
        ax4 = axes[1][i*2 + 1]
        _ = plot_patterns(
            patterns_data_pair[session1].patterns,
            info, best_pattern,
            axes=ax3,
            show=False,
            name_format=session_to_name[session1],
            vlim=vlim, scalings=1.
        )
        _ = plot_patterns(
            patterns_data_pair[session2].patterns,
            info,
            best_pattern,
            axes=ax4,
            show=False,
            name_format=session_to_name[session2],
            vlim=vlim, scalings=1.,
            colorbar=False
        )
        if i*2 + 1 == 3:
            fig.colorbar(
                cm.ScalarMappable(norm=mc.Normalize(-vlim[-1], vlim[-1]), cmap='RdBu_r'),
                cax=middle_cax
            )

        x_low, x_high = ax2.get_xlim()
        middle_time = (best_pattern_time[0] + best_pattern_time[-1])/2
        time_perc = (middle_time - x_low)/(x_high - x_low)

        con = ConnectionPatch(
            [time_perc, (best_pattern - .75)/patterns_data_pair[session1].patterns.shape[-1] ],
            [.5, 1.15],
            axesA=ax2,
            axesB=ax3,
            coordsA='axes fraction',
            coordsB='axes fraction',
            color='black'
        )
        fig.add_artist(con)
        con = ConnectionPatch(
            [time_perc, (best_pattern - .75)/patterns_data_pair[session1].patterns.shape[-1]],
            [.5, 1.15],
            axesA=ax2,
            axesB=ax4,
            coordsA='axes fraction',
            coordsB='axes fraction',
            color='black'
        )
        fig.add_artist(con)
        rect = Rectangle(
            xy=(best_pattern_time[0] - .01*(times[-1] - times[0]) , best_pattern - .5),
            width = best_pattern_time[-1] - best_pattern_time[0] + 2*.01*(times[-1] - times[0]),
            height=2,#/patterns_data_pair[session1].patterns.shape[-1],
            edgecolor='black',
            facecolor='none'
        )
        ax2.add_patch(rect)
    ax5, ax6 = axes[-1][0], axes[-1][1]
    plot_histogram(spatial_prob, bins=5, axis=ax5)
    ax5.set_xlabel('Scalar product value')
    ax5.set_ylabel('Counts #')
    ax6.plot(times, temporal_prob[0], color='#333', label='Learning')
    ax6.plot(times, temporal_prob[1], color='#999', label='Learned')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Probability')
    ax6.legend(loc='upper right')

    for ax in list(itertools.chain(*axes)):
        ax.spines[['right', 'top']].set_visible(False)

    # fig.suptitle(f'subject: {subject}')#\ncomponent: {best_pattern}\ntime: {best_pattern_time[0]:.3f} - {best_pattern_time[-1]:.3f} s', y=1.05)
    # fig.tight_layout()

    return fig



def plot_histogram(data, bins=10, axis=None, **kwargs) -> plt.Figure:
    if axis is None:
        # Create a new figure and axis
        fig, axis = plt.subplots(**kwargs)

    # Plot the histogram
    axis.hist(data, bins=bins, alpha=0.75, edgecolor='black', density=True)

    if axis is None:
        return fig, axis


def plot_boxplot(data1, data2, labels=None, axis=None, **kwargs) -> plt.Figure:
    if axis is None:
        # Create a new figure and axis
        fig, axis = plt.subplots(**kwargs)
    else:
        fig = None

    # Create a list of data for boxplot
    boxplot_data = [data1, data2]

    # Plot the boxplots
    axis.boxplot(boxplot_data, labels=labels)

    if axis is None:
        return fig, axis



if __name__ == '__main__':
    subjects_dir = './Source/Subjects'
    pics_dir = './Source/Pictures'
    model_name = 'LFCNN'
    classification_name = 'RM_vs_RI_vs_LM_vs_LI'
    # sessions = ['B1-B3', 'B10-B12']
    # sessions = ['B10-B12', 'B1-B3']
    plot_data = list()
    cosine = None
    prob = list()
    for sessions in [['B1-B3', 'B10-B12'], ['B10-B12', 'B1-B3']]:
        cluster_ranges_prob, cluster_ranges_prob_pos, cluster_ranges_prob_neg = list(), list(), list()
        all_patterns_cosine, sig_patterns_cosine, notsig_patterns_cosine = list(), list(), list()
        times = None
        for subject in os.listdir(subjects_dir):
            info, waves_data_pair, patterns_data_pair = process_subject(subjects_dir, model_name, classification_name, sessions, subject)
            session1, session2 = waves_data_pair.keys()
            waves_data_pair_np = {
                k: np.squeeze(v.tcs)
                for k, v in waves_data_pair.items()
            }

            test_data = list()
            for n_latent in range(waves_data_pair_np[session1].shape[-1]):
                data1 = waves_data_pair_np[session1][:, :, n_latent]
                data2 = waves_data_pair_np[session2][:, :, n_latent]

                if len(data1) != len(data2):
                    min_len = min(len(data1), len(data2))
                    data1 = data1[:min_len]
                    data2 = data2[:min_len]
                if len(data1.T) != len(data2.T):
                    min_len = min(len(data1.T), len(data2.T))
                    data1 = data1[:, :min_len]
                    data2 = data2[:, :min_len]

                threshold = 6.0
                # T_obs, clusters, cluster_p_values, H0
                test_data.append(mne.stats.permutation_cluster_test(
                    [data1, data2],
                    n_permutations=10000,
                    threshold=threshold,
                    tail=1,
                    n_jobs=None,
                    out_type="mask",
                ))

            cluster_ranges = list()
            t_obs = list()
            times = waves_data_pair[session1].times - 0.5# + 60/200
            highest_t_obs_val = [-1e-7, -1e-7]
            highest_t_obs_val_neg, highest_t_obs_val_pos = -1e-7, -1e-7
            best_patterns = [None, None]
            best_pattern_times = [None, None]
            best_pos_pattern, best_neg_pattern = None, None
            best_pos_pattern_time, best_neg_pattern_time = None, None

            condition1 = waves_data_pair_np[session1]
            condition2 = waves_data_pair_np[session2]
            condition1, condition2 = validate_arrays(condition1, condition2)
            diff = condition1.mean(axis=0) - condition2.mean(axis=0) # time x n_latent

            sig_component = False
            for n_latent in range(waves_data_pair_np[session1].shape[-1]):
                T_obs, clusters, cluster_p_values, H0 = test_data[n_latent]
                t_obs.append(T_obs)
                cluster_range = np.zeros_like(times)
                patterns_cosine = np.corrcoef(
                    patterns_data_pair[session1].patterns[:, n_latent],
                    patterns_data_pair[session2].patterns[:, n_latent],
                )[0, 1]
                all_patterns_cosine.append(
                    patterns_cosine
                )

                for i_c, c in enumerate(clusters):
                    c = c[0]
                    if cluster_p_values[i_c] <= 0.05:
                        sig_component = True
                        cluster_range[c.start : c.stop - 1] = 1
                        t_obs_val = T_obs[c.start : c.stop - 1].sum()

                        if t_obs_val > highest_t_obs_val[0]:
                            highest_t_obs_val[0] = t_obs_val
                            best_patterns[0], best_patterns[1] = n_latent, best_patterns[0]
                            best_pattern_times[0], best_pattern_times[1] = times[c.start : c.stop - 1], best_pattern_times[0]
                        elif t_obs_val > highest_t_obs_val[1]:
                            highest_t_obs_val[1] = t_obs_val
                            best_patterns[0], best_patterns[1] = n_latent, best_patterns[0]
                            best_pattern_times[0], best_pattern_times[1] = times[c.start : c.stop - 1], best_pattern_times[0]

                        if diff[c.start : c.stop - 1, n_latent].mean() < 0:
                            if t_obs_val > highest_t_obs_val_neg:
                                highest_t_obs_val_neg = t_obs_val
                                best_neg_pattern = n_latent
                                best_neg_pattern_time = times[c.start : c.stop - 1]
                        else:
                            if t_obs_val > highest_t_obs_val_pos:
                                highest_t_obs_val_pos = t_obs_val
                                best_pos_pattern = n_latent
                                best_pos_pattern_time = times[c.start : c.stop - 1]

                if sig_component:
                    sig_component = False
                    sig_patterns_cosine.append(patterns_cosine)
                else:
                    notsig_patterns_cosine.append(patterns_cosine)

                cluster_ranges.append(cluster_range)

            cluster_ranges = np.array(cluster_ranges)
            cluster_ranges_prob.append(cluster_ranges.sum(axis=0))

            dirs = np.sign((diff.T * cluster_ranges))
            pos_dirs_map = dirs.copy()
            pos_dirs_map[dirs < 0] = 0
            neg_dirs_map = dirs.copy()
            neg_dirs_map[dirs > 0] = 0
            neg_dirs_map = - neg_dirs_map

            cluster_ranges_prob.append(cluster_ranges.sum(axis=0))
            cluster_ranges_prob_pos.append(pos_dirs_map.sum(axis=0))
            cluster_ranges_prob_neg.append(neg_dirs_map.sum(axis=0))

            t_obs = np.array(t_obs)
            plot_data.append(
                (t_obs, times, cluster_ranges, patterns_data_pair, waves_data_pair_np, info, best_patterns, best_pattern_times, session1, session2, subject)
            )

        all_patterns_cosine = np.array(all_patterns_cosine)
        sig_patterns_cosine = np.array(sig_patterns_cosine)
        if cosine is None:
            cosine = sig_patterns_cosine
        notsig_patterns_cosine = np.array(notsig_patterns_cosine)
        cluster_ranges_prob = np.array(cluster_ranges_prob).sum(axis=0)
        cluster_ranges_prob_pos = np.array(cluster_ranges_prob_pos).sum(axis=0)
        cluster_ranges_prob_neg = np.array(cluster_ranges_prob_neg).sum(axis=0)

        cluster_ranges_prob = cluster_ranges_prob/len(cluster_ranges_prob)
        cluster_ranges_prob_pos = cluster_ranges_prob_pos/len(cluster_ranges_prob_pos)
        cluster_ranges_prob_neg = cluster_ranges_prob_neg/len(cluster_ranges_prob_neg)
        prob.append(cluster_ranges_prob)

    for data in plot_data:
        subject = data[-1]
        session1, session2 = data[-3], data[-2]
        fig = plot_sescomp(
            *data,
            cosine,
            prob
        )
        fig.savefig(
            os.path.join(
                pics_dir, 'Learning_Effect', f'{subject}_{model_name}_{classification_name}_{session1}_{session2}.png'
            ), dpi=300, bbox_inches='tight'
        )
        plt.close(fig)

