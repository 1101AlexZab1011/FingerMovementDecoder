from __future__ import annotations
import os
import argparse
import re
import warnings
from time import perf_counter

from ndp.signal import Signal

from library.config_schema import MainConfig, SimpleNetConfig, FeatureExtractorConfig,\
    ParamsDict, flatten_dict, get_selected_params
from library.func_utils import infinite, limited
from library.interpreter import ModelInterpreter
from library.runner import LossFunction, TestIter, TrainIter, eval_model, train_model
from library.visualize import get_model_weights_figure
from library.metrics import Metrics
from library.type_aliases import ChanBatch

import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

import numpy as np
import numpy.typing as npt
import matplotlib as mpl
import pandas as pd

import mne

from typing import Optional, Union
from dataclasses import asdict, dataclass
from functools import total_ordering
import logging

from combiners import EpochsCombiner

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tqdm import trange
from omegaconf.omegaconf import OmegaConf

from utils.storage_management import check_path
from utils.console import Silence

from library.type_aliases import ChanBatchTensor, SigBatchTensor
from library.models import FeatureExtractor
from scipy.signal import firwin


class FeatureExtractorFrozenEnvelopes(FeatureExtractor):
    def __init__(self,
            cfg: FeatureExtractorConfig,
            sfreq: int,
            lfreq: int
        ):
        self.filter_weights = firwin(cfg.filtering_size, lfreq, pass_zero='lowpass', nyq=sfreq//2)
        super().__init__(cfg)


    def _create_envelope_detector(self, nch: int) -> nn.Sequential:
        kern_conv, kern_env = self.cfg.filtering_size, self.cfg.envelope_size
        pad_conv, pad_env = int(self.cfg.filtering_size / 2), int(self.cfg.envelope_size / 2)
        model = nn.Sequential(
            nn.Conv1d(nch, nch, kernel_size=kern_conv, bias=False, groups=nch, padding=pad_conv),
            nn.BatchNorm1d(nch, affine=False),
            nn.LeakyReLU(-1),  # just absolute value
            nn.Conv1d(nch, nch, kernel_size=kern_env, groups=nch, padding=pad_env),
        )

        with torch.no_grad():
            model[-1].weights = nn.Parameter(
                torch.from_numpy(np.array([
                    np.expand_dims(self.filter_weights, 0)
                    for _ in range(self.cfg.hidden_channels)
                ]))
            )
        model[-1].requires_grad_(False)

        return model


class SimpleNet(nn.Module):
    def __init__(
        self,
        cfg: SimpleNetConfig,
        feature_extractor: Optional[FeatureExtractor] = None
    ):
        super(self.__class__, self).__init__()
        self.is_use_lstm = cfg.use_lstm
        ext_cfg = cfg.feature_extractor
        self.feature_extractor = FeatureExtractor(ext_cfg) if feature_extractor is None else feature_extractor

        window_size = cfg.lag_backward + cfg.lag_forward + 1
        final_out_features = (window_size // ext_cfg.downsampling + 1) * ext_cfg.hidden_channels

        assert window_size > ext_cfg.filtering_size
        assert window_size > ext_cfg.envelope_size

        self.features_batchnorm = nn.BatchNorm1d(final_out_features, affine=False)
        if cfg.use_lstm:
            assert ext_cfg.hidden_channels % 2 == 0, "hidden channels number must be even"
            hc = ext_cfg.hidden_channels
            self.lstm = nn.LSTM(hc, hc // 2, num_layers=1, batch_first=True, bidirectional=True)

        self.fc_layer = nn.Linear(final_out_features, cfg.out_channels)

    def forward(self, x: SigBatchTensor) -> ChanBatchTensor:
        features = self.feature_extractor(x)
        if self.is_use_lstm:
            features = self.lstm(features.transpose(1, 2))[0].transpose(1, 2)
        features = features.reshape((features.shape[0], -1))
        self.features_scaled = self.features_batchnorm(features)
        output = self.fc_layer(self.features_scaled)
        return output


Loaders = dict[str, DataLoader]
log = logging.getLogger(__name__)


def prepare_data(epochs_list: list[mne.Epochs]):
    X = np.array([])
    Y = list()
    scaler = StandardScaler()
    for i, epochs in enumerate(epochs_list):
        epochs = epochs.load_data().pick_types(meg='grad')
        data = np.array([scaler.fit_transform(epoch) for epoch in epochs.get_data()])

        if i == 0:
            X = data.copy()
        else:
            X = np.append(X, data, axis=0)

        Y += [i for _ in range(data.shape[0])]

    Y = np.array(Y)

    return X, Y


def create_data_loaders(
    X: Signal[npt._32Bit],
    Y: Signal[npt._32Bit],
    batch_size: Optional[int] = 100
):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    train, test = Discrete(X_train, Y_train), Discrete(X_test, Y_test)

    train = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    test = DataLoader(test, batch_size=batch_size, shuffle=True, drop_last=True)

    return dict(
        train=train,
        test=test
    )


def simple_accuracy_score(
    y_true: Union[list[int], np.ndarray],
    y_pred: Union[list[int], np.ndarray]
) -> float:
    assert len(y_true) == len(y_pred), f"{len(y_true) = } but {len(y_pred) = }"
    correct = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            correct += 1
    return correct / len(y_true)


@total_ordering
@dataclass
class ClassificationMetrics(Metrics):
    acc: float
    loss: float

    @classmethod
    def calc(cls, y_predicted: ChanBatch, y_true: ChanBatch, loss: float) -> ClassificationMetrics:
        acc = simple_accuracy_score(one_hot_decoder(y_true), one_hot_decoder(y_predicted))
        return cls(acc, loss)

    def __lt__(self, other: ClassificationMetrics) -> bool:
        return (self.acc, -self.loss) < (other.acc, -other.loss)


def get_metrics(model: nn.Module, loss: LossFunction, ldrs: Loaders, n_iter: int) -> ParamsDict:
    metrics = {}
    for stage, ldr in ldrs.items():
        tr = trange(n_iter, desc=f"Evaluating model on {stage}")
        eval_iter = limited(TestIter(model, ldr, loss)).by(tr)
        metrics[stage] = asdict(eval_model(eval_iter, ClassificationMetrics, n_iter))

    log.debug(f"{metrics=}")
    return flatten_dict(metrics, sep="/")


def one_hot_encoder(Y: np.ndarray) -> np.ndarray:
    y = list()
    n_classes = len(np.unique(Y))

    for val in Y:
        new_y_value = np.zeros(n_classes)
        new_y_value[val] = 1
        y.append(new_y_value)

    return np.array(y)


def one_hot_decoder(y: np.array) -> np.array:
    y_decoded = list()
    for val in y:
        y_decoded.append(np.where(val == val.max())[0][0])

    return np.array(y_decoded)


@dataclass
class Discrete(Dataset):

    X: np.ndarray  # array of shape (n_samples, n_sensors)
    Y: np.ndarray  # array of shape (n_samples, n_classes)

    def __post_init__(self) -> None:
        assert self.X.dtype == np.float32,\
            f'wrong features type: {self.X.dtype} (must be np.float32)'
        assert len(self.X) == len(self.Y),\
            f'features and target have different sizes: {self.X.shape = }, {self.Y.shape = }'

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> tuple[np.ndarray, int]:
        X = self.X[i].T
        Y = self.Y[i]
        return X, Y


if __name__ == '__main__':
    mpl.use('agg')
    parser = argparse.ArgumentParser(
        description='A script for applying the neural network "LFRNN" to the '
        'epoched data from gradiometers related to events for classification'
    )
    parser.add_argument('-es', '--exclude-sessions', type=str, nargs='+',
                        default=[], help='Sessions to exclude')
    parser.add_argument('-ep', '--exclude-participants', type=str, nargs='+',
                        default=[], help='IDs of subjects to exclude')
    parser.add_argument('-l', '--lock', type=str,
                        default='RespCor', help='Stimulus lock to consider')
    parser.add_argument('-c', '--cases', type=str, nargs='+',
                        default=['LI', 'LM', 'RI', 'RM'],
                        help='Cases to consider (must match epochs '
                        'file names for the respective classes)')
    parser.add_argument('-cmb', '--combine-cases', type=str, nargs='+',
                        default=None,
                        help='Cases to consider (must be the number of strings in '
                        'which classes to combine are written separated by a space, '
                        'indices corresponds to order of "--cases" parameter)')
    parser.add_argument('-sd', '--subjects-dir', type=str,
                        default=os.path.join(os.getcwd(), 'Source', 'Subjects'),
                        help='Path to the subjects directory')
    parser.add_argument('--trials-name', type=str,
                        default='B', help='Name of trials')
    parser.add_argument('--name', type=str,
                        default=None, help='Name of a task')
    parser.add_argument('--postfix', type=str,
                        default='', help='String to append to a task name')
    parser.add_argument('--prefix', type=str,
                        default='', help='String to set in the start of a task name')
    parser.add_argument('--project_name', type=str,
                        default='fingers_movement_epochs', help='Name of a project')
    parser.add_argument('-lp', '--low_pass', type=float,
                        default=None, help='Low-pass filter (Hz) for envelope detector')

    excluded_sessions, \
        excluded_subjects, \
        lock, \
        cases, \
        cases_to_combine, \
        subjects_dir, \
        sessions_name,\
        classification_name,\
        classification_postfix,\
        classification_prefix, \
        project_name,\
        lp = vars(parser.parse_args()).values()

    if excluded_sessions:
        excluded_sessions = [
            sessions_name + session
            if sessions_name not in session
            else session
            for session in excluded_sessions
        ]

    cases_to_combine = [
        case.split(' ')
        for case in cases
    ] if cases_to_combine is None else [
        case.split(' ')
        for case in cases_to_combine
    ]
    cases = list(filter(lambda case: any([case in cmb for cmb in cases_to_combine]), cases))
    cases_to_combine = sorted(cases_to_combine, reverse=True)
    class_names = [
        '&'.join(sorted(
            cases_combination,
            reverse=True
        ))
        for cases_combination in cases_to_combine
    ]

    if classification_name is None:
        classification_name = '_vs_'.join(class_names)

    classification_name_formatted = "_".join(list(filter(
        lambda s: s not in (None, ""),
        [
            classification_prefix,
            classification_name,
            classification_postfix
        ]
    )))

    perf_tables_path = os.path.join(os.path.dirname(subjects_dir), 'perf_tables')
    dumps_path = os.path.join(os.path.dirname(subjects_dir), 'model_dumps')
    tb_path = os.path.join(os.path.dirname(subjects_dir), 'TB')
    check_path(perf_tables_path, dumps_path, tb_path)
    subjects_performance = list()

    fh = logging.FileHandler('./Source/simplenet_history.log')
    fh.setLevel(logging.DEBUG)
    log.addHandler(fh)

    featuresextcfg = FeatureExtractorConfig(
        in_channels=204,
        downsampling=10,
        # downsampling=102,  # half of input
        hidden_channels=4,
        # hidden_channels=32,  # as in LFCNN
        filtering_size=51,  # band-pass filter -> as in the 2nd layer of LFCNN
        envelope_size=51  # low-pass filter, same as band-pass
    )

    modelcfg = SimpleNetConfig(
        out_channels=None,
        lag_backward=99,  # -500ms with 200 Hz -> 0.5*200 = 100 samples
        lag_forward=99,  # 500ms with 200 Hz -> .5*200 = 100 samples
        use_lstm=False,
        feature_extractor=featuresextcfg
    )

    maincfg = MainConfig(
        debug=False,
        lag_backward=99,
        lag_forward=99,
        target_features_cnt=1,
        selected_channels=[i for i in range(306) if (i + 1) % 3],
        model=modelcfg,
        dataset=None,
        batch_size=100,
        n_steps=20000,
        metric_iter=250,
        model_upd_freq=250,
        train_test_ratio=.7,
        learning_rate=0.0003,
        subject='test-subject',
        plot_loaded=False
    )

    for subject_name in os.listdir(subjects_dir):

        maincfg.subject = subject_name

        if subject_name in excluded_subjects:
            continue

        train_acc, train_loss, val_acc, val_loss, test_acc, test_loss = (list() for _ in range(6))
        subject_path = os.path.join(subjects_dir, subject_name)
        epochs_path = os.path.join(subject_path, 'Epochs')
        epochs = {case: list() for case in cases}
        any_info = None

        for epochs_file in os.listdir(epochs_path):
            if lock not in epochs_file:
                continue

            session = re.findall(r'_{0}\d\d?'.format(sessions_name), epochs_file)[0][1:]

            if session in excluded_sessions:
                continue

            for case in cases:
                if case in epochs_file:
                    with Silence(), warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        epochs_ = mne.read_epochs(os.path.join(epochs_path, epochs_file))
                        epochs_.resample(200)

                        if any_info is None:
                            any_info = epochs_.info

                        epochs[case].append(epochs_)

        epochs = dict(
            zip(
                epochs.keys(),
                map(
                    mne.concatenate_epochs,
                    list(epochs.values())
                )
            )
        )

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

        combiner = EpochsCombiner(*cases_to_combine_list).combine(*cases_indices_to_combine)
        n_classes, classes_samples = np.unique(combiner.Y, return_counts=True)
        n_classes = len(n_classes)
        modelcfg.out_channels = n_classes
        classes_samples = classes_samples.tolist()
        combiner.shuffle()
        X, Y = combiner.X, combiner.Y
        X = np.array([
            StandardScaler().fit_transform(epoch)
            for epoch in X
        ])
        X = np.transpose(X, (0, 2, 1))

        ldrs = create_data_loaders(X.astype(np.float32), one_hot_encoder(Y).astype(np.float32), 100)
        ildrs = dict(
            train=infinite(ldrs['train']),
            test=infinite(ldrs['test'])
        )
        if lp is not None:
            fe = FeatureExtractorFrozenEnvelopes(
                modelcfg.feature_extractor, any_info['sfreq'], lp
            )
        else:
            fe = FeatureExtractor(
                modelcfg.feature_extractor
            )
        model = SimpleNet(
            modelcfg,
            fe
        )
        # loss = nn.MSELoss()
        loss = nn.BCEWithLogitsLoss()
        # loss = nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr=maincfg.learning_rate)
        optimizer = torch.optim.AdamW(model.parameters(), lr=maincfg.learning_rate)
        train_iter = map(
            lambda x: ClassificationMetrics.calc(*x),
            infinite(TrainIter(model, ldrs["train"], loss, optimizer))
        )
        test_iter = map(
            lambda x: ClassificationMetrics.calc(*x),
            infinite(TestIter(model, ldrs["test"], loss))
        )

        tr = trange(maincfg.n_steps, desc="Experiment main loop")
        with SummaryWriter(
            os.path.join(tb_path, subject_name)
        ) as sw:
            t1 = perf_counter()
            train_model(train_iter, test_iter, tr, model, maincfg.model_upd_freq, sw)
            runtime = perf_counter() - t1
            metrics = get_metrics(model, loss, ildrs, maincfg.metric_iter)
            log.info("Final metrics: " + ", ".join(f"{k}={v}" for k, v in metrics.items()))
            print("Final metrics: " + ", ".join(f"{k}={v}" for k, v in metrics.items()))
            options = {"debug": [True, False]}
            hparams = get_selected_params(OmegaConf.create(maincfg.__dict__))
            sw.add_hparams(hparams, metrics, hparam_domain_discrete=options, run_name="hparams")
            n_branches = maincfg.model.feature_extractor.hidden_channels
            signal = Signal(np.reshape(X, (X.shape[0] * X.shape[1], X.shape[2])), 600, [])
            mi = ModelInterpreter(model.feature_extractor, signal)
            fig = get_model_weights_figure(mi, any_info, n_branches)
            sw.add_figure(tag=f"nsteps = {maincfg.n_steps}", figure=fig)

        train_acc, train_loss = metrics['train/acc'], metrics['train/loss']
        test_acc, test_loss = metrics['test/acc'], metrics['test/loss']

        perf_table_path = os.path.join(
            perf_tables_path,
            f'simplenet_{classification_name_formatted}.csv'
        )
        processed_df = pd.Series(
            [
                n_classes,
                *classes_samples,
                sum(classes_samples),
                int(len(X) * (1 - maincfg.train_test_ratio)),
                train_acc,
                train_loss,
                test_acc,
                test_loss,
                runtime
            ],
            index=[
                'n_classes',
                *class_names,
                'total',
                'test_set',
                'train_acc',
                'train_loss',
                'test_acc',
                'test_loss',
                'runtime'
            ],
            name=subject_name
        ).to_frame().T

        if os.path.exists(perf_table_path):
            pd.concat([pd.read_csv(perf_table_path, index_col=0, header=0), processed_df], axis=0)\
                .to_csv(perf_table_path)
        else:
            processed_df.to_csv(perf_table_path)
