from typing import Optional, Union

import numpy as np
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegression

from utils.machine_learning import AbstractTransformer


def binary_dicision_boundary(
    clf: Union[
        BaseEstimator,
        ClassifierMixin,
        RegressorMixin
    ],
    linespace: Optional[np.ndarray] = np.linspace(-3, 3, 100)
):
    i_mesh, j_mesh = np.meshgrid(linespace, linespace)
    true_mesh, false_mesh = list(), list()
    for i in range(linespace.shape[-1]):
        x_mesh = np.array([i_mesh[i, :], j_mesh[i, :]]).T
        prediction_mesh = clf.predict(x_mesh)
        class1 = x_mesh[prediction_mesh == 0]
        class2 = x_mesh[prediction_mesh == 1]
        true_mesh.append(class1)
        false_mesh.append(class2)
    return np.array(true_mesh), np.array(false_mesh)


class DistributionPlotter(AbstractTransformer):
    def __init__(
            self,
            clf: Optional[Union[
                BaseEstimator, ClassifierMixin, RegressorMixin, AbstractTransformer
            ]] = LogisticRegression(),
            scale: Optional[np.ndarray] = np.linspace(-3, 3, 100)
    ):
        self.X = None
        self.Y = None
        self.boundary = None
        self.scale = scale
        self.clf = clf

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.clf.fit(self.X, self.Y)

    def transform(self, X):
        if self.boundary is None:
            self.boundary = binary_dicision_boundary(self.clf)

        true_mesh, false_mesh = self.boundary

        for true, false in zip(true_mesh, false_mesh):
            plt.plot(
                true[:, 0],
                true[:, 1],
                'om',
                false[:, 0],
                false[:, 1],
                'oc'
            )
        class1 = self.X[self.Y == 0]
        class2 = self.X[self.Y == 1]
        plt.plot(
            class1[:, 0], class1[:, 1], 'or',
            class2[:, 0], class2[:, 1], 'ob'
        )
        plt.plot(
            X[:, 0], X[:, 1], 'xw'
        )
        plt.show()

        return X

    def fit_transform(self, X, Y):
        self.fit(X, Y)
        return self.transform(X)
