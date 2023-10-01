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
    """
    Compute decision boundaries for a binary classification or regression model.

    This function computes decision boundaries for a binary classification or regression model by evaluating the model's
    predictions on a grid of input points.

    Args:
        clf (Union[BaseEstimator, ClassifierMixin, RegressorMixin]): A scikit-learn compatible classifier or regressor
            model that can make predictions.
        linespace (Optional[np.ndarray]): An array representing the values on the x and y axes for the grid of input
            points. Defaults to a linear space between -3 and 3 with 100 points.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
            - The first array contains the points on the decision boundary where the model's predictions are 0.
            - The second array contains the points on the decision boundary where the model's predictions are 1.

    Note:
        - For binary classification models, the decision boundary is where the model's predicted class changes from 0 to 1.
        - For regression models, the decision boundary represents the values where the model's predictions transition.
    """
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
    """
    Class for visualizing the distribution of data points and decision boundaries of a binary classification model.

    This class allows you to visualize the distribution of data points in a binary classification problem and plot
    decision boundaries created by a classifier. It is designed to work with two-dimensional data.

    Args:
        clf (Optional[Union[BaseEstimator, ClassifierMixin, RegressorMixin, AbstractTransformer]]): The classifier or
            transformer used for creating decision boundaries. Defaults to Logistic Regression.
        scale (Optional[np.ndarray]): An array representing the values on the x and y axes for the grid of input points.
            Defaults to a linear space between -3 and 3 with 100 points.

    Attributes:
        X: The input data used for fitting and transformation.
        Y: The target labels used for fitting.
        boundary: The decision boundary computed by the classifier.
        scale: An array representing the values on the x and y axes for the grid of input points.
        clf: The classifier or transformer used for creating decision boundaries.

    Methods:
        fit(X, Y): Fit the classifier using the provided data and labels.
        transform(X): Visualize the data distribution and decision boundaries.
        fit_transform(X, Y): Fit the classifier and then visualize the data distribution and decision boundaries.

    Note:
        - This class is specifically designed for two-dimensional data.
        - The `transform` method plots the decision boundaries and data points.
    """
    def __init__(
            self,
            clf: Optional[Union[
                BaseEstimator, ClassifierMixin, RegressorMixin, AbstractTransformer
            ]] = LogisticRegression(),
            scale: Optional[np.ndarray] = np.linspace(-3, 3, 100)
    ):
        """
        Initialize a DistributionPlotter instance.

        Args:
            clf (Optional[Union[BaseEstimator, ClassifierMixin, RegressorMixin, AbstractTransformer]]): The classifier or
                transformer used for creating decision boundaries. Defaults to Logistic Regression.
            scale (Optional[np.ndarray]): An array representing the values on the x and y axes for the grid of input points.
                Defaults to a linear space between -3 and 3 with 100 points.
        """
        self.X = None
        self.Y = None
        self.boundary = None
        self.scale = scale
        self.clf = clf

    def fit(self, X, Y):
        """
        Fit the classifier using the provided data and labels.

        Args:
            X: Input data.
            Y: Target labels.
        """
        self.X = X
        self.Y = Y
        self.clf.fit(self.X, self.Y)

    def transform(self, X):
        """
        Visualize the data distribution and decision boundaries.

        Args:
            X: Input data to visualize.

        Returns:
            X: The input data.
        """
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
        """
        Fit the classifier and then visualize the data distribution and decision boundaries.

        Args:
            X: Input data.
            Y: Target labels.

        Returns:
            X: The input data.
        """
        self.fit(X, Y)
        return self.transform(X)
