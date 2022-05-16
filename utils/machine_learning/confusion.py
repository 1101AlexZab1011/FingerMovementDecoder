from typing import Union, Any

import numpy as np


class ConfusionEstimator(object):
    """Class for computation a classification performance

    Args:
        tp (:obj:`float` or `np.ndarray`): True positive predictions
        tn (:obj:`float` or `np.ndarray`): True negative predictions
        fp (:obj:`float` or `np.ndarray`): False positive predictions
        fn (:obj:`float` or `np.ndarray`): False negative predictions

    Raises:
        ValueError: If given drguments are not numbers
    """
    def __init__(
            self,
            tp: Union[float, np.ndarray],
            tn: Union[float, np.ndarray],
            fp: Union[float, np.ndarray],
            fn: Union[float, np.ndarray]
    ):
        tp, tn, fp, fn = (
            val.mean()
            if isinstance(val, np.ndarray)
            else val
            for val in (tp, tn, fp, fn)
        )

        if any([
            np.isnan(val)
            or val is None
            or np.isinf(val)
            or not np.isreal(val)
            for val in (tp, tn, fp, fn)
        ]):
            raise ValueError(
                'All values of confusion matrix must be '
                'real numbers or arrays of real numbers'
            )

        self._tp, self._tn, self._fp, self._fn = tp, tn, fp, fn
        self.__init_params()

    @property
    def tp(self):
        return self._tp

    @tp.setter
    def tp(self, value: float):
        self._tp = value
        self.__init_params()

    @property
    def tn(self):
        return self._tn

    @tn.setter
    def tn(self, value: float):
        self._tn = value
        self.__init_params()

    @property
    def fp(self):
        return self._fp

    @fp.setter
    def fp(self, value: float):
        self._fp = value
        self.__init_params()

    @property
    def fn(self):
        return self._fn

    @fn.setter
    def fn(self, value: float):
        self._fn = value
        self.__init_params()

    @property
    def acc(self):
        return self._acc

    @acc.setter
    def acc(self, value: Any):
        raise AttributeError(
            'Accuracy can not be set, '
            'it must be computed from confusion matrix'
        )

    @property
    def tpr(self):
        return self._tpr

    @tpr.setter
    def tpr(self, value: Any):
        raise AttributeError(
            'True Positive Rate can not be set, '
            'it must be computed from confusion matrix'
        )

    @property
    def sens(self):
        return self._tpr

    @sens.setter
    def sens(self, value):
        raise AttributeError(
            'Sensitivity can not be set, '
            'it must be computed from confusion matrix'
        )

    @property
    def tnr(self):
        return self._tnr

    @tnr.setter
    def tnr(self, value: Any):
        raise AttributeError(
            'True Negative Rate can not be set, '
            'it must be computed from confusion matrix'
        )

    @property
    def spec(self):
        return self._tnr

    @spec.setter
    def spec(self, value):
        raise AttributeError(
            'Specificity can not be set, '
            'it must be computed from confusion matrix'
        )

    @property
    def fpr(self):
        return self._fpr

    @fpr.setter
    def fpr(self, value: Any):
        raise AttributeError(
            'False Positive Rate can not be set, '
            'it must be computed from confusion matrix'
        )

    @property
    def mr(self):
        return self._fpr

    @mr.setter
    def mr(self, value):
        raise AttributeError(
            'Miss Rate can not be set, '
            'it must be computed from confusion matrix'
        )

    @property
    def fnr(self):
        return self._fnr

    @fnr.setter
    def fnr(self, value):
        raise AttributeError(
            'False Negative Rate can not be set, '
            'it must be computed from confusion matrix'
        )

    @property
    def fallout(self):
        return self._fnr

    @fallout.setter
    def fallout(self, value):
        raise AttributeError(
            'Fall-Out can not be set, '
            'it must be computed from confusion matrix'
        )

    @property
    def ppv(self):
        return self._ppv

    @ppv.setter
    def ppv(self, value):
        raise AttributeError(
            'Positive Predictive Value can not be set, '
            'it must be computed from confusion matrix'
        )

    @property
    def prec(self):
        return self._ppv

    @prec.setter
    def prec(self, value):
        raise AttributeError(
            'Precision can not be set, '
            'it must be computed from confusion matrix'
        )

    @property
    def npv(self):
        return self._npv

    @npv.setter
    def npv(self, value):
        raise AttributeError(
            'Negative Predictive Value can not be set, '
            'it must be computed from confusion matrix'
        )

    @property
    def fdr(self):
        return self._fdr

    @fdr.setter
    def fdr(self, value):
        raise AttributeError(
            'False Discovery Rate can not be set, '
            'it must be computed from confusion matrix'
        )

    @property
    def for_(self):
        return self._for

    @for_.setter
    def for_(self, value):
        raise AttributeError(
            'False Omission Rate can not be set, '
            'it must be computed from confusion matrix'
        )

    @property
    def pt(self):
        return self._pt

    @pt.setter
    def pt(self, value):
        raise AttributeError(
            'Prevalence Threshold can not be set, '
            'it must be computed from confusion matrix'
        )

    @property
    def ts(self):
        return self._ts

    @ts.setter
    def ts(self, value):
        raise AttributeError(
            'Threat Score can not be set, '
            'it must be computed from confusion matrix'
        )

    @property
    def csi(self):
        return self._ts

    @csi.setter
    def csi(self, value):
        raise AttributeError(
            'Critical Success Index can not be set, '
            'it must be computed from confusion matrix'
        )

    @property
    def ba(self):
        return self._ba

    @ba.setter
    def ba(self, value):
        raise AttributeError(
            'Balanced Accuracy can not be set, '
            'it must be computed from confusion matrix'
        )

    @property
    def f1(self):
        return self._f1

    @f1.setter
    def f1(self, value):
        raise AttributeError(
            'F1-score can not be set, '
            'it must be computed from confusion matrix'
        )

    @property
    def mcc(self):
        return self._mcc

    @mcc.setter
    def mcc(self, value):
        raise AttributeError(
            'Matthews Correlation Coefficient can not be set, '
            'it must be computed from confusion matrix'
        )

    @property
    def fm(self):
        return self._fm

    @fm.setter
    def fm(self, value):
        raise AttributeError(
            'Fowlkes-Mallows Index can not be set, '
            'it must be computed from confusion matrix'
        )

    @property
    def bm(self):
        return self._bm

    @bm.setter
    def bm(self, value):
        raise AttributeError(
            'Informedness can not be set, '
            'it must be computed from confusion matrix'
        )

    @property
    def mk(self):
        return self._mk

    @mk.setter
    def mk(self, value):
        raise AttributeError(
            'Markedness can not be set, '
            'it must be computed from confusion matrix'
        )

    @property
    def lr_plus(self):
        return self._lr_plus

    @lr_plus.setter
    def lr_plus(self, value):
        raise AttributeError(
            'Positive likelihood ratio can not be set, '
            'it must be computed from confusion matrix'
        )

    @property
    def lr_minus(self):
        return self._lr_minus

    @lr_minus.setter
    def lr_minus(self, value):
        raise AttributeError(
            'Negative likelihood ratio can not be set, '
            'it must be computed from confusion matrix'
        )

    @property
    def dor(self):
        return self._dor

    @dor.setter
    def dor(self, value):
        raise AttributeError(
            'Diagnostic odds ratio can not be set, '
            'it must be computed from confusion matrix'
        )

    @property
    def prev(self):
        return self._prev

    @prev.setter
    def prev(self, value):
        raise AttributeError('Prevalence can not be set, it must be computed from confusion matrix')

    def __init_params(self):
        self._acc = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
        self._tpr = self.tp / (self.tp + self.fn)
        self._tnr = self.tn / (self.tn + self.fp)
        self._fpr = 1 - self.tpr
        self._fnr = 1 - self.tnr
        self._ppv = self.tp / (self.tp + self.fp)
        self._npv = self.tn / (self.tn + self.fn)
        self._fdr = 1 - self.ppv
        self._for = 1 - self.npv
        self._pt = np.sqrt(self.fpr) / (np.sqrt(self.tpr) + np.sqrt(self.fpr))
        self._ts = self.tp / (self.tp + self.fn + self.fp)
        self._ba = (self.tpr + self.tnr) / 2
        self._f1 = 2 * (self.ppv * self.tpr) / (self.ppv + self.tpr)
        self._mcc = (self.tp * self.tn - self.fp * self.fn) / \
            np.sqrt(
            (
                self.tp + self.fp
            ) * (
                self.tp + self.fn
            ) * (
                self.tn + self.fp
            ) * (
                self.tn + self.fn
            )
        )
        self._fm = np.sqrt(self.ppv * self.tpr)
        self._bm = self.tpr + self.tnr - 1
        self._mk = self.ppv + self.npv - 1
        self._lr_plus = self.tpr / self.fpr
        self._lr_minus = self.fnr / self.tnr
        self._dor = self.lr_plus / self.lr_minus
        self._prev = (self.tp + self.fp) / (self.tp + self.fp + self.fn + self.tn)

    def get_confusion(self):
        return self.tp, self.tn, self.fp, self.fn
