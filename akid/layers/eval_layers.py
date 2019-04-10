from __future__ import absolute_import
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

from ..core.blocks import ProcessingLayer
from ..backend import NamedTensorTuple, NamedScalar
from .. import backend as A
from .. import nn
from six.moves import range


class EvalLayer(ProcessingLayer):
    NAME = "Eval"

    @property
    def data(self):
        return self._eval


class AUCLayer(EvalLayer):
    """
    Args:
        return_average_auc: bool
            If is set `True`, and the labels are multi-label, an label-wise
            averaged result would be returned. Otherwise, return a list of auc
            values, each of which (in the list) is the auc for that particular
            class.
    """
    NAME = "AUC"

    def __init__(self, return_average_auc=True, *args, **kwargs):
        super(AUCLayer, self).__init__(*args, **kwargs)

        self.return_average_auc = return_average_auc

    def _forward(self, inputs):
        v = A.eval(inputs)
        y, y_pred = v[1], v[0]

        self._eval = None
        if self.return_average_auc:
            auc = []
            for i in range(y.shape[1]):
                if np.sum(y[:, i]) != 0:
                    auc.append(roc_auc_score(y[:, i], y_pred[:, i], average='macro'))

            self._eval = NamedScalar("batch_AUC", np.mean(auc))

        if self.is_val:
            self._verbose_eval = NamedTensorTuple("AUC", (y_pred, y))

        return self._eval


class MultiLabelAccuracy(EvalLayer):
    """
    Given a tuple of (prediction score, label), compute the mean accuracy of
    multi-label classification.
    """
    def _forward(self, inputs):
        inputs = A.eval(inputs)
        for i, v in enumerate(inputs):
            shape = A.get_shape(v)
            if len(shape) != 2:
                raise ValueError("Input {} should be a 2-dim array. Got {} instead.".format(i, shape))

        mean_precision = []
        for i in range(shape[1]):
            if np.sum(inputs[1][:, i]) != 0:
                mean_precision.append(average_precision_score(inputs[1][:, i], inputs[0][:, i]))

        mean_precision = np.mean(mean_precision)
        mean_precision = A.NamedScalar("MAP", mean_precision)
        self._eval = mean_precision

        return self._eval


class BinaryAccuracy(EvalLayer):
    NAME = "Binary_Acc"

    def __init__(self, hinge_loss_label=False, *args, **kwargs):
        super(BinaryAccuracy, self).__init__(*args, **kwargs)
        self.hinge_loss_label = hinge_loss_label

    def _forward(self, inputs):
        x, y = inputs[0], inputs[1]
        self._eval = nn.binary_accuracy(x, y, self.hinge_loss_label, name="Acc")
        return self._eval
