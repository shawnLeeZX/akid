"""
This module holds classes that standardize evaluation metrics on algorithms.
"""
from __future__ import division

from __future__ import absolute_import
import numpy as np
from sklearn.metrics import roc_auc_score

from .blocks import DataBlock
from .. import backend as A


class EvalBlock(DataBlock):
    """
    Evaluation blocks serve the function to accumulate evaluation metric on
    batches, and summarize the accumulated results on these batches.

    It overrides the `+` operator to accumulate evaluation results. Current
    summary on the accumulated batches is obtainable through the `data`
    property.

    For example, to register an `EvalBlock` for a tensor::

        from akid import backend as A
        from akid import BatchEvalBlock

        a = ... # A is a tensor with a name
        A.register_eval_block(A.get_name(a), BatchEvalBlock)
    """
    NAME = "Eval"


class BatchEvalBlock(EvalBlock):
    NAME = "BatchEval"

    def __init__(self, v=0):
        self.acc_value = v
        self.num_batches = 0 if v == 0 else 1

    def add(self, v):
        self.acc_value += v
        self.num_batches += 1

    @property
    def data(self):
        return self.acc_value / self.num_batches


class AUCEvalBlock(EvalBlock):
    NAME = "AUC"

    def __init__(self):
        self.y_pred = []
        self.y = []

    def add(self, v):
        self.y_pred.append(v[0])
        self.y.append(v[1])

    @property
    def data(self):
        auc = []
        y = np.stack(self.y)
        y_pred = np.stack(self.y_pred)
        assert y.shape[1] == len(self.y[0]), "`np.stack` works wrongly."

        for i in range(len(self.y)):
            if not all(y[1:, i] == y[0, i]):
                auc.append(roc_auc_score(y[:, i], y_pred[:, i], average=None))
        return np.mean(auc)
