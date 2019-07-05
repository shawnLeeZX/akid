"""
This module holds classes that standardize evaluation metrics on algorithms.
"""
from __future__ import division

from __future__ import absolute_import
import numpy as np
from sklearn.metrics import roc_auc_score

from .blocks import DataBlock


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

    It also provide `__str__` method to convert evaluation metrics to a text
    representation. Thus, it may enable a uniform logging interface.
    """
    NAME = "Eval"

    def __str__(self, data=None):
        """
        When taking argument `data`, it converts the `data` to text
        representation, instead of converting the internal data it holds now.
        """
        if data is None:
            data = self.data

        return str(self.data)


class BatchEvalBlock(EvalBlock):
    # TODO: it has a bug. If the last batch is smaller, the weights of its
    # samples are increased, thus leading to wrong accuracy calculation.
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

    def __str__(self, data=None):
        if data is None:
            data = self.data

        return "{:0.4f}".format(data)


class MultiEvalBlock(EvalBlock):
    """
    The EvalBlock that keeps and accumulates a list of accuracy.
    """
    def __init__(self, v=None):
        self.acc_value = v
        self.num_batches = [0] * len(v) if v is not None else None

    def add(self, v):
        if self.acc_value is None:
            self.acc_value = list(v[:])
            self.num_batches = [1] * len(self.acc_value)
            # Remove possible nan values.
            for i, v in enumerate(self.acc_value):
                if np.isnan(v):
                    self.acc_value[i] = 0
                    self.num_batches[i] = 0
        else:
            for i, _ in enumerate(self.acc_value):
                if not np.isnan(v[i]):
                    # Sometimes, we do not have certain info in a batch. In
                    # this case the value should be passed as nan, and we do
                    # accumulate the results.
                    self.acc_value[i] += v[i]
                    self.num_batches[i] += 1

    @property
    def data(self):
        return [v / self.num_batches[i] for i, v in enumerate(self.acc_value)]

    def __str__(self, data=None):
        if data is None:
            data = self.data

            data_str = ("{:0.4f} " * len(data)).format(*data)
            num_batches_str = ("{} " * len(self.num_batches)).format(*self.num_batches)
            return data_str + num_batches_str
        else:
            return ("{:0.4f} " * len(data)).format(*data)



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
        y = np.stack(self.y)
        y_pred = np.stack(self.y_pred)

        assert y.shape[1] == len(self.y[0]), "`np.stack` works wrongly."

        auc = []
        for i in range(y.shape[1]):
            if len(y[:, i]) > 1:
                if not all(y[1:, i] == y[0, i]):
                    auc.append(roc_auc_score(y[:, i], y_pred[:, i], average=None))
        if len(auc) == 0:
            col_auc = None
        else:
            col_auc = np.mean(auc)

        auc = []
        for i in range(y.shape[0]):
            if len(y[i, :]) > 1:
                if not all(y[i, 1:] == y[i, 0]):
                    auc.append(roc_auc_score(y[i, :], y_pred[i, :], average=None))
        if len(auc) == 0:
            row_auc = None
        else:
            row_auc = np.mean(auc)

        return [row_auc, col_auc]

    def __str__(self, data=None):
        if data is None:
            data = self.data

        return "Row AUC: {}, Col AUC {}".format(data[0], data[1])



# Eval blocks for evaluations.
# #######################################################################
tensor_name_to_eval_block_map = {}


def register_eval_block(tensor_name, eval_block):
    """
    For different evaluation metrics, it may take different ways to combine
    the evaluation obtained in batches. This function registers an
    `EvalBlock` to handle the combination for evaluation results obtained
    in the tensor named `tensor_name`.

    If for an evaluation metric tensor that is not registered, `BatchEvalBlock`
    be its eval block by default.
    """
    tensor_name_to_eval_block_map[tensor_name] = eval_block


def get_eval_block(tensor_name):
    try:
        return tensor_name_to_eval_block_map[tensor_name]
    except KeyError:
        return BatchEvalBlock


def reset_eval_block_map():
    global tensor_name_to_eval_block_map
    tensor_name_to_eval_block_map = {}
