from torch.autograd import Variable
from tensorboard import SummaryWriter

from . import computational_graph as cg
from .. import computational_graph as cg_general

# Collections to keep summary ops, so it can be run later in Kid.
from akid.core.common import (
    TRAIN_SUMMARY_COLLECTION,
    VALID_SUMMARY_COLLECTION,
    TRAINING_DYNAMICS_COLLECTION,
)
_collections = {}
_collections[TRAIN_SUMMARY_COLLECTION] = []
_collections[VALID_SUMMARY_COLLECTION] = []
_collections[TRAINING_DYNAMICS_COLLECTION] = []


summary_writer = None


class SummaryOp(object):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class HistogramSummaryOp(SummaryOp):
    def __call__(self, step):
        summary_writer.add_histogram(
            self.name, cg.eval(cg.tensor_by_name[self.name]), global_step=step, bins='auto')


class ScalarSummaryOp(SummaryOp):
    def __call__(self, step):
        t = cg.tensor_by_name[self.name]
        v = cg.eval(t)  if type(t) is Variable else t
        summary_writer.add_scalar(
            self.name, v, global_step=step)


def init(dir=None):
    """
    Create the summary file writer.

    Args:
        dir: str
            The directory to save event files.
    """
    global summary_writer
    summary_writer = SummaryWriter(dir)


def histogram(name, values, collections=None):
    for c in collections:
        _collections[c].append(HistogramSummaryOp(name))


def scalar(name, value, collections=None):
    for c in collections:
        _collections[c].append(ScalarSummaryOp(name))


def add_scalar(name, value, step):
    summary_writer.add_scalar(name, value, global_step=step)


def add_graph():
    """Does nothing. Since the implementation of tensorboard-pytorch is buggy now."""
    pass


def get_collection(name=None):
    """
    If name is None, return all summary ops.
    """
    if name:
        return _collections[name]
    else:
        ret = []
        for v in _collections.itervalues():
            ret.extend(v)
        return ret


def merge(l):
    """Does nothing. Just to be compatible with Tensorflow backend"""
    return l


def run_summary_op(op, feed_dict=None):
    """
    Args:
        op: a list of SummaryOp.
        feed_dict: not used. Tensorflow compatibility.
    """
    if type(op) is list:
        for c in op:
            c(cg_general.get_step())
    else:
        c(cg_general.get_step(op))
