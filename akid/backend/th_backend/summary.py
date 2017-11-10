from torch.autograd import Variable
from tensorboard import SummaryWriter

from akid.utils import glog as log

from . import computational_graph as cg
from .. import computational_graph as cg_general

# Collections to keep summary ops, so it can be run later in Kid.
from akid.core.common import (
    TRAIN_SUMMARY_COLLECTION,
    VALID_SUMMARY_COLLECTION,
    TRAINING_DYNAMICS_COLLECTION,
)
from akid.utils.tools import currentframe


_collections = None
summary_writer = None


def reset_collections():
    global _collections

    _collections = {}
    _collections[TRAIN_SUMMARY_COLLECTION] = []
    _collections[VALID_SUMMARY_COLLECTION] = []
    _collections[TRAINING_DYNAMICS_COLLECTION] = []


class SummaryOp(object):
    def __init__(self, name):
        self.name = name
        # Save where the summary is initialized.
        f = currentframe()
        f = f.f_back.f_back.f_back
        co = f.f_code
        self.position = (co.co_filename, f.f_lineno, co.co_name)

    def __str__(self):
        return "<{} for {}> in {}".format(type(self).__name__, self.name, self.position)

    def __repr__(self):
        return "<{} for {}> in {}".format(type(self).__name__, self.name, self.position)

    def __call__(self, step):
        try:
            self._call(step)
        except Exception as e:
            print self.__str__()
            raise e


class HistogramSummaryOp(SummaryOp):
    def __init__(self, summary_on_grad=False, **kwargs):
        super(HistogramSummaryOp, self).__init__(**kwargs)
        self.summary_on_grad = summary_on_grad

    def _call(self, step):
        summary_writer.add_histogram(
            self.name, cg.eval(cg.tensor_by_name[self.name]), global_step=step, bins='auto')
        if self.summary_on_grad:
            grad = cg.tensor_by_name[self.name].grad
            if grad is not None:
                summary_writer.add_histogram(
                    self.name + "_grad", cg.eval(grad), global_step=step, bins='auto')
            else:
                log.debug("Gradient of {} is None. Skip summary.".format(self.name))


class ScalarSummaryOp(SummaryOp):
    def _call(self, step):
        t = cg.tensor_by_name[self.name]
        v = cg.eval(t)  if type(t) is Variable else t
        summary_writer.add_scalar(
            self.name, v, global_step=step)


class ImageSummaryOp(SummaryOp):
    def _call(self, step):
        t = cg.tensor_by_name[self.name]
        if len(t.size()) == 4:
            t = t[0]
            t = t.permute(1, 2, 0)
        v = cg.eval(t)  if type(t) is Variable else t
        summary_writer.add_image(
            self.name + ' image', v, global_step=step)


def init(dir=None):
    """
    Create the summary file writer.

    Args:
        dir: str
            The directory to save event files.
    """
    global summary_writer
    summary_writer = SummaryWriter(dir)


def histogram(name, values, summary_on_grad=False, collections=None):
    """
    Args:
        summary_on_grad: bool
            Whether to do summary on the grad of the variable.
    """
    for c in collections:
        _collections[c].append(HistogramSummaryOp(name=name, summary_on_grad=summary_on_grad))


def scalar(name, value, collections=None):
    for c in collections:
        _collections[c].append(ScalarSummaryOp(name))


def image(name, value, collections=None):
    for c in collections:
        _collections[c].append(ImageSummaryOp(name))


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
        op(cg_general.get_step())


reset_collections()
