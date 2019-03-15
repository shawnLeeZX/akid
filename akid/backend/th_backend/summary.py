import multiprocessing
import Queue

import torch as th
from torch.autograd import Variable
from tensorboardX import SummaryWriter

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
_done_event = None
_queue = None
QUEUE_SIZE = 1000
summary_writer = None
_denormalize = None


class Denormalize(object):
    """De-normalize an tensor image with mean and standard deviation.

    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = channel * std + mean

    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respectively.
    """

    def __init__(self, mean, std):
        # TODO: handle the manual type cast here. Wont' work at least when in
        # cpu tensor.
        self.mean = Variable(th.Tensor(mean).cuda())
        self.std = Variable(th.Tensor(std).cuda())

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be de-normalized.

        Returns:
            Tensor: Normalized image.
        """
        tensor = th.mul(tensor, self.std[None, None, :])
        tensor = th.add(tensor, self.mean[None, None, :])
        return tensor


def set_normalization(mean, std):
    """
    Set the normalization values for image summary, which is used to
    de-normalize the image for better view, given that the normalized images are
    hard to recognize.
    """
    global _denormalize
    _denormalize = Denormalize(mean, std)


def reset_collections():
    global _collections

    _collections = {}
    _collections[TRAIN_SUMMARY_COLLECTION] = []
    _collections[VALID_SUMMARY_COLLECTION] = []
    _collections[TRAINING_DYNAMICS_COLLECTION] = []


class SummaryOp(object):
    def __init__(self, name):
        if name is None:
            raise ValueError("Name of the summary cannot be None.")

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

    def __call__(self, step, v):
        try:
            self._call(step, v)
        except Exception as e:
            print self.__str__()
            raise e


class HistogramSummaryOp(SummaryOp):
    def __init__(self, summary_on_grad=False, **kwargs):
        super(HistogramSummaryOp, self).__init__(**kwargs)
        self.summary_on_grad = summary_on_grad

    def _call(self, step, tensor):
        summary_writer.add_histogram(
            self.name, cg.eval(tensor), global_step=step, bins='auto')
        if self.summary_on_grad:
            grad = tensor.grad
            if grad is not None:
                summary_writer.add_histogram(
                    self.name + "_grad", cg.eval(grad), global_step=step, bins='auto')
            else:
                log.debug("Gradient of {} is None. Skip summary.".format(self.name))


class ScalarSummaryOp(SummaryOp):
    def _call(self, step, t):
        v = cg.eval(t)  if type(t) is Variable else t
        summary_writer.add_scalar(
            self.name, v, global_step=step)


class ImageSummaryOp(SummaryOp):
    def _call(self, step, t):
        if len(t.size()) == 4:
            t = t[0]
            # Since TensorboardX 1.5, it handles the torch format (CHW) by
            # default. Permuted shape without specifying data format would lead
            # to errors.
            # t = t.permute(1, 2, 0)
        if _denormalize is not None:
            t = _denormalize(t)
        if cg.torch_version < 0.4:
            v = cg.eval(t)  if type(t) is Variable else t
        else:
            v = cg.eval(t)  if type(t) is th.Tensor else t
        summary_writer.add_image(
            self.name + ' image', v, global_step=step)


def _summary_writer_worker(dir, queue, done_event):
    global summary_writer
    summary_writer = SummaryWriter(dir)
    while True:
        if done_event.is_set() and queue.empty():
            return

        try:
            op, v, step = queue.get(timeout=cg_general.TIMEOUT)
            op(step, v)
        except Queue.Empty:
            continue


def init(dir=None):
    """
    Create the summary file writer.

    It creates a process that polls a queue to see if are there anymore summary
    to write.

    Args:
        dir: str
            The directory to save event files.
    """
    global _done_event
    global _queue
    _done_event = multiprocessing.Event()
    _queue = multiprocessing.Queue(QUEUE_SIZE)
    worker_process = multiprocessing.Process(target=_summary_writer_worker, args=(dir, _queue, _done_event))
    worker_process.start()


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
    op = ScalarSummaryOp(name)
    _queue.put((op, value, step))


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


# TODO: since queue is used, we do not need to wait to run such merged summary;
# we can add the summary to be written to on-the-fly during training.
def run_summary_op(ops, feed_dict=None):
    """
    Args:
        op: a list of SummaryOp.
        feed_dict: not used. Tensorflow compatibility.
    """
    if isinstance(ops, SummaryOp):
        ops = [ops]
    elif type(ops) is list:
        pass
    else:
        raise ValueError("ops should be a `SummaryOp` or a list; got {}".format(type(op)))

    # NOTE: it may take some time to transfer from GPU to CPU, if this becomes
    # a bottleneck, a thread should be used to transfer the data.
    op_values = [cg.tensor_by_name[op.name].to("cpu") for op in ops]
    op_value_tuples = zip(ops, op_values)
    for t in op_value_tuples:
        # TODO: resize queue if its size is not enough.
        _queue.put((t[0], t[1], cg_general.get_step()))


def close():
    if _done_event is not None:
        _done_event.set()


reset_collections()
