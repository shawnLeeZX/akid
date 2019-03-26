from __future__ import absolute_import
import tensorflow as tf

from akid.utils import glog as log
from akid.core.common import (
    TRAIN_SUMMARY_COLLECTION,
    VALID_SUMMARY_COLLECTION,
    TRAINING_DYNAMICS_COLLECTION,
)
from .. import computational_graph as general_cg
from . import computational_graph as cg


summary_writer = None
summary_ops = None


def init(dir=None):
    """
    Create the summary file writer.

    Args:
        dir: str
            The directory to save event files.
    """
    global summary_writer
    if not dir:
        dir = log.get_random_log_dir()
    summary_writer = tf.summary.FileWriter(dir)

    log.info("Summary event file will be saved to {}".format(dir))


def histogram(name, values, summary_on_grad=False, collections=None, step=None):
    """
    `step`, `summary_on_grad` are not used, just for compatibility with PyTorch.
    """
    tf.summary.histogram(name, values, collections)


def image(name, values, collections=None, step=None):
    tf.summary.image(name, values, collections=collections)


def scalar(name, value, collections=None, step=None):
    """
    Same as `histogram`.
    """
    if len(value.get_shape().as_list()) != 0:
        value = tf.reshape(value, [])
    tf.summary.scalar(name, value, collections=collections)


def add_graph():
    summary_writer.add_graph(tf.get_default_graph())


def add_scalar(name, value, step):
    summary = tf.Summary()
    summary.value.add(tag=name, simple_value=value)
    summary_writer.add_summary(summary, step)


def get_collection(name=None):
    if name:
        return tf.get_collection(name)
    else:
        ret = []
        ret.extend(tf.get_collection(TRAIN_SUMMARY_COLLECTION))
        ret.extend(tf.get_collection(VALID_SUMMARY_COLLECTION,))
        ret.extend(tf.get_collection(TRAINING_DYNAMICS_COLLECTION))

        return ret


def run_summary_op(op, feed_dict=None):
    summary_str = cg.sess.run(op, feed_dict=feed_dict)
    summary_writer.add_summary(summary_str, general_cg.get_step())


def merge(ops):
    return tf.summary.merge(ops)
