#!/usr/bin/env python

from __future__ import print_function

from __future__ import absolute_import
import argparse
import operator

import tensorflow as tf

from akid import LEARNING_RATE_TAG

parser = argparse.ArgumentParser(
    description="""
Description:
# #########################################################################
Given a tensorflow event file, print out the top 10 accuracy during training.
    """,
    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("EVENT_FILE",
                    type=str,
                    help="path to the event file.")
parser.add_argument("eval_metric",
                    type=str,
                    help="""
                    The evaluation metric to evaluate the model. Since a model
                    may have multiple evaluation metrics associated with it,
                    and normally only one of them is critical (at least for
                    now), it is supposed to be told. The name is the name of an
                    evaluation metric in the training log. A full name is not
                    needed as long as the part provided is unique enough to
                    distinguish it from other metrics (otherwise, the last
                    match will be used, which normally will not be the
                    desirable behavior).
                    """)
parser.add_argument("-s", "--step",
                    type=int,
                    help="""
                    The start step to check. The reason that this argument
                    exists is, in certain situation, accuracy reported is wrong
                    at the beginning of the training, thus should not be
                    counted when trying to find the max ones.
                    """)
arguments = parser.parse_args()

EVENT_FILE_PATH = arguments.EVENT_FILE
eval_metric_name = arguments.eval_metric
start_step = arguments.step

# Gather the list of validation accuracy.
val_acc_tuples = []
previous_step = None
for e in tf.train.summary_iterator(EVENT_FILE_PATH):
    if e.step < start_step:
        continue
    # Since a step may be logged by multiple event files, state variables are
    # only updated when we are switching steps.
    if e.step != previous_step:
        val_acc = None
        lr = None
        previous_step = e.step

    for v in e.summary.value:
        if v.tag.find(eval_metric_name) is not -1:
            val_acc = v.simple_value
        if v.tag == LEARNING_RATE_TAG:
            lr = v.simple_value
    if val_acc and lr:
        val_acc_tuples.append((e.step, val_acc, lr))


# Find and print the top 10.
sorted_val_acc_tuples = sorted(val_acc_tuples,
                               key=operator.itemgetter(1),
                               reverse=True)
for i, t in enumerate(sorted_val_acc_tuples[0:10]):
    print("Rank {} : iter {}, accuracy ({}) {:.5f}, lr {:.8f}".format(
        i, t[0], eval_metric_name, t[1], t[2]))
