#!/usr/bin/env python

from __future__ import print_function

import argparse
import operator

import tensorflow as tf

from akid import VALIDATION_ACCURACY_TAG, LEARNING_RATE_TAG

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
arguments = parser.parse_args()

EVENT_FILE_PATH = arguments.EVENT_FILE

# Gather the list of validation accuracy.
val_acc_tuples = []
for e in tf.train.summary_iterator(EVENT_FILE_PATH):
    val_acc = None
    lr = None
    for v in e.summary.value:
        if v.tag == VALIDATION_ACCURACY_TAG:
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
    print("Rank {} : iter {}, accuracy {:.5f}, lr {:.8f}".format(i,
                                                                 t[0],
                                                                 t[1],
                                                                 t[2]))
