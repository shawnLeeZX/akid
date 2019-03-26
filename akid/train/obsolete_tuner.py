"""
See the argument parser's description below for introduction and usage.

# Mechanism
# #########################################################################
Given the available GPU numbers, a semaphore is created to control access to
GPUs. A lock is created to control access to the mask to indicator which GPU is
available. After a process has modified the gpu mask, it releases the lock
immediately, so other process could access it. But the semaphore is still not
release, since it is used to control access to the actual GPU. A training
instance will be launched in a subshell using the GPU acquired. The semaphore
is only released after the training has finished.
"""
from __future__ import absolute_import
from __future__ import print_function
import multiprocessing
import importlib
import argparse

import tensorflow as tf
import pycuda.autoinit
import pycuda.driver as cuda
from six.moves import range


def spawn(s, l, gpu_mask, **kwargs):
    with s:
        # Look up a GPU and mark it used.
        # A lock is unnecessary for manager list, but in order to let the
        # printed information print right, a lock is used to control access to
        # stdout.
        with l:
            for idx, avail in enumerate(gpu_mask):
                if avail == 1:
                    print(("GPU mask {}.".format(gpu_mask)))
                    print(("Using GPU {}.".format(idx)))
                    gpu_mask[idx] = 0
                    break

        # Train using the acquired GPU.
        with tf.device("/gpu:{}".format(idx)):
            # Built the graph here.
            kid = trainee.setup(tf.get_default_graph(), **kwargs)

        # Actually call the code that run a session.
        kid.practice()

        # Release the GPU.
        with l:
            print(("Released GPU {}.".format(idx)))
            gpu_mask[idx] = 1


# Setup argument parser.
parser = argparse.ArgumentParser(
    description="""
Description:
# #########################################################################
A template for tuning parameters using multiple GPUs. Different trainees will
be put on different GPUs.

Usage:
# #########################################################################
A trainee is a script with a function `setup` that would set up and return a
Kid, but not train it, which is to say, call the `practice` method.

The `setup` function should expose the network parameter to be tuned as formal
parameters.

Currently, supported parameters are:

    * learning rate

See maxout.py in the same folder for an example.
    """,
    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("TRAINEE_FILENAME",
                    type=str,
                    help="the path to the table to be drawn. The file has to"
                    " be in the same folder for now.")
parser.add_argument("-lr", "--learning-rates",
                    type=float,
                    nargs='+',
                    help="The list of learning rates to try.")
arguments = parser.parse_args()

# Gather trainee info.
# #########################################################################
# Strip the file suffix
trainee_name = arguments.TRAINEE_FILENAME.split('.')[0]
trainee = importlib.import_module(trainee_name)

lr_list = arguments.learning_rates

# Set up data structures.
# #########################################################################
manager = multiprocessing.Manager()
gpu_num = cuda.Device.count()
gpu_mask = manager.list([1] * gpu_num)

# Logistics
# #########################################################################
s = multiprocessing.Semaphore(len(gpu_mask))
l = multiprocessing.Lock()
process_pool = []

# Start tuning.
# #########################################################################
for i in range(len(lr_list)):
    p = multiprocessing.Process(target=spawn,
                                args=(s, l, gpu_mask),
                                kwargs={"lr": lr_list[i]})
    process_pool.append(p)
    p.start()

# Wait for all processes to finish.
for p in process_pool:
    p.join()
