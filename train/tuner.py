"""
A module to provide a mechanism to ease network tuning.

It defines a function `tune` that takes a Brain jinja2 template class and a
parameters to fill the template in runtime. Parameters provided should complete
the remaining network parameters in the template. The tuner is not aware of the
content of the list items. It is up to the user to define template right, so
parameters will be filled in the right place.

The jinja2 template must be a function named `setup`, and return a set up
`Survivor`. All necessary module imports should be put in the function instead
of module level import usually.

The `tune` function would use all available GPUs to train networks with all
given different set of parameters. If available GPUs are not enough, the ones
that cannot be trained will wait till some others finish, and get its turn.

## Parameter Tuning Usage

Tunable parameters are divided into two set, network hyper parameters and
optimization hyper parameters. Each set is specified by a list whose item is a
dictionary that holds the actual value of whatever hyper parameters defined as
jinja2 templates. Each item in the list corresponds to a tentative training
instance. network paras and optimization paras combine with each other
exponentially(or in Cartesian Product way if we could use Math terminology),
which is to say if you have two items in network parameter list, and two in
optimization parameters, the total number of training instances will be four.

Final training precisions will be returned as a list. Since the final precision
normally will not be the optimal one, which normally occurs during training,
the returned values are used for testing purpose only now

## GPU Resources Allocation

Given the available GPU numbers, a semaphore is created to control access to
GPUs. A lock is created to control access to the mask to indicator which GPU is
available. After a process has modified the gpu mask, it releases the lock
immediately, so other process could access it. But the semaphore is still not
release, since it is used to control access to the actual GPU. A training
instance will be launched in a subshell using the GPU acquired. The semaphore
is only released after the training has finished.
"""
from __future__ import print_function

import sys
import inspect
import multiprocessing
import subprocess
import os

from jinja2 import Template
from tqdm import tqdm
import tensorflow as tf
import pycuda.autoinit
import pycuda.driver as cuda
import gflags as flags

from ..utils import glog as log

FLAGS = flags.FLAGS
flags.DEFINE_boolean("use_sub_shell", True, "Use sub shell to run training"
                     " instances or not. This is to get around the issue"
                     " tensorflow will not release memory after a process"
                     " finishes its work. So to force tensorflow release "
                     " resources, just run the process in a sub shell.")
flags.DEFINE_integer("gpu_start_No", 0, "The start No of GPU to use. This flag"
                     " is to make sure the correct GPU mask is passed when"
                     " running training instances using subshell. For example,"
                     " if you are going to use GPU 1-9, you need to pass"
                     " --gpu_start_no=1.")

NETWORK_LOG_HEADER = "Network Setup: \\n"


def spawn(s, l, gpu_mask, return_values, setup_func, repeat):
    with s:
        # Look up a GPU and mark it used.
        # A lock is unnecessary for manager list, but in order to let the
        # printed information print right, a lock is used to control access to
        # stdout.
        with l:
            for idx, avail in enumerate(gpu_mask):
                if avail == 1:
                    print("GPU mask {}.".format(gpu_mask))
                    print("Using GPU {}.".format(idx))
                    gpu_mask[idx] = 0
                    break

        if FLAGS.use_sub_shell:
            repeat_folder = str(repeat)
            # Create folder to hold one training repeat.
            if not os.path.exists(repeat_folder):
                os.mkdir(repeat_folder)

            # Add training code to the end.
            training_call = """


kid = setup(None)
import inspect
from akid.utils import glog as log
log.info("{}" + inspect.getsource(setup))
kid.practice()
            """.format(NETWORK_LOG_HEADER)

            training_code = setup_func + training_call
            # Save code to file.
            file_name = "net_{}.py".format(idx)
            with open(os.path.join(repeat_folder, file_name), 'w') as f:
                f.write(training_code)
            # Run.
            subprocess.call(
                "cd {}; CUDA_VISIBLE_DEVICES={} python {}".format(
                    repeat_folder,
                    idx + FLAGS.gpu_start_No,
                    file_name),
                shell=True)
        else:
            # Train using the acquired GPU.
            # TODO(Shuai): handle repeat times.
            exec setup_func
            with tf.device("/gpu:{}".format(idx)):
                # Built the graph here.
                kid = setup(tf.get_default_graph())

            # Save network to log.
            log.info(NETWORK_LOG_HEADER + setup_func)
            # Actually call the code that run a session.
            precision = kid.practice()
            return_values.append(precision)

        # Release the GPU.
        with l:
            print("Released GPU {}.".format(idx))
            gpu_mask[idx] = 1


def tune(template, opt_paras_list, net_paras_list, repeat_times=1, debug=False):
    # Parse command line flags
    FLAGS(sys.argv)
    # Set up data structures.
    # #########################################################################
    manager = multiprocessing.Manager()
    gpu_num = cuda.Device.count()
    gpu_mask = manager.list([1] * gpu_num)
    return_values = manager.list()

    # Logistics
    # #########################################################################
    s = multiprocessing.Semaphore(len(gpu_mask))
    l = multiprocessing.Lock()
    process_pool = []
    template_str = Template(inspect.getsource(template))

    # Start tuning.
    # #########################################################################
    for repeat in xrange(0, repeat_times):
        for opt_paras in opt_paras_list:
            for net_paras in net_paras_list:
                setup_func = template_str.render(opt_paras=opt_paras,
                                                 net_paras=net_paras)
                p = multiprocessing.Process(target=spawn,
                                            args=(s,
                                                  l,
                                                  gpu_mask,
                                                  return_values,
                                                  setup_func,
                                                  repeat))
                process_pool.append(p)
                p.start()

    # Wait for all processes to finish.
    for p in tqdm(process_pool):
        p.join()

    # TODO(Shuai): Think what should be the return value for subprocess call.
    return return_values
