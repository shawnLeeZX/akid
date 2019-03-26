"""
A module to provide a mechanism to ease network tuning.
"""
from __future__ import print_function

from __future__ import absolute_import
import sys
import inspect
import multiprocessing
from .semaphore import Semaphore
import subprocess
import os

from jinja2 import Template
from tqdm import tqdm
import pycuda.autoinit
import pycuda.driver as cuda
import gflags as flags
from six.moves import range

FLAGS = flags.FLAGS
flags.DEFINE_boolean("use_sub_shell", True, "Deprecated. Not used."
                     " Use sub shell to run training"
                     " instances or not. This is to get around the issue"
                     " tensorflow will not release memory after a process"
                     " finishes its work. So to force tensorflow release "
                     " resources, just run the process in a sub shell.")
flags.DEFINE_integer("gpu_start_No", 0, "The start No of GPU to use. This flag"
                     " is to make sure the correct GPU mask is passed when"
                     " running training instances using subshell. For example,"
                     " if you are going to use GPU 1-9, you need to pass"
                     " --gpu_start_no=1.")
flags.DEFINE_boolean("chk_point", False, "Whether to continue from check point."
                     " This option is for the interrupted training instances. Make sure"
                     " `log_dir` is the same to let akid find the checkpoint.")

NETWORK_LOG_HEADER = "Network Setup: \\n"


def spawn_using_sub_shell(setup_func, work_dir, idxs):
    gpu_No_str = ""
    # Make comma separated gpu No list.
    for idx in idxs:
        gpu_No_str += "{},".format(idx + FLAGS.gpu_start_No)

    # Add training code to the end.
    training_call = """


kid = setup()
import inspect
from akid.utils import glog as log
log.info("{}" + inspect.getsource(setup))
kid.practice(continue_from_chk_point={})
    """.format(NETWORK_LOG_HEADER, FLAGS.chk_point)

    training_code = setup_func + training_call
    # Save code to file.
    file_name = "net_{}.py".format(gpu_No_str)
    with open(os.path.join(work_dir, file_name), 'w') as f:
        f.write(training_code)
    # Run.
    subprocess.call(
        "cd {}; CUDA_VISIBLE_DEVICES={} python {}".format(
            work_dir,
            gpu_No_str,
            file_name),
        shell=True)


def spawn(s, l, gpu_mask, gpu_num, return_values, setup_func, repeat):
    s.acquire(gpu_num)
    print ("Acquired resources. Remaining semaphore {}".format(s.semaphore))

    with l:
        # Look up GPU(s) and mark it used.
        # The lock is to make sure the logging is not mixed.
        print ("Entering critical region.")
        acquired_gpu = 0
        idxs = []
        for idx, avail in enumerate(gpu_mask):
            if avail == 1:
                gpu_mask[idx] = 0
                idxs.append(idx)
                acquired_gpu += 1
                if acquired_gpu == gpu_num:
                    break

        print("GPU mask {}.".format(gpu_mask))
        print("Using GPU {}.".format(idxs))

        print ("Exiting critical region.")

    repeat_folder = str(repeat)
    # Create folder to hold one training repeat.
    if not os.path.exists(repeat_folder):
        os.mkdir(repeat_folder)
    work_dir = repeat_folder

    spawn_using_sub_shell(setup_func, work_dir, idxs)

    # Release the GPU.
    with l:
        print ("Entering critical region.")
        for idx in idxs:
            print("Released GPU {}.".format(idx))
            gpu_mask[idx] = 1
            print("GPU mask {}.".format(gpu_mask))

        print ("Exiting critical region.")

    s.release(gpu_num)
    print ("Released resources. Remaining semaphore {}".format(s.semaphore))

def tune(template,
         opt_paras_list=[{}],
         net_paras_list=[{}],
         repeat_times=1,
         gpu_num_per_instance=1,
         debug=False):
    """
    A function `tune` that takes a Brain jinja2 template class and a parameters
    to fill the template in runtime. Parameters provided should complete the
    remaining network parameters in the template. The tuner is not aware of the
    content of the list items. It is up to the user to define template right,
    so parameters will be filled in the right place.

    The jinja2 template must be a function named `setup`, and return a set up
    `Kid`. All necessary module imports should be put in the function instead
    of module level import usually.

    The `tune` function would use all available GPUs to train networks with all
    given different set of parameters. If available GPUs are not enough, the
    ones that cannot be trained will wait till some others finish, and get its
    turn.

    ## Parameter Tuning Usage

    Tunable parameters are divided into two set, network hyper parameters,
    `net_paras_list`, and optimization hyper parameters, `opt_paras_list`. Each
    set is specified by a list whose item is a dictionary that holds the actual
    value of whatever hyper parameters defined as jinja2 templates. Each item
    in the list corresponds to a tentative training instance. network paras and
    optimization paras combine with each other exponentially(or in Cartesian
    Product way if we could use Math terminology), which is to say if you have
    two items in network parameter list, and two in optimization parameters,
    the total number of training instances will be four.

    Final training precisions will be returned as a list. Since the final
    precision normally will not be the optimal one, which normally occurs
    during training, the returned values are used for testing purpose only now

    ## Run repeated experiment

    To run repeated experiment, just leave `opt_paras_list` and
    `net_paras_list` to their default value.

    ## GPU Resources Allocation

    If the `gpu_num_per_instance` is None, a gpu would be allocated to each
    thread, otherwise, the length of the list should be the same with that of
    the training instance (aka the #opt_paras_list * #net_paras_list *
    repeat_times), or an int.

    Given the available GPU numbers, a semaphore is created to control access
    to GPUs. A lock is created to control access to the mask to indicator which
    GPU is available. After a process has modified the gpu mask, it releases
    the lock immediately, so other process could access it. But the semaphore
    is still not release, since it is used to control access to the actual
    GPU. A training instance will be launched in a subshell using the GPU
    acquired. The semaphore is only released after the training has finished.

    ## Example

    For example, to tune the activation function and learning rates of a
    network, first we set up network parameters in `net_paras_list`,
    optimization parameters in `opt_paras_list`, build a network in the `setup`
    function, then pass all of it to tune::

        net_paras_list = []
        net_paras_list.append({
            "activation": [
                {"type": "relu"},
                {"type": "relu"},
                {"type": "relu"},
                {"type": "relu"}],
            "bn": True})
        net_paras_list.append({
            "activation": [
                {"type": "maxout", "group_size": 2},
                {"type": "maxout", "group_size": 2},
                {"type": "maxout", "group_size": 2},
                {"type": "maxout", "group_size": 5}],
            "bn": True})

        opt_paras_list = []
        opt_paras_list.append({"lr": 0.025})
        opt_paras_list.append({"lr": 0.05})

        def setup(graph):

            brain.attach(cnn_block(
                ksize=[8, 8],
                init_para={
                    "name": "uniform",
                    "range": 0.005},
                wd={"type": "l2", "scale": 0.0005},
                out_channel_num=384,
                pool_size=[4, 4],
                pool_stride=[2, 2],
                activation={{ net_paras["activation"][1] }},
                keep_prob=0.5,
                bn={{ net_paras["bn"] }}))

        tune(setup, opt_paras_list, net_paras_list)
    """
    # Parse command line flags
    FLAGS(sys.argv)
    # Set up data structures.
    # #########################################################################
    manager = multiprocessing.Manager()
    gpu_num = cuda.Device.count()
    gpu_mask = manager.list([1] * gpu_num)
    return_values = manager.list()

    net_num = len(net_paras_list)
    opt_num = len(opt_paras_list)

    if type(gpu_num_per_instance) is not int:
        if len(net_paras_list) * len(opt_paras_list) * repeat_times \
           != len(gpu_num_per_instance):
            raise Exception("""
            The number of gpu used per training instance should match
            `#net_paras_list({}) * #opt_paras_list({}) * repeat_times({}): {}`,
            or a single int.
            """.format(net_num,
                       opt_num,
                       repeat_times,
                       net_num * opt_num * repeat_times)
            )

    # Logistics
    # #########################################################################
    s = Semaphore(len(gpu_mask))
    l = multiprocessing.Lock()
    process_pool = []
    template_str = Template(inspect.getsource(template))

    # Start tuning.
    # #########################################################################
    for repeat in range(0, repeat_times):
        for i, opt_paras in enumerate(opt_paras_list):
            for j, net_paras in enumerate(net_paras_list):
                setup_func = template_str.render(opt_paras=opt_paras,
                                                 net_paras=net_paras)
                _gpu_num = gpu_num_per_instance[
                    repeat*(net_num * opt_num) + i*net_num + j] \
                    if type(gpu_num_per_instance) is list \
                    else gpu_num_per_instance
                p = multiprocessing.Process(target=spawn,
                                            args=(s,
                                                  l,
                                                  gpu_mask,
                                                  _gpu_num,
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
