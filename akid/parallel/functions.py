from __future__ import absolute_import
from .. import backend as A
from ..utils.tools import is_tuple_or_list
import torch as th
import tensorflow as tf

import threading
from six.moves import map
from six.moves import range
from six.moves import zip


_lock = threading.Lock()


def scatter(data, devices):
    """
    Split the data of a list of data at the first dimension by devices number,
    and send data respectively to devices.
    """
    def scatter_inner(data):
        if type(data) is list:
            return tuple(zip(*list(map(scatter_inner, data))))
        else:
            return A.scatter_parallel(data, devices)

    with A.variable_scope("data_scatter"):
        return scatter_inner(data)


def gather(data, output_device=0):
    def gather_inner(data):
        d = data[0]
        if is_tuple_or_list(d):
            return list(map(gather_inner, list(zip(*data))))
        else:
            return A.gather_parallel(data, output_device)

    return gather_inner(data)


def device(device_id):
    if A.backend() == A.TORCH:
        # remove the device type prefix and return
        return th.cuda.device(int(device_id[-1]))
    elif A.backend() == A.TF:
        return tf.device(device_id)
    else:
        raise ValueError("Backend is not supported.")


def broadcast(data, devices):
    """
    Broadcast variables in `data` to `devices`. The results are kept in the
    'globally', which can be retried using `get_variable`.
    """
    if A.torch_version < 0.4:
        out = th.nn.parallel._functions.Broadcast(devices)(*data)
    else:
        out = th.nn.parallel._functions.Broadcast.apply(devices, *data)
    # Deflatten the data.
    if len(data) > 0:
        out = [out[i:i + len(data)] for i in range(0, len(out), len(data))]

    if A.get_name(data[0]) is not None:
        device_num = len(devices)
        for i in range(1, device_num):
            with device("/gpu:{}".format(i)):
                for j, t in enumerate(out[i]):
                    name = "{}:{}".format(A.get_name(data[j]).split(':')[0], i)
                    A.cache_tensor(t, name)

    return out


def thread_run(func, *args, **kwargs):
    """
    Run code in a separate thread. Arbitrary positional arguments and keywords
    arguments are supported.

    Args:
        func: the function object to run

    Returns:
        results: list
            A list that contains returned results of the `func`. Empty if none.
        thread: threading.Thread
            The thread that is created for running.
    """
    # Give that new thread loses context manager, enforce it again.
    current_device = str(th.cuda.current_device())
    def _worker(func, results, *args, **kwargs):
        with device("/gpu:{}".format(current_device)):
            ret = func(*args, **kwargs)
            with _lock:
                results.append(ret)

    results = []
    args = [a for a in args]
    args.insert(0, results)
    args.insert(0, func)
    thread = threading.Thread(target=_worker,
                              args=tuple(args),
                              kwargs=kwargs)
    thread.start()
    return results, thread
