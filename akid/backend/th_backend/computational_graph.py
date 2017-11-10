"""
PyTorch backend for akid.
"""
import os

import numpy as np
import torch as th
from torch.autograd import Variable

from .. import computational_graph as cg
from akid.utils import glog as log


# Maintain two hash table for looking up variables.
tensor_by_name = {}

# the number of checkpoints to save maximally
_max_checkpoint_count = 5
_checkpoint_count = 0
_checkpoint_name_queue = []


def get_variable(name=None, shape=None,
                 initializer=None, trainable=True,
                 shared=True):
    """
    Create or reuse variables.

    `shared`, and `trainable` are not used yet. They are legacy code from
    tensorflow, which may be useful when torch is used for distributed
    training.
    """
    if name:
        reuse = cg.get_variable_scope().reuse
        name = _get_name_with_scope(name)

        if reuse:
            # Return the variable if already allocated.
            if name in tensor_by_name:
                log.debug("Reuse variable {}".format(name))
                return tensor_by_name[name]
        elif name in tensor_by_name:
            name = _append_num(name)

    # Allocate the variable.
    if not callable(initializer):
        shape = None
        t = th.Tensor(initializer)
    else:
        t = th.Tensor(initializer(shape))

    if cg.use_cuda():
        t = t.cuda()

    t = Variable(t, requires_grad=trainable)

    if name:
        cache_tensor(t, name)

    return t


def retrieve_tensor(name):
    """
    Retrieve tensor by name.
    """
    return tensor_by_name[name]


def cache_tensor(tensor, name):
    tensor_by_name[name] = tensor
    # The reverse direction (get tensor by name) only works when it is a
    # variable. We occasionally cache numeric values as well, e.g. learning rate.
    if isinstance(tensor, Variable):
        tensor.name = name


def cache_tensor_auto_scope(tensor, name):
    """
    Cache `tensor` with `name`, whose scope information is prepended.
    """
    name = _get_name_with_scope(name)
    cache_tensor(tensor, name)


def save(path):
    """
    Save trainable variables and current step number to file. If a maximal
    number of checkpoints have been reached, the old ones will be deleted.

    Note that data concerning training are not all saved yet, e.g. the previous
    gradient that is used for momentum optimizer, which would make the
    continued training slightly different from the original one, but the guess
    is it does not matter much.
    """
    tensor_by_name['step'] = cg.get_step()
    name = path + "/checkpoint-{}".format(cg.get_step())
    th.save(tensor_by_name, name)
    src = 'checkpoint-{}'.format(cg.get_step())
    dst = path + "/checkpoint"
    if os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)

    _checkpoint_name_queue.append(name)
    if len(_checkpoint_name_queue) >= _max_checkpoint_count:
        name = _checkpoint_name_queue.pop(0)
        os.remove(name)


def restore(path):
    """
    Restore variables from checkpoints saved under `path`.

    The logic works like tensorflow --- the recovery is name based. As long as
    the code that builds the computational graph does not change, the variable
    of each block will be created with the same name as the saved one, so when
    the network is being built, the variables would be created with the
    restored values.
    """
    global tensor_by_name
    tensor_by_name = th.load(path + "/checkpoint")
    cg.set_step(tensor_by_name.pop('step'))
    # Put the name back. Seems torch's save does not save additional names
    for k in tensor_by_name:
        if type(tensor_by_name[k]) is Variable:
            tensor_by_name[k].name = k

    cg.get_variable_scope().reuse_variables()


def _get_name_with_scope(name):
    scope_name = cg.get_scope_name()
    if cg.use_cuda():
        name = '{}/{}:{}'.format(scope_name, name, int(th.cuda.current_device()))
    else:
        name = '{}/{}'.format(scope_name, name)
    return name


def remove_scope_from_name(name):
    """
    Remove scope and device information in the name.
    """
    if cg.use_cuda():
        name, _ = name.split(':')

    name = name.split('/')[-1]

    return name


def append_suffix(name, suffix):
    """
    Append suffix to the name of a tensor, above the level of devices.
    """
    name, device = name.split(':')
    name += '/' + suffix
    name += ':' + device
    return name


def cache_name_if_exist(func):
    def inner_func(*args, **kwargs):
        # Due to cooperative inheritance, name can only exist in kwargs.
        d = func(*args, **kwargs)
        if 'name' in kwargs and kwargs['name']:
            name = _get_name_with_scope(kwargs['name'])
            cache_tensor(d, name)
        return d
    return inner_func


def _append_num(name):
    """
    Similarly with tensorflow, we add numbers in names to distinguish variable
    with the same name.

    The format is to add an underscore and a number.

    TODO: no validation for bad naming now.
    """
    # The number needs to be added before device id.
    if name[-2] == ':':
        name, device_id = name.split(':')

    if name[-2] != '_':
        # means no number has been appended before
        name = "{}_1".format(name)
    else:
        # increase the No by one.
        parts = name.split('_')
        no = int(parts[-1])
        no += 1
        parts[-1] = str(no)
        name = ''
        for p in parts:
            name += p

    # Add back the device id.
    name += ":" + device_id

    return name


def get_name(v, with_device_id=True):
    """
    Given a tensor, return its name if available.

    The purpose of the function is to give a unique identifier that can be used
    to identify a Tensor.
    """
    if hasattr(v, "name"):
        if with_device_id:
            return v.name
        else:
            return v.name.split(':')[0]
    else:
        return None


def standardize_data_format(data, old_format):
    """
    Stardardize data to PyTorch format, which is channel first.

    Args:
        data: Tensor or numpy array.
            The input data.
        old_format: str
            A string describe the original format. For example, if converting
            from Tensorflow, it would be 'hwio' for parameter. See
            `SUPPORT_DATA_FORMAT` and `SUPPORT_PARA_FORMAT` for supported
            strings.
    """
    if old_format not in cg.SUPPORT_PARA_FORMAT \
       and old_format not in cg.SUPPORT_DATA_FORMAT:
        raise ValueError("The data format {} is not well specified.".format(old_format))

    if old_format in cg.SUPPORT_PARA_FORMAT:
        out_format = 'oihw'
    else:
        out_format = 'nchw'

    if type(data) == np.ndarray:
        return np.einsum('{}->{}'.format(old_format, out_format), data)
    else:
        raise ValueError("Type {} is not supported.".format(type(data)))


def init():
    """
    Does nothing. Just to be compatible with tensorflow backend.
    """
    pass


def run(op, *args, **kwargs):
    """Run an operation with arguments. For compatibility with Tensorflow"""
    return op(*args, **kwargs)


def close():
    """
    The same as `init()`.
    """
    global tensor_by_name
    tensor_by_name = {}


def eval(t):
    """
    Convert torch tensor to numpy array.
    """
    if type(t) is list or type(t) is tuple:
        return [eval(i) for i in t]

    # Convert to CPU anyway. May be redundant.
    t = t.cpu()

    if type(t) is Variable:
        v = t.data.numpy()
    else:
        v = t.numpy()

    if len(v) == 1:
        # Return the value directly if it is not an array.
        v = v[0]

    return v


@cache_name_if_exist
def Tensor(t, requires_grad=False, name=None):
    t = th.Tensor(t)
    if requires_grad:
        t =  Variable(t)
    return t


@cache_name_if_exist
def is_tensor(T):
    return type(T) is th.Tensor or type(T) is Variable


@cache_name_if_exist
def mul(a, b, name=None):
    return a * b


@cache_name_if_exist
def mean(v, name=None):
    return th.mean(v)


def get_shape(t):
    if type(t) is Variable:
        return list(t.data.shape)
    else:
        return list(t.shape)


def reshape(v, shape, name=None):
    return v.view(shape)


def convert_to_tensor(v):
    """
    Convert to tensor if necessary.
    """
    t = type(v)
    if t is Variable or t is th.Tensor:
        return v

    return th.Tensor(v)


@cache_name_if_exist
def add(a, b, name=None):
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    v = a + b

    return v

@cache_name_if_exist
def add_n(l, name=None):
    """
    Add all the tensors in the list.
    """
    acc = l[0].clone()
    for v in l[1:]:
        acc += v

    return acc


@cache_name_if_exist
def div(v, denominator, name=None):
    return th.div(v, denominator)


def scatter(data, devices):
    return th.nn.parallel._functions.Scatter(devices, dim=0)(data)


@cache_name_if_exist
def gather(data, output_device, name=None):
    return th.nn.parallel._functions.Gather(output_device, dim=0)(*data)
