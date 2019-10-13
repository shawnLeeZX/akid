"""
PyTorch backend for akid.
"""
from __future__ import absolute_import
import os
import sys
import inspect

import numpy as np
import torch as th
from torch.autograd import Variable

from .. import computational_graph as cg
from akid.utils import glog

float32 = th.float32
uint8 = th.uint8
to_th_dtype = {
    int: th.long,
    float: th.float
}

DATA_FORMAT = "CHW"

# Maintain two hash table for looking up variables.
tensor_by_name = {}

# the number of checkpoints to save maximally
_max_checkpoint_count = 5
_checkpoint_count = 0
_checkpoint_name_queue = []

# PyTorch version
torch_version = float(th.__version__[0:3])
assert type(torch_version) is float, "Torch version number extraction failed. Version str: {}. Extracted value: {}".format(th.__version__, float(th.__version__[0:3]))

# Device to use. Only useful for torch newer than or equal to 0.4.
device = "cuda:0" if cg.use_cuda() else "cpu"


def remove_variable_contains_str(str):
    name_list = [name for name in tensor_by_name]
    for n in name_list:
        # Clean up the monitoring tensors created to save memory.
        if str in n:
            tensor_by_name.pop(n)


def get_variable(name=None, shape=None,
                 initializer=None, trainable=True,
                 shared=True):
    """
    Create or reuse variables. Compared with directly get a tensor from
    `Tensor`, it cached a name for the variable returned. It makes the variable
    available to be retrieved by name.

    `shared` is not used yet. They are legacy code from tensorflow, which may
    be useful when torch is used for distributed training.
    """
    if name:
        reuse = cg.get_variable_scope().reuse
        name = _get_name_with_scope(name)

        if reuse:
            # Return the variable if already allocated.
            if name in tensor_by_name:
                glog.debug("Reuse variable {}".format(name))
                return tensor_by_name[name]
        elif name in tensor_by_name:
            name = _append_num(name)

    # Allocate the variable.
    if not callable(initializer):
        shape = None
        if type(initializer) is th.Tensor:
            t = initializer
            t.requires_grad = True
        else:
            t = Tensor(initializer, requires_grad=trainable)
    else:
        t = Tensor(initializer(shape), requires_grad=trainable)

    if name:
        cache_tensor(t, name)

    t._is_variable = True

    glog.debug("Created new variable {}".format(name))

    return t


def get_all_variables():
    all_variables = {}
    for t in tensor_by_name:
        n = tensor_by_name[t]
        if type(n) is th.Tensor and n.is_leaf:
            all_variables[t] = n

    return all_variables


def retrieve_tensor(name):
    """
    Retrieve tensor by name.
    """
    return tensor_by_name[name]


def cache_tensor(tensor, name):
    tensor_by_name[name] = tensor

    # The reverse direction (get tensor by name) only works when it is a
    # variable. We occasionally cache numeric values as well, e.g. learning rate.

    if isinstance(tensor, cg.NamedValue):
        tensor.name = name
    # The instance below is to ensure we are dealing with an object, so
    # attributes could be added.
    elif torch_version < 0.4:
        if isinstance(tensor, Variable):
            tensor.name_ = name
    else:
        if isinstance(tensor, th.Tensor):
            tensor.name_ = name


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
    if not os.path.isdir(path):
        os.mkdir(path)
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
        if os.path.exists(name):
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
        # Remove GPU device id if using CPU
        if not cg.use_cuda():
            if is_variable(tensor_by_name[k]):
                v = tensor_by_name.pop(k)
                k = remove_gpu_device_id(k)
                if torch_version < 0.4:
                    v = Variable(v.data.cpu(), requires_grad=True) if v.is_cuda else v
                else:
                    v = v.to("cpu") if v.is_cuda else v
                tensor_by_name[k] = v
        if is_variable(tensor_by_name[k]):
            tensor_by_name[k].name_ = k


    cg.get_variable_scope().reuse_variables()


def remove_gpu_device_id(name):
    if ':' in name:
        name = name.split(':')[0]

    return name


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
    if ':' in name:
        name, _ = name.split(':')

    name = name.split('/')[-1]

    return name


def append_suffix(name, suffix):
    """
    Append suffix to the name of a tensor, above the level of devices.
    """
    if ':' in name:
        name, device = name.split(':')
        name += '/' + suffix
        name += ':' + device
    else:
        # The name is a name of a tensor on CPU, add suffix directly.
        name += '/' + suffix
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
    if cg.use_cuda():
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

    if cg.use_cuda():
        # Add back the device id.
        name += ":" + device_id

    return name


def get_name(v, with_device_id=True, no_scope=False):
    """
    Given an object, return its name if available.

    The purpose of the function is to give a unique identifier that can be used
    to identify a computed result.
    """
    if type(v) is th.Tensor:
        if hasattr(v, "name_"):
            if with_device_id:
                name = v.name_
            else:
                name = v.name_.split(':')[0]
        else:
            name = None
    elif isinstance(v, cg.NamedValue):
        name = v.name
    else:
        raise TypeError("Type {} not supported".format(type(v)))

    if no_scope and name is not None:
        name = remove_scope_from_name(name)

    return name


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

    if isinstance(t, np.ndarray) or np.isscalar(t):
        return t

    if type(t) is cg.NamedTensorTuple:
        return cg.NamedTensorTuple(t.name, [eval(v) for v in t])

    if isinstance(t, cg.NamedScalar):
        return t

    # Convert to CPU anyway. May be redundant.
    t = t.cpu()

    if torch_version < 0.4:
        if type(t) is Variable:
            v = t.data.numpy()
        else:
            v = t.numpy()
    else:
        v = t.detach().numpy()

    if torch_version < 0.4:
        if len(v) == 1:
            # Return the value directly if it is not an array.
            v = v[0]

    return v


@cache_name_if_exist
def Tensor(t, requires_grad=False, name=None):
    """
    Refer to `Tensor` in Tensorflow backend for its semantics.

    We always copy data when creating new tensors given data `t`.
    """
    type_t = type(t)
    if type_t is np.ndarray:
        if t.dtype == np.int:
            t = th.from_numpy(t)
        else:
            t = th.from_numpy(t.astype(np.float32))
    elif np.isscalar(t):
        t = th.tensor(float(t))
    elif type_t is list or type_t is tuple:
        t_element = t[0]
        element_type = type(t_element)
        # Get the type of the first element in the iterable.
        while element_type is list or element_type is tuple:
            t_element = t_element[0]
            element_type = type(t_element)
        if element_type  == th.Tensor:
            t = th.tensor(t)
        else:
            t = th.tensor(t, dtype=to_th_dtype[element_type])
    elif type_t is th.Tensor:
        pass
    else:
        raise TypeError("Unknown type {} to create tensor.".format(type(t)))

    if cg.use_cuda():
        t = t.cuda()

    if requires_grad:
        if torch_version < 0.4:
            t =  Variable(t)
        else:
            t.requires_grad_(requires_grad)


    return t


@cache_name_if_exist
def is_tensor(T):
    return type(T) is th.Tensor or type(T) is Variable


def is_variable(T):
    if torch_version < 0.4:
        return type(T) is Variable
    else:
        if type(T) is not th.Tensor:
            return False
        elif hasattr(T, "_is_variable"):
            return T._is_variable
        # When the variable is loaded from file by torch.load, it does not have
        # the `_is_variable` attribute. In such a case, we check if the
        # variable is a leaf node.
        else:
            return T.is_leaf


@cache_name_if_exist
def mul(a, b, name=None):
    return a * b


@cache_name_if_exist
def mean(v, dim=None, keep_dim=False, name=None):
    if v.dtype != th.float:
        v = v.type(th.float)
    if dim is None:
        return th.mean(v)
    else:
        return th.mean(v, dim, keepdim=keep_dim)


@cache_name_if_exist
def exp(v, name=None):
    return th.exp(v)


@cache_name_if_exist
def abs(v, name=None):
    return th.abs(v)


def std(v, dim=None, keep_dim=False, name=None):
    if dim is None:
        return th.std(v)
    else:
        return th.std(v, dim, keepdim=keep_dim)


def var(v, dim=None, keep_dim=False, name=None):
    if dim is None:
        return th.var(v)
    else:
        return th.var(v, dim, keepdim=keep_dim)


@cache_name_if_exist
def max(v, name=None):
    return th.max(v)


@cache_name_if_exist
def log(v, name=None):
    return th.log(v)


@cache_name_if_exist
def linspace(start, end, space, name=None):
    return th.linspace(start, end, space)


def get_shape(t):
    if isinstance(t, cg.NamedScalar):
        return 0

    if isinstance(t, cg.NamedTensorTuple):
        return [get_shape(v) for v in t]

    if torch_version < 0.4:
        if type(t) is Variable:
            return list(t.data.shape)
        else:
            return list(t.shape)
    else:
        return list(t.shape)


def reshape(v, shape, name=None):
    return v.view(shape)


def split(v, chunk_size, dim=0, name=None):
    return v.split(chunk_size, dim=dim)


def convert_to_tensor(v):
    """
    Convert to tensor if necessary.
    """
    t = type(v)
    if t is Variable or t is th.Tensor:
        return v

    return th.tensor(v)


@cache_name_if_exist
def sqrt(v, name=None):
    return th.sqrt(v)


@cache_name_if_exist
def add(a, b, name=None):
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


@cache_name_if_exist
def cat(v, name=None):
    return th.cat(v)


@cache_name_if_exist
def gather(input, dim, index, name=None):
    return th.gather(input, dim, index)


def scatter_parallel(data, devices):
    if torch_version < 0.4:
        return th.nn.parallel._functions.Scatter(devices, dim=0)(data)
    else:
        return th.nn.parallel._functions.Scatter.apply(devices, None, 0, data)


@cache_name_if_exist
def gather_parallel(data, output_device, name=None):
    if torch_version < 0.4:
        return th.nn.parallel._functions.Gather(output_device, dim=0)(*data)
    else:
        return th.nn.parallel._functions.Gather.apply(output_device, 0, *data)


@cache_name_if_exist
def expand_dims(x, axis, name=None):
    return th.unsqueeze(x, dim=axis)


@cache_name_if_exist
def transpose(x, name=None):
    return x.t()


@cache_name_if_exist
def squeeze(x, dim=None, name=None):
    if dim is not None:
        return th.squeeze(x, dim=dim)
    else:
        return th.squeeze(x)


@cache_name_if_exist
def cast(x, dtype, name=None):
    return x.type(dtype)


@cache_name_if_exist
def stack(x, name=None):
    return th.stack(x)


@cache_name_if_exist
def value(x, name=None):
    """
    Get the value of a Tensor.
    """
    return x.detach().clone()


@cache_name_if_exist
def zeros(shape, name=None):
    return th.zeros(shape, device=device)


@cache_name_if_exist
def ones(shape, name=None):
    return th.ones(shape, device=device)


@cache_name_if_exist
def eye(n, name=None):
    return th.eye(n, device=device)


@cache_name_if_exist
def range(start, end, name=None):
    return th.arange(start, end)


@cache_name_if_exist
def symeig(H):
    return th.symeig(H, eigenvectors=True)


@cache_name_if_exist
def randn(shape, name=None):
    return th.randn(*shape, device=device)


@cache_name_if_exist
def randint(low, high, size=1, name=None):
    return th.randint(low=low, high=high, size=size, device=device)


@cache_name_if_exist
def pad(x, paddings, name=None):
    # Reverse the padding para, since torch uses it backwardly.
    paddings.reverse()
    padding_flattened = []
    for i, v in enumerate(paddings):
        padding_flattened.extend(v)
    padding_flattened = tuple(padding_flattened)
    return th.nn.functional.pad(x, padding_flattened)


@cache_name_if_exist
def matmul(a, b, name=None):
    return th.matmul(a, b)


attributes = dir(sys.modules[__name__])
public_attributes = []
for a in attributes:
    if a[0] == '_' or inspect.ismodule(getattr(sys.modules[__name__], a)):
        continue
    public_attributes.append(a)
__all__ = public_attributes
