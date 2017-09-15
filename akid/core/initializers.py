"""
The module to holds variable initializers.
"""
import numpy as np
import math

from ..ops import msra_initializer
from .common import SEED
from ..utils import glog as log
from .. import backend as A


# Put all available initializers in a dict, so they can be retrieved by name.
inits = {}


def get(name, **kwargs):
    """
    Get the initializer by name.

    name: str
        The name of the initializer to use.
    """
    try:
        init = inits[name]
    except KeyError as e:
        raise KeyError("`{}`. {} initialization method is not supported".format(e.message, name))

    # Make sure required fields exist.
    for p in init.required_fields:
        if p not in kwargs:
            raise KeyError("Required field {} not found in the provided"
                           " initialization parameters, `init_para`. Perhaps"
                           " you have some typos.".format(p))

    # Fill some default values if not given.
    if name != "constant" and "seed" not in kwargs:
        # Use default seed.
        kwargs["seed"] = SEED

    if name == "range_uniform":
        range = kwargs.pop("range")
        kwargs["minval"] = -range
        kwargs["maxval"] = range
    elif name == "uniform_unit_scaling" or name == "msra":
        if "factor" not in kwargs:
            log.info("Key factor is not found in `init_para`. Use 1")
            kwargs["factor"] = 1

    return init.get(**kwargs)


class InitializerRegistry(object):
    """
    Top level class for initializers.

    Call help to see documentation for each initializer.
    """
    def __init__(self,
                 name,
                 obj,
                 message=None,
                 required_fields=(),
                 default_paras={}):
        """
        Args:
            name: str:
                Name of the initializer. The unique identifier.
            obj: whether is an object, which is used for more
                sophisticated initialization.
            message: str
                Usage help message.
            default_paras: dict
                Default parameters if not specified.
            required_fields: tuple
                A tuple of str that specified required field for this initializer.
        """
        self.obj = obj
        self.message = message
        self.name = name
        self.default_paras = default_paras
        self.required_fields = required_fields
        inits[name] = self

    def get(self, **kwargs):
        if self.obj:
            # Create a callable instance that takes parameters and does the
            # actually initialization.
            for k in self.default_paras:
                if k not in kwargs:
                    kwargs[k] = self.default_paras[k]
            return self.obj(**kwargs)
        else:
            return kwargs["value"]

    def help(self):
        print self.message


class Initializer(object):
    def compute_fan_in(self, shape):
        if A.backend() == A.TF:
            # The weight layout of tf is HWIO.
            v = 1
            for i in shape[:-1]:
                v *= i
            return v
        elif A.backend() == A.TORCH:
            # The weight layout of tf is OIHW.
            v = 1
            for i in shape[1:]:
                v *= i
            return v

    def compute_fan_out(self, shape):
        if A.backend() == A.TF:
            # The weight layout of tf is HWIO.
            return shape[-1]
        elif A.backend() == A.TORCH:
            # The weight layout of tf is OIHW.
            return shape[0]


class TheOldInitializer(Initializer):
    """
    The old way of doing initialization before even the (Glorot
    paper)(http://proceedings.mlr.press/v9/glorot10a.html).
    """
    def __init__(self, seed):
        np.random.seed(seed)

    def __call__(self, shape):
        n = self.compute_fan_in(shape)
        stdv = 1. / math.sqrt(n)
        value = np.random.uniform(-stdv, stdv, shape)
        value = A.Tensor(value)

        return value


class ConstantInitializer(Initializer):
    def __init__(self, value=0):
        super(ConstantInitializer, self).__init__()
        self.value = value

    def __call__(self, shape):
        v = np.ones(shape) * self.value

        return v


# CAUTION: if only one required field exists, use ("require",) instead of
# ("require") to make it a tuple.

InitializerRegistry("tensor",
            None,
            "Require fields: value (Tensor holds the initial values)",
            ("value",))

if A.backend() == A.TF:
    # TODO: port the initializers that I coded myself to torch.
    import tensorflow as tf
    InitializerRegistry("default",
                        tf.uniform_unit_scaling_initializer,
                        "Uniform unit scaling with factor 1.0/(3)**0.5",
                        # The strange factor here is to make variance
                        # `1/sqrt(dim)`. For the meaning of `dim`, see the doc
                        # of `tf.uniform_unit_scaling_initializer`.
                        default_paras={'factor': 1.0/(3)**0.5})
    InitializerRegistry("truncated_normal",
                tf.truncated_normal_initializer,
                "Required fields: stddev (Standard deviation)",
                ("stddev",))
    InitializerRegistry("uniform_unit_scaling",
                tf.uniform_unit_scaling_initializer,
                "Optional fields: factor")
    InitializerRegistry("range_uniform",
                tf.random_uniform_initializer,
                "Require fields: range (Range of the uniform init)",
                ("range",))
    InitializerRegistry("msra",
                msra_initializer)
    InitializerRegistry("constant",
                tf.constant_initializer,
                "Require fields: value (initial value)",
                ("value",))
elif A.backend() == A.TORCH:
    InitializerRegistry("default",
                        TheOldInitializer,
                        TheOldInitializer.__doc__)
    InitializerRegistry("constant",
                        ConstantInitializer,
                        None,
                        ("value",))
