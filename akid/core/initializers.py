"""
The module to holds variable initializers.

To get an initializer, call `get`::

    init_para={"stddev": 0.1},
    init = initializers.get(name, **init_para)

where the parameter needed by the initializer is passed in as a dict.

Note that all initializer return tensors/ndarray at least of dimension 1. To
get a tensor of dimension 0 --- that's to say to get a scalar ---, use
`ScalarInitializer` explicitly.
"""
from __future__ import division


from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import math

from ..ops.random_ops import msra_initializer
from .common import SEED
from ..utils import glog as log
from .. import backend as A


# Put all available initializers in a dict, so they can be retrieved by name.
inits = {}


def get(name, **kwargs):
    """
    Get the initializer by name. All necessary parameters should be passed in
    through keyword arguments.

    name: str
        The name of the initializer to use.
    """
    try:
        init = inits[name]
    except KeyError as e:
        if A.backend() == A.TORCH and name == "truncated_normal":
            init = inits["normal"]
            log.warn("Used normal initializer instead of truncated_normal. Truncated normal in not support in torch backend yet.")
        else:
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

    if name == "range_uniform" and A.backend() == A.TF:
        range = kwargs.pop("range")
        kwargs["minval"] = -range
        kwargs["maxval"] = range
    elif name == "uniform_unit_scaling" or name == "msra" and A.backend() == A.TF:
        if "factor" not in kwargs:
            log.info("Key factor is not found in `init_para`. Use 1")
            kwargs["factor"] = 1

    return init.get(**kwargs)


class InitializerRegistry(object):
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
        print(self.message)


class Initializer(object):
    """
    Top level class for initializers.
    """
    def __init__(self, seed=SEED):
        np.random.seed(seed)

        # A flag to indicate the initializer is native to akid. This flag is
        # necessary is due to the fact that sometimes circular import may
        # arise, so the issubclass check may not applicable.
        self.native = True


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


class FixedStdInitializer(Initializer):
    def __init__(self, stddev, **kwargs):
        super(FixedStdInitializer, self).__init__(**kwargs)
        self.std = stddev

    def __call__(self, shape):
        value = np.random.normal(loc=0, scale=self.std, size=shape)

        return value


class RangeUniformInitializer(Initializer):
    def __init__(self, low=0, high=1, **kwargs):
        super(RangeUniformInitializer, self).__init__(**kwargs)
        self.low = low
        self.high = high

    def __call__(self, shape):
        v = np.random.uniform(low=self.low, high=self.high, size=shape)
        v = v.astype(A.get_np_dtype())
        return v


class ConstantInitializer(Initializer):
    def __init__(self, value=0):
        super(ConstantInitializer, self).__init__()
        self.value = value

    def __call__(self, shape):
        v = np.ones(shape) * self.value
        v = v.astype(A.get_np_dtype())

        return v


class ScalarInitializer(Initializer):
    def __init__(self, value=0, **kwargs):
        super(ScalarInitializer, self).__init__(**kwargs)
        self.value = value

    def __call__(self, shape):
        # The shape parameter is unused. Just check to make sure it is not
        # accidentally (or intentionally) set wrong.
        if shape != 0:
            raise ValueError("The shape of `ScalarInitializer` should be 0. Got {}".format(shape))

        v = np.array(self.value)
        v = v.astype(A.get_np_dtype())

        return v


class AutoInitializer(Initializer):
    def __init__(self, normal_dist=False, fan_out=False, **kwargs):
        """
        The super class for all initializers that do not need parameters.

        Args:
            normal_dist: bool
                Use normal distribution or not.
        """
        super(AutoInitializer, self).__init__(**kwargs)
        self.normal_dist = normal_dist
        self.fan_out = fan_out


class TheOldInitializer(AutoInitializer):
    """
    The old way of doing initialization before even the (Glorot
    paper)(http://proceedings.mlr.press/v9/glorot10a.html).
    """
    def __call__(self, shape):
        n_in = self.compute_fan_in(shape)
        if self.normal_dist:
            if self.fan_out:
                n_out = self.compute_fan_out(shape)
                stdv = math.sqrt(1. / (n_in + n_out))
            else:
                stdv = math.sqrt(1. / 2 / n_in)
            value = np.random.normal(-stdv, stdv, shape)
        else:
            # NOTE: fan out is implemented for now.
            stdv = 1. / math.sqrt(n_in)
            value = np.random.uniform(-stdv, stdv, shape)

        return value




class XavierInitializer(AutoInitializer):
    """
    The initialization method from [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).
    """
    def __call__(self, shape):
        n_in = self.compute_fan_in(shape)
        n_out = self.compute_fan_out(shape)

        if self.normal_dist:
            std = np.sqrt(2./(n_in + n_out))
            v = np.random.normal(loc=0, scale=std, size=shape)
            v = v.astype(A.get_np_dtype())
        else:
            range = np.sqrt(6./(n_in + n_out))
            v = np.random.uniform(-range, range, size=shape)
            v = v.astype(A.get_np_dtype())

        return v


class MSRAInitializer(AutoInitializer):
    """
    Refer to `akid.ops.random_ops.msra_initializer`.
    """
    def __call__(self, shape):
        n_in = self.compute_fan_in(shape)
        if self.fan_out:
            n_out = self.compute_fan_out(shape)
            std = np.sqrt(4 / (n_in + n_out))
        else:
            std = np.sqrt(2 / n_in)
        v = np.random.normal(loc=0, scale=std, size=shape)
        return v


# CAUTION: if only one required field exists, use ("require",) instead of
# ("require") to make it a tuple.

InitializerRegistry("tensor",
            None,
            "Require fields: value (Tensor holds the initial values)",
            ("value",))
InitializerRegistry("xavier",
                    XavierInitializer,
                    XavierInitializer.__doc__)
InitializerRegistry("scalar",
                    ScalarInitializer,
                    ScalarInitializer.__doc__)

if A.backend() == A.TF:
    import tensorflow as tf
    InitializerRegistry("default",
                        tf.uniform_unit_scaling_initializer,
                        "Uniform unit scaling with factor 1.0/(3)**0.5",
                        # The strange factor here is to make variance
                        # `1/sqrt(dim)`. For the meaning of `dim`, see the doc
                        # of `tf.uniform_unit_scaling_initializer`.
                        default_paras={'factor': 1.0/(3)**0.5})
    InitializerRegistry("normal",
                tf.truncated_normal_initializer,
                "Required fields: stddev (Standard deviation). It is another name "
                "for the truncated normal initializer offered in tensorflow",
                ("stddev",))
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
    InitializerRegistry("normal",
                        FixedStdInitializer,
                        FixedStdInitializer.__doc__,
                        ("stddev",))
    InitializerRegistry("truncated_normal",
                        FixedStdInitializer,
                        FixedStdInitializer.__doc__,
                        ("stddev",))
    InitializerRegistry("msra",
                        MSRAInitializer,
                        MSRAInitializer.__doc__)
    InitializerRegistry("range_uniform",
                        RangeUniformInitializer,
                        RangeUniformInitializer.__doc__)
