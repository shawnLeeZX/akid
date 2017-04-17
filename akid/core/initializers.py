"""
The module to holds variable initializers.
"""
import tensorflow as tf
from ..ops import msra_initializer
from .common import SEED
from ..utils import glog as log


# Put all available initializers in a dict, so they can be retrieved by name.
inits = {}


def get(name, **kwargs):
    """
    Get the initializer by name.

    name: str
        The name of the initializer to use.
    """
    # Handle default initializer.
    if name is "default":
        # By default, we use the most preliminary initialization (for
        # conforming with torch).
        name = "uniform_unit_scaling"
        # The strange factor here is to make variance `1/sqrt(dim)`. For
        # the meaning of `dim`, see the doc of
        # `tf.uniform_unit_scaling_initializer`.
        kwargs["factor"] = 1.0/(3)**0.5

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
    if "seed" not in kwargs:
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


class Initializer(object):
    """
    Top level class for initializers.

    Call help to see documentation for each initializer.
    """
    def __init__(self, name, func, message=None, required_fields=()):
        """
        Args:
            name: str:
                Name of the initializer. The unique identifier.
            func: function symbol
            message: str
                Usage help message.
            required_fields: tuple
                A tuple of str that specified required field for this initializer.
        """
        self.func = func
        self.message = message
        self.name = name
        self.required_fields = required_fields
        inits[name] = self

    def get(self, **kwargs):
        if self.func:
            return self.func(**kwargs)
        else:
            return kwargs["value"]

    def help(self):
        print self.message


# CAUTION: if only one required field exists, use ("require",) instead of
# ("require") to make it a tuple.
Initializer("default",
            tf.uniform_unit_scaling_initializer,
            "Uniform unit scaling with factor 1.0/(3)**0.5")
Initializer("truncated_normal",
            tf.truncated_normal_initializer,
            "Required fields: stddev (Standard deviation)",
            ("stddev",))
Initializer("uniform_unit_scaling",
            tf.uniform_unit_scaling_initializer,
            "Optional fields: factor")
Initializer("range_uniform",
            tf.random_uniform_initializer,
            "Require fields: range (Range of the uniform init)",
            ("range",))
Initializer("msra",
            msra_initializer)
Initializer("tensor",
            None,
            "Require fields: value (Tensor holds the initial values)",
            ("value",))
