import tensorflow as tf

# Data format
# #########################################################################
SUPPORT_PARA_FORMAT = ['oihw', 'hwio']
SUPPORT_DATA_FORMAT = ['nchw', 'nhwc']


# Backends
# #########################################################################
TF = 'tensorflow'
TORCH = 'pytorch'
_backends = [TF, TORCH]


def available_backends():
    return _backends


variable_scope = tf.variable_scope
get_variable_scope = tf.get_variable_scope


def get_scope_name():
    return get_variable_scope().name


# Clock (global step)
# # #######################################################################
# The name assigned to the current training step. It is used to create
# universal access to current training step. This is similar to clock in a
# computer, but in distributed system, such synchronized clock should not
# exist, but it could understand as physical time, meaning how long this
# kid has been trained or born.
_step = 0


def get_step():
    return _step


def step():
    """
    Move to next step
    """
    global _step
    _step += 1
