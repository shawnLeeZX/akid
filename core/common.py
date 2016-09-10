"""
This module holds common stuffs of all, such as global constants.
"""
import tensorflow as tf
import inspect

# The name assigned to the current training step. It is used to create
# universal access to current training step. This is similar to clock in a
# computer, but in distributed system, such synchronized clock should not
# exist, but it could understand as physical time, meaning how long this
# kid has been trained or born.
GLOBAL_STEP = "step"
# The associated variable scope.
with tf.variable_scope("global") as global_var_scope:
    pass

# Graph collection names
TRAIN_SUMMARY_COLLECTION = "train_summary"
VALID_SUMMARY_COLLECTION = "val_summary"
FILTER_WEIGHT_COLLECTION = "filter_weight"
TRAINING_DYNAMICS_COLLECTION = "training_dynamics"
# Tough activation collection exists, nothing has been added to it yet. It is
# created for the sole purpose I do not want to delete the obsolete
# `get_activation` method in observer.
ACTIVATION_COLLECTION = "ACTIVATIONS"
# Auxilliary collections that may not always be used during training, such us
# eigenvalues of weight matrices (not included yet) and norm of filters are in
# this collection.
AUXILLIARY_SUMMARY_COLLECTION = "auxiliary_summary"
AUXILLIARY_STAT_COLLECTION = "auxiliary_stat"

# Global constants
# Shared seed for all involved randomness.
SEED = 66478  # Set to None for random seed.
# Manually named tag names.
LEARNING_RATE_TAG = "Learning Rate"
# tag suffixes
SPARSITY_SUMMARY_SUFFIX = "sparsity"


def init():
    """
    Kick start the world clock.
    """
    with tf.variable_scope(global_var_scope):
        global global_step_tensor
        global_step_tensor = tf.get_variable(
            name=GLOBAL_STEP,
            shape=[],
            initializer=tf.constant_initializer(0),
            trainable=False)


__all__ = [name for name, x in locals().items() if not inspect.ismodule(x)]
