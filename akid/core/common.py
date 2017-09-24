"""
This module holds common stuffs of all, such as global constants.
"""
import inspect

# Graph collection names
# #########################################################################
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
# #########################################################################
# Shared seed for all involved randomness.
SEED = 66478  # Set to None for random seed.
# Manually named tag names.
LEARNING_RATE_TAG = "Learning Rate"
# tag suffixes
SPARSITY_SUMMARY_SUFFIX = "sparsity"

__all__ = [name for name, x in locals().items() if not inspect.ismodule(x)]
