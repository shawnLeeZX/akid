from akid import backend as A

@A.cache_name_if_exist
def binary_accuracy(x, y, hinge_loss_label=False, name=None):
    """
    If `hinge_loss_label` is true, the function assumes that elements in the
    label `y` is given as -1, or 1, and 0 is used as the threshold to tell
    apart classes. Otherwise, the label is assume to be given
    in 1, or 0, and 0.5 is used as the threshold.
    """
    x = A.value(x)

    if hinge_loss_label:
        x[x > 0] = 1
        x[x < 0] = -1
    else:
        x[x >= 0.5] = 1
        x[x < 0.5] = 0

    acc = A.mean(A.cast(x == y, A.float32))
    return acc
