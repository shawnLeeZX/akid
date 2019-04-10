from akid import backend as A

@A.cache_name_if_exist
def hinge_loss(y, t, weights=None, name=None):
    """
    Given 1D, or 2D Tensor y, t, calculate the (weighted) hinge loss as follows
    for each element in the tensor, and average them. That means, that it
    supports binary classification, and multi-label binary classification.

    .. math::
        l(y, t) = weights * \max{0, 1 - t * y}

    TODO: implement margin in the loss.
    """
    loss = 1 - t * y
    loss[loss < 0] = 0
    if weights is not None:
        loss *= weights
    loss = A.mean(loss)
    return loss
