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


@A.cache_name_if_exist
def tripod_hinge_loss(y, t, weights=None, signal_balancing=False, name=None):
    """
    Given 1D, or 2D Tensor y, t, calculate the (weighted) hinge loss as follows
    for each element in the tensor, and average them. That means, that it
    supports binary classification, and multi-label binary classification.

    if t = -1, or 1
    .. math::
        l(y, t) = weights * \max{0, 1 - t * y}

    if t = 0
    .. math::
        l(y, t) = weights * |y|

    TODO: implement margin in the loss.
    Args:
        signal_balancing: bool
            If True, will average the loss of different labels first, and then
            average the mean loss of different labels. It puts equal weights on
            different label, thus balancing the different label values.
    """
    middle_idx = A.cast(t, A.uint8) == 0
    middle_score = y[middle_idx]

    end_idx = ~middle_idx
    end_score = y[end_idx]
    end_label = t[end_idx]

    end_score_loss = 1 - end_label * end_score
    end_score_loss[end_score_loss < 0] = 0
    middle_score_loss = A.abs(middle_score)

    # TODO: Deal with weight later.
    # if weights is not None:
    #     loss *= weights

    if signal_balancing:
        loss = (A.mean(end_score_loss) + A.mean(middle_score_loss)) / 2
    else:
        loss = A.cat([end_score_loss, middle_score_loss])
        loss = A.mean(loss)
    return loss


@A.cache_name_if_exist
def hinge_ranking_loss(x1, x2, margin, hard_sample_mining=False, name=None):
    """
    Hinge loss adopted for ranking. To understand the loss, recall that hinge
    loss is an almost stair-case style approximation to 0-1 loss, and 0-1 loss
    can be understood as a relationship between two states. For ranking, we
    need more than one state to represent the ranking order. More details can
    be found at *Ranking with Ordered Weighted Pairwise Classification*, the
    paper that motivates the loss.

    Currently in this implementation, only ranking order between positive and
    negative samples are characterized. The loss is of the formula

    .. math::
        l(x1, x2) = \max{0, margin - (x1 - x2)}

    where x1 is the positive sample, and x2 is the negative sample.

    Intuitively, compared with hinge loss, it simply enforces a margin between
    x1 and x2, instead of dictating that the score should be larger than an
    absolute value as in the hinge loss, :meth:`akid.nn.losses.hinge_loss`.

    Args:
        hard_sample_mining: bool
            Whether to use hard sample mining in the loss. It would use the
            pair (x1, x2), where x1 < x2.
    """
    delta = (x1 - x2)
    if hard_sample_mining:
        delta = delta[delta - margin < 0]
    loss = margin - delta
    loss[loss < 0] = 0
    loss = A.mean(loss)
    return loss
