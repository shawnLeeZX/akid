def expand_kernel(ksize):
    if len(ksize) == 1:
        ksize = [1, ksize[0], ksize[0], 1]
    elif len(ksize) == 2:
        ksize = [1, ksize[0], ksize[1], 1]
    else:
        ksize = ksize

    return ksize
