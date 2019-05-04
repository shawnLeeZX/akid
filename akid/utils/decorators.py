def coroutine(func):
    def start(*args, **kwargs):
        cr = func(*args, **kwargs)
        ret = cr.__next__()
        return cr, ret
    return start
