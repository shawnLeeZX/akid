import sys
import os
import pdb
import functools
import traceback


def debug_on(*exceptions, on=True):
    if not exceptions:
        exceptions = (AssertionError, )
    def decorator(f):
        if not on:
            return f

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except exceptions:
                info = sys.exc_info()
                traceback.print_exception(*info)
                pdb.post_mortem(info[2])
        return wrapper
    return decorator
