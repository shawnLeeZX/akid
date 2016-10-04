"""
An extension of `multiprocessing`'s `Semaphore` to provide batch acquire of
semaphore.
"""

from multiprocessing import Semaphore as _Semaphore


class Semaphore(object):
    def __init__(self, count):
        self.semaphore = _Semaphore(count)

    def acquire(self, count):
        """
        Args:
            count: int
                The number of semaphores to acquire.
        """
        for i in xrange(0, count):
            self.semaphore.acquire()

    def release(self, count):
        """
        Args:
            count: int
                The number of semaphores to acquire.
        """
        for i in xrange(0, count):
            self.semaphore.release()
