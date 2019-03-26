"""
An extension of `multiprocessing`'s `Semaphore` to provide batch acquire of
semaphore.
"""
from __future__ import print_function

from __future__ import absolute_import
from multiprocessing import Semaphore as _Semaphore
from multiprocessing import Lock
from six.moves import range


class Semaphore(object):
    def __init__(self, count):
        self.semaphore = _Semaphore(count)
        self.in_lock = Lock()
        self.out_lock = Lock()

    def acquire(self, count):
        """
        Args:
            count: int
                The number of semaphores to acquire.
        """
        with self.in_lock:
            for i in range(0, count):
                self.semaphore.acquire()
                print ("Acquired semaphore.")

    def release(self, count):
        """
        Args:
            count: int
                The number of semaphores to acquire.
        """
        with self.out_lock:
            for i in range(0, count):
                self.semaphore.release()
                print ("Released semaphore.")
