import abc


class UpdateBlock(object):
    """
    A block that supports an `update` method to update any parameters in the
    block.
    """
    def update(self, *args, **kwargs):
        return self._update(*args, **kwargs)

    @abc.abstractmethod
    def _update(self):
        raise NotImplementedError("The update method is not implemented.")
