class UpdateBlock(object):
    """
    A block that supports an `update` method to update any parameters in the
    block.
    """
    def update(self, *args, **kwargs):
        return self._update(*args, **kwargs)

    def _update(self):
        raise NotImplementedError("The update method is not implemented."
                                  "The program should not reach here.")

    def on_update(self, *args, **kwargs):
        """
        Hook to be called after parameter update.
        """
        pass
