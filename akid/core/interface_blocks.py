class UpdateBlock(object):
    """
    A block that supports an `update` method to update any parameters in the
    block.
    """
    def __init__(self, call_on_update=False, **kwargs):
        super(UpdateBlock, self).__init__(**kwargs)
        self.call_on_update = call_on_update

    def update(self, *args, **kwargs):
        d = self._update(*args, **kwargs)

        if self.call_on_update:
            self._on_update(*args, **kwargs)

        return d

    def _update(self):
        raise NotImplementedError("The update method is not implemented."
                                  "The program should not reach here.")

    def _on_update(self, *args, **kwargs):
        """
        Hook to be called after parameter update.
        """
        pass
