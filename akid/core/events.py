"""
Event to pass around to communicate events happening in blocks.
"""
class DataPrefetchThreadsDeadEvent(Exception):
    pass


class DataPrefetchProcessesDeadEvent(Exception):
    pass


class EarlyStoppingEvent(Exception):
    def __init__(self, val_loss, val_evals):
        """
        When the training early stops, it should save current validation loss
        and evaluations.
        """
        self.val_loss = val_loss
        self.val_evals = val_evals


class EpochCompletedEvent(Exception):
    pass


class DoneEvent(Exception):
    pass
