import numpy as np

class EarlyStopper:
    """
        Early Stopper objects handles the early stopping technique.

        Args:
            `args`: Object containing the arguments for early stopping.\n
            `patience` (int): The number of epochs to wait before early stopping.\n
            `min_delta` (float): The minimum change in loss required to be considered as an improvement.
    """
    def __init__(self, args, patience=3, min_delta=0.05):
        self.args = args
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.prev_loss = np.inf

        if self.args.es:
            settings = self.args.es
            self.patience = int(settings[0])
            self.min_delta=settings[1]

    def early_stop(self, loss):
        """
        Checks if early stopping criteria are met based on the current loss value.

        Args:
            `loss` (float): The current loss value.

        Returns:
            bool: `True` if early stopping criteria are met, `False` otherwise.
        """
        if (self.prev_loss - loss) > self.min_delta:
            self.counter = 0
        elif abs(loss - self.prev_loss) < self.min_delta or\
            (self.prev_loss - loss) < 0:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        self.prev_loss = loss
        return False
    
    def reset_counter(self):
        """
        Resets the counter used for early stopping.

        Returns:
            None
        """
        self.counter = 0