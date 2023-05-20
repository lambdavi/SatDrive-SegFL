import numpy as np

class EarlyStopper:
    def __init__(self, patience=3, min_delta=0.05):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if (validation_loss - self.min_validation_loss) > self.min_delta:
            self.counter = 0
        elif abs(validation_loss - self.min_validation_loss) < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        self.min_validation_loss = validation_loss
        return False