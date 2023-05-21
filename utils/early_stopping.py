import numpy as np

class EarlyStopper:
    def __init__(self, args, patience=3, min_delta=0.05):
        self.args = args
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.prev_loss = np.inf

        if self.args.es:
            print("found arg es")
            settings = self.args.es
            print(settings)
            self.patience = int(settings[0])
            self.min_delta=settings[1]

    def early_stop(self, loss):
        if (loss - self.prev_loss) > self.min_delta:
            self.counter = 0
        elif abs(loss - self.prev_loss) < self.min_delta or\
            (loss - self.prev_loss) < 0:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        self.prev_loss = loss
        return False