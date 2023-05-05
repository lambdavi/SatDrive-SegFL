import torch.nn as nn
import numpy as np

# Internal function
def split_list_numpy(lst, m):
    arr = np.array(lst)
    print(len(lst))
    split_sizes = np.random.randint(1, len(lst), size=m-1)
    split_sizes.sort()
    return np.split(arr, split_sizes)

# Internal function
def get_some(lst: np.ndarray, n: int):
    indeces =  np.random.randint(1, len(lst), size=n)
    return lst[indeces]

class HardNegativeMining(nn.Module):

    def __init__(self, perc=0.25):
        super().__init__()
        self.perc = perc

    def forward(self, loss, _):
        b = loss.shape[0]
        loss = loss.reshape(b, -1)
        p = loss.shape[1]
        tk = loss.topk(dim=1, k=int(self.perc * p))
        loss = tk[0].mean()
        return loss

class MeanReduction:
    def __call__(self, x, target):
        x = x[target != 255]
        return x.mean()
