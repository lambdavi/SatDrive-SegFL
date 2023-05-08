import torch.nn as nn
import numpy as np

# Internal function
def split_list_numpy(lst, m):
    n = len(lst)
    if m <= 0 or n <= 0:
        return []
    elif m >= n:
        return [lst[i:i+1] for i in range(n)]
    else:
        sizes = np.random.choice(n - m + 1, size=m-1, replace=False)
        sizes.sort()
        sizes = np.concatenate(([sizes[0]], sizes[1:] - sizes[:-1], [n - sizes[-1]]))
        return [lst[sum(sizes[:i]):sum(sizes[:i+1])] for i in range(m)]


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
