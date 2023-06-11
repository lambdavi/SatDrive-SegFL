import torch.nn as nn
import numpy as np

def get_save_string(args, is_source=False):
    prefix=args.dataset + "_" + args.model + "_"
    mode=""
    if args.centr:
        mode = "centr"
    elif args.fda:
        if is_source:
            mode="fda_source"
        else:
            mode="fda"
    else:
        mode="distr"

    return prefix+mode

# Internal function
def split_list_random(lst, m):
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

def split_list_balanced(lst, m):
    n = len(lst)
    if m <= 0 or n <= 0:
        return []
    elif m >= n:
        return [lst[i:i+1] for i in range(n)]
    else:
        # Compute the size of each sublist
        sublist_size = n // m
        sizes = np.full(m, sublist_size, dtype=np.int32)
        remainder = n % m
        if remainder != 0:
            # Add the remaining elements to the first few sublists
            sizes[:remainder] += 1

        # Create the sublists by slicing the input list
        sublists = [lst[sum(sizes[:i]):sum(sizes[:i+1])] for i in range(m)]
        return sublists

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
