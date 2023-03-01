import torch
import random
import numpy as np

def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
def calculate_pos_weight(dl):
    pos_cnt = None
    cnt = 0
    with torch.no_grad():
        for xs, ys in dl:
            if pos_cnt is None:
                pos_cnt = ys.sum(0).clone()
            else:
                pos_cnt += ys.sum(0).clone()
            cnt += ys.shape[0]
    
    neg_cnt = cnt - pos_cnt
    pos_weight = neg_cnt / pos_cnt
    
    return pos_weight
