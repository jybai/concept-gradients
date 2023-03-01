import os
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torchmetrics

from .dataset import CUBC2YDataset

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('attr_npy_path', type=str)
    parser.add_argument('dset_name', type=str, choices=['cuba-unvoted'])
    # optional
    # parser.add_argument('recall_avg', type=str, default='samples', choices=['samples', 'macro', 'micro'])
    parser.add_argument('--data_split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--data_root_dir', type=str, default='/home/andrewbai/data/')
    parser.add_argument('--bsize', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--ks', type=str, nargs='+', help='List of `k`s to evaluate recall@k.', default=[30, 40, 50])
    return parser.parse_args()

def main():
    
    args = parse_arguments()
    
    if args.dset_name == 'cuba-unvoted':
        gt_dset = CUBC2YDataset(root_dir=args.data_root_dir, split=args.data_split, 
                                voted_concept_labels=False)
    else:
        raise NotImplementedError
    gt_dl = DataLoader(gt_dset, batch_size=args.bsize, num_workers=args.num_workers)
    
    attr_dset = TensorDataset(torch.from_numpy(np.load(args.attr_npy_path)).float())
    attr_dl = DataLoader(attr_dset, batch_size=args.bsize, num_workers=args.num_workers)
    
    assert len(gt_dset) == len(attr_dset)
    assert len(gt_dl) == len(attr_dl)
    
    for k in args.ks:
        
        rk = torchmetrics.Recall(top_k=k, average='samples')
        for (cs, ys), (attrs,) in zip(gt_dl, attr_dl):
            rk(attrs, cs.int())

        # https://support.google.com/a/users/answer/9308645?hl=en
        print(f"{rk.compute().item():.4f}", end='\t')
    print("")

if __name__ == '__main__':
    main()