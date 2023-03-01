import os
import argparse
import numpy as np
from tqdm.auto import tqdm, trange
from sklearn.metrics import f1_score, auc
from multiprocessing import Pool

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
    parser.add_argument('--reduce', type=str, default='signed_avg', choices=['signed_avg', 'avg', 'ce_necessary',
                                                                             'ce_sufficient'])
    parser.add_argument('--data_split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--data_root_dir', type=str, default='/home/andrewbai/data/')
    parser.add_argument('--bsize', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--ks', type=str, nargs='+', help='List of `k`s to evaluate recall@k.', 
                        default=[30, 40, 50])
    # ce args
    parser.add_argument('--x2y_npy_path', type=str, default=None, help="Only used when `reduce=ce_necessary` or `reduce=ce_sufficient`")
    return parser.parse_args()

def main():
    
    args = parse_arguments()
    
    if args.dset_name == 'cuba-unvoted':
        gt_dset = CUBC2YDataset(root_dir=args.data_root_dir, split=args.data_split, 
                                voted_concept_labels=False)
        N_CLASSES = 200
        N_CONCEPTS = 112
    else:
        raise NotImplementedError
    
    attr_dset = TensorDataset(torch.from_numpy(np.load(args.attr_npy_path)).float())
    assert len(gt_dset) == len(attr_dset)
        
    # extract the per-class concept gt labels and per-class concept predicted labels.
    class_to_concept_labels_gt = {}
    class_to_concept_labels_pred = {}
    
    for (c, y), (pred_c,) in zip(gt_dset, attr_dset):
        y = y.int().item() # if use torch.tensor, same value tensor would be considered as different items.
        
        if y not in class_to_concept_labels_gt:
            class_to_concept_labels_gt[y] = [c]
        else:
            class_to_concept_labels_gt[y].append(c)
        
        if y not in class_to_concept_labels_pred:
            class_to_concept_labels_pred[y] = [pred_c]
        else:
            class_to_concept_labels_pred[y].append(pred_c)
    assert len(class_to_concept_labels_gt) == N_CLASSES
    
    # majority vote gt concept labels
    for k in class_to_concept_labels_gt.keys():
        v = class_to_concept_labels_gt[k]
        v = (torch.stack(v, 0).mean(0) > 0.5).int()
        class_to_concept_labels_gt[k] = v
    
    assert all([len(v) == N_CONCEPTS for v in class_to_concept_labels_gt.values()])
    
    # reduce pred concept labels
    if args.reduce == 'signed_avg':
        for k in class_to_concept_labels_pred.keys():
            v = class_to_concept_labels_pred[k]
            v = (torch.stack(v, 0) > 0).float().mean(0)
            class_to_concept_labels_pred[k] = v
    elif args.reduce == 'avg':
        for k in class_to_concept_labels_pred.keys():
            v = class_to_concept_labels_pred[k]
            v = torch.stack(v, 0).mean(0)
            class_to_concept_labels_pred[k] = v
    elif args.reduce.startswith('ce'):
        x2y_dset = TensorDataset(torch.from_numpy(np.load(args.x2y_npy_path)).float())
        assert len(x2y_dset) == len(gt_dset)
        
        class_to_class_pred = {}
        for (c, y), (pred_y,) in zip(gt_dset, x2y_dset):
            y = y.int().item() # if use torch.tensor, same value tensor would be considered as different items.

            if y not in class_to_class_pred:
                class_to_class_pred[y] = [pred_y]
            else:
                class_to_class_pred[y].append(pred_y)
        assert len(class_to_class_pred) == N_CLASSES
        
        for k in class_to_class_pred.keys():
            v = torch.stack(class_to_class_pred[k], 0)
            assert v.shape[1] == N_CLASSES
            class_to_class_pred[k] = v[:, k]
        
        for k in tqdm(class_to_concept_labels_pred.keys(), leave=False):
            pred_c = torch.stack(class_to_concept_labels_pred[k], 0).detach().cpu().numpy()
            pred_y = class_to_class_pred[k].detach().cpu().numpy()
            
            aucs = []
            n = 50
            ts = np.linspace(0, 1, n)
            
            for ci in trange(N_CONCEPTS, leave=False):
                f1s = []
                for t in ts:
                    pred_c_ = pred_c[:, ci] >= t
                    pred_y_ = pred_y >= t
                    if args.reduce == 'ce_necessary':
                        f1 = f1_score(pred_c_, pred_y_, zero_division=0)
                    elif args.reduce == 'ce_sufficient':
                        f1 = f1_score(pred_y_, pred_c_, zero_division=0)
                    else:
                        raise NotImplementedError
                    f1s.append(f1)
                auc_ = auc(ts, f1s)
                aucs.append(auc_)
            class_to_concept_labels_pred[k] = torch.tensor(aucs)
    else:
        raise NotImplementedError
    assert all([len(v) == N_CONCEPTS for v in class_to_concept_labels_pred.values()])
    
    for k in args.ks:
        
        rk = torchmetrics.Recall(top_k=k, average='samples')
        for k in class_to_concept_labels_gt.keys():
            concept_labels_pred = class_to_concept_labels_pred[k].unsqueeze(0)
            concept_labels_gt = class_to_concept_labels_gt[k].unsqueeze(0).bool()
            rk(concept_labels_pred, concept_labels_gt)

        # https://support.google.com/a/users/answer/9308645?hl=en
        print(f"{rk.compute().item():.4f}", end='\t')
    print("")

if __name__ == '__main__':
    main()