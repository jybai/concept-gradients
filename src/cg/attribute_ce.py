import os
import argparse
import numpy as np
from tqdm.auto import tqdm, trange

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from .loader import load_dataset_and_model

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dset_name', type=str, choices=['cuba-unvoted'])
    parser.add_argument('task', metavar='T', type=str, choices=['x2y', 'x2c'])
    parser.add_argument('model_path', type=str, help="Should be the checkpoint file.")
    parser.add_argument('save_path', type=str, help="Path to save the attributed numpy file.")
    parser.add_argument('--arch_name', type=str, 
                        choices=['inception_v3', 'resnet50', 'vgg16_bn'])
    parser.add_argument('--layer', type=str, default=None)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--data_root_dir', type=str, default='/home/andrewbai/data/')
    
    parser.add_argument('--bsize', type=int, default=32,
                        help="Does not affect result. Only affects speed. Reduce when OOM.")
    return parser.parse_args()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_arguments()
    
    dl_train, dl_valid, dl_test, model = load_dataset_and_model(
        dset_name=args.dset_name, task=args.task, data_root_dir=args.data_root_dir, 
        use_all_data=False, arch_name=args.arch_name, bsize=args.bsize)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device).eval()
    
    attrs = []
    
    if args.split == 'val':
        dl = dl_valid
    elif args.split == 'test':
        dl = dl_test
    else:
        raise ValueError

    for x, c in tqdm(dl, leave=True):
        x = x.to(device)
        
        logits_pred = model(x)
        
        if args.task == 'x2y':
            attr = torch.softmax(logits_pred, dim=1)
        elif args.task == 'x2c':
            attr = torch.sigmoid(logits_pred)
        else:
            raise NotImplementedError
        
        attr = attr.detach().cpu().numpy()
        attrs.append(attr)

    attrs = np.concatenate(attrs, axis=0)
    
    np.save(args.save_path, attrs)

if __name__ == '__main__':
    main()
