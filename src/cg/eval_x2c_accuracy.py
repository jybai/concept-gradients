import os
import argparse
import numpy as np
import yaml
from tqdm.auto import tqdm, trange

import torch
import torch.nn as nn
from torchmetrics import Accuracy

from .loader import load_dataset_and_model

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dset_name', type=str)
    parser.add_argument('x2c_model_path', type=str, help="Should be the checkpoint file.")
    parser.add_argument('--x2c_arch_name', type=str, 
                        choices=['inception_v3', 'resnet50', 'dup-resnet50', 'vgg11_bn', 'vgg16_bn', None])
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--data_root_dir', type=str, default='/home/andrewbai/data/')
    parser.add_argument('--model_cfg_path', type=str, default=None)
    # parser.add_argument('--save_fname_suffix', type=str, default=None)
    
    parser.add_argument('--bsize', type=int, default=128,
                        help="Does not affect result. Only affects speed. Reduce when OOM.")
    return parser.parse_args()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_arguments()
    
    model_kwargs = {} 
    if args.model_cfg_path is not None:
        with open(args.model_cfg_path, 'r') as f:
            model_kwargs = yaml.safe_load(f)
    
    x2c_dl_train, x2c_dl_valid, x2c_dl_test, x2c_model = load_dataset_and_model(
        dset_name=args.dset_name, task='x2c', data_root_dir=args.data_root_dir, 
        use_all_data=False, arch_name=args.x2c_arch_name, bsize=args.bsize, model_kwargs=model_kwargs)
    x2c_model.load_state_dict(torch.load(args.x2c_model_path))
    x2c_model = x2c_model.to(device).eval()
    
    if args.split == 'train':
        dl = x2c_dl_train # shuffled and augmented
    elif args.split == 'val':
        dl = x2c_dl_valid
    elif args.split == 'test':
        dl = x2c_dl_test
    else:
        raise ValueError

    with torch.no_grad():
        acc = Accuracy().to(device)
        for x, c in tqdm(dl, leave=False):
            x = x.to(device)
            c = c.to(device).int()
            
            pred_c = torch.sigmoid(x2c_model(x))
            acc(pred_c, c)

        print(f"{acc.compute().item():.4f}")
    
if __name__ == '__main__':
    main()
