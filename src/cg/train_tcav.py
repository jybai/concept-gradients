import os
import argparse
import numpy as np
from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .loader import load_dataset_and_model
from .train_utils import seed_everything, calculate_pos_weight
from .tcav import TCAV

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dset_name', metavar='D', type=str, choices=['cuba', 'cuba-unvoted'])
    parser.add_argument('x2y_model_path', type=str, help="Path to the x2y model checkpoint")
    parser.add_argument('--arch_name', type=str, choices=['inception_v3', 'resnet50', 'vgg16_bn'])
    parser.add_argument('--layers', type=str, nargs='+', help='The layers to calculate TCAV.')
    # training args
    parser.add_argument('--n_repeat', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3, help='search range: [1e-2, 1e-3]')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='search range: [1e-2, 1e-3]')
    parser.add_argument('--bsize', type=int, default=32)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--nepochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    # save args
    parser.add_argument('--data_root_dir', type=str, default='/home/andrewbai/data/')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('-f', '--force', action='store_true', help='Force rewrite CAVs.')
    return parser.parse_args()

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args = parse_arguments()
    
    if args.seed is not None:
        seed_everything(args.seed)
        print(f"Using random seed {args.seed}.")
        
    x2y_dl_train, x2y_dl_valid, x2y_dl_test, x2y_model = load_dataset_and_model(
        dset_name=args.dset_name, task='x2y', data_root_dir=args.data_root_dir, 
        use_all_data=False, arch_name=args.arch_name, bsize=args.bsize, return_dataloader=True)
    del x2y_dl_train, x2y_dl_valid, x2y_dl_test
    
    x2y_model.load_state_dict(torch.load(args.x2y_model_path))
    x2y_model = x2y_model.to(device).eval()
    
    x2c_dset_train, x2c_dset_valid, x2c_dset_test, x2c_model = load_dataset_and_model(
        dset_name=args.dset_name, task='x2c', data_root_dir=args.data_root_dir, 
        use_all_data=False, arch_name=args.arch_name, bsize=args.bsize, return_dataloader=False)
    del x2c_dset_test, x2c_model
    
    tcav = TCAV(x2y_model, args.layers, cache_dir=args.save_dir)
    
    x2c_dl_train = DataLoader(x2c_dset_train, batch_size=args.bsize, num_workers=8)
    pos_weight = calculate_pos_weight(x2c_dl_train)
    hparams = dict(task='classification', n_epochs=args.nepochs, patience=args.patience, batch_size=args.bsize,
                   lr=args.lr, weight_decay=args.weight_decay, pos_weight=pos_weight)
    
    tcav.generate_random_CAVs(x2c_dset_train, x2c_dset_valid, n_repeat=args.n_repeat, force_rewrite_cache=args.force)
    tcav.generate_CAVs(x2c_dset_train, x2c_dset_valid, n_repeat=args.n_repeat, hparams=hparams, force_rewrite_cache=args.force)
    
if __name__ == '__main__':
    main()
