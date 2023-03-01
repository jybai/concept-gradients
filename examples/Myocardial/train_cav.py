import os
import argparse
import numpy as np
from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cg.train_utils import seed_everything, calculate_pos_weight
from cg.tcav import TCAV

from data import MIComplications
from model import ClassifierHead

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('x2y_model_path', type=str, help="Path to the x2y model checkpoint")
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
        
    x2c_dset = MIComplications(args.data_root_dir, return_attribute=True)
    x2c_dl = DataLoader(x2c_dset, batch_size=args.bsize, num_workers=8)
        
    x2y_model = ClassifierHead(x2c_dset.X.shape[1], 1, squeeze_final=True)
    x2y_model.load_state_dict(torch.load(args.x2y_model_path))
    x2y_model = x2y_model.to(device).eval()
    
    tcav = TCAV(x2y_model, args.layers, cache_dir=args.save_dir)
    
    x2c_dl_train = DataLoader(x2c_dset, batch_size=args.bsize, num_workers=8)
    pos_weight = calculate_pos_weight(x2c_dl)
    hparams = dict(task='classification', n_epochs=args.nepochs, patience=args.patience, batch_size=args.bsize,
                   lr=args.lr, weight_decay=args.weight_decay, pos_weight=pos_weight)
    
    tcav.generate_random_CAVs(x2c_dset, x2c_dset, n_repeat=args.n_repeat, force_rewrite_cache=args.force)
    tcav.generate_CAVs(x2c_dset, x2c_dset, n_repeat=args.n_repeat, hparams=hparams, force_rewrite_cache=args.force)
    
if __name__ == '__main__':
    main()
