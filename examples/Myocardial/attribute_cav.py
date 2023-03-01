import os
import argparse
import numpy as np
from tqdm.auto import tqdm, trange

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cg.tcav import TCAV

from data import MIComplications
from model import ClassifierHead

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('x2y_model_path', type=str, help="Should be the checkpoint file.")
    parser.add_argument('cav_dir', type=str, help="Should be the folder containing `CAV.npz`.")
    # optional
    parser.add_argument('--layers', type=str, nargs='+', help='The layers to attribute TCAV.', default=None)
    parser.add_argument('--mode', type=str, default='inner_prod', help="Mode to calculate CAV attribution.",
                        choices=['inner_prod', 'cosine_similarity'])
    parser.add_argument('--data_root_dir', type=str, default='/home/andrewbai/data/')
    
    parser.add_argument('--bsize', type=int, default=32,
                        help="Does not affect result. Only affects speed. Reduce when OOM.")
    parser.add_argument('--signed', action='store_true', help='Take sign before aggregating.')
    return parser.parse_args()

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args = parse_arguments()
    
    x2c_dset = MIComplications(args.data_root_dir, return_attribute=True)
    dl = DataLoader(x2c_dset, batch_size=args.bsize, num_workers=8)
    
    x2y_model = ClassifierHead(x2c_dset.X.shape[1], 1, squeeze_final=True)
    x2y_model.load_state_dict(torch.load(args.x2y_model_path))
    x2y_model = x2y_model.to(device).eval()
    
    tcav = TCAV(x2y_model, layer_names=None, cache_dir=args.cav_dir)
    
    if args.layers is None:
        layers = tcav.layer_names
    else:
        layers = args.layers
        assert all([(layer in tcav.layer_names) for layer in layers])
    
    print(x2c_dset.C_labels)
    
    for layer_name in layers:
        attrs = []
        for xs, cs in dl:
            xs = xs.to(device)
            
            attr = tcav.attribute(xs, layer_name, args.mode, target=None)
            attr = attr.detach().cpu().numpy()
            attrs.append(attr)

        attrs = np.concatenate(attrs, axis=0)
        print(layer_name)
        if args.signed:
            print((attrs > 0).mean(0))
        else:
            print(attrs.mean(0))    
        
if __name__ == '__main__':
    main()