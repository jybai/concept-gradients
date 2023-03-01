import os
import argparse
import yaml
import numpy as np
from tqdm.auto import tqdm, trange

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from captum.attr import IntegratedGradients

from .dataset import CUBADataset
from .loader import load_dataset_and_model

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dset_name', type=str)
    parser.add_argument('x2y_model_path', type=str, help="Should be the checkpoint file.")
    parser.add_argument('save_path', type=str, help="Path to save the attributed numpy file.")
    parser.add_argument('--x2y_arch_name', type=str, default='inception_v3', 
                        choices=['inception_v3', 'resnet50', 'dup-resnet50', 'vgg11_bn', 'vgg16_bn'])
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--data_root_dir', type=str, default='/home/andrewbai/data/')
    # parser.add_argument('--attr_root_dir', type=str, default='/home/andrewbai/attrs/')
    # parser.add_argument('--save_fname_suffix', type=str, default=None)
    
    parser.add_argument('--bsize', type=int, default=32,
                        help="Does not affect result. Only affects speed. Reduce when OOM.")
    parser.add_argument('--x2y_cfg_path', type=str, default=None)
    return parser.parse_args()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_arguments()
    
    x2y_model_kwargs = {}
    if args.x2y_cfg_path is not None:
        with open(args.x2y_cfg_path, 'r') as f:
            x2y_model_kwargs = yaml.safe_load(f)
    
    x2y_dl_train, x2y_dl_valid, x2y_dl_test, x2y_model = load_dataset_and_model(
        dset_name=args.dset_name, task='x2y', data_root_dir=args.data_root_dir, 
        use_all_data=False, arch_name=args.x2y_arch_name, bsize=args.bsize, model_kwargs=x2y_model_kwargs)
    x2y_model.load_state_dict(torch.load(args.x2y_model_path))
    x2y_model = x2y_model.to(device).eval()

    ig = IntegratedGradients(x2y_model, multiply_by_inputs=False)
    attrs = []
    
    if args.split == 'val':
        dl = x2y_dl_valid
    elif args.split == 'test':
        dl = x2y_dl_test
    else:
        raise ValueError

    for x, y in tqdm(dl, leave=True):
        x = x.to(device)
        x.requires_grad = True
        y = y.to(device)

        attr = ig.attribute(x, baselines=None, target=y)
        attr = attr.detach().cpu().numpy()
        attrs.append(attr)

    attrs = np.concatenate(attrs, axis=0)
    np.save(args.save_path, attrs)

if __name__ == '__main__':
    main()
