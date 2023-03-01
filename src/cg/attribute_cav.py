import os
import argparse
import numpy as np
from tqdm.auto import tqdm, trange

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .loader import load_dataset_and_model
from .tcav import TCAV

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dset_name', type=str)
    parser.add_argument('x2y_model_path', type=str, help="Should be the checkpoint file.")
    parser.add_argument('cav_dir', type=str, help="Should be the folder containing `CAV.npz`.")
    parser.add_argument('--base_save_path', type=str, help="Base path to save the attributed file.")
    parser.add_argument('--arch_name', type=str, choices=['inception_v3', 'resnet50', 'vgg16_bn'])
    # optional
    parser.add_argument('--layers', type=str, nargs='+', help='The layers to attribute TCAV.', default=None)
    parser.add_argument('--mode', type=str, default='inner_prod', help="Mode to calculate CAV attribution.",
                        choices=['inner_prod', 'cosine_similarity'])
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--data_root_dir', type=str, default='/home/andrewbai/data/')
    
    parser.add_argument('--bsize', type=int, default=32,
                        help="Does not affect result. Only affects speed. Reduce when OOM.")
    return parser.parse_args()

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args = parse_arguments()
    
    x2y_dl_train, x2y_dl_valid, x2y_dl_test, x2y_model = load_dataset_and_model(
        dset_name=args.dset_name, task='x2y', data_root_dir=args.data_root_dir, 
        use_all_data=False, arch_name=args.arch_name, bsize=args.bsize, return_dataloader=True)
    x2y_model.load_state_dict(torch.load(args.x2y_model_path))
    x2y_model = x2y_model.to(device).eval()
    
    tcav = TCAV(x2y_model, layer_names=None, cache_dir=args.cav_dir)
    
    if args.layers is None:
        layers = tcav.layer_names
    else:
        layers = args.layers
        assert all([(layer in tcav.layer_names) for layer in layers])
    
    if args.split == 'val':
        dl = x2y_dl_valid
    elif args.split == 'test':
        dl = x2y_dl_test
    else:
        raise ValueError
    
    for layer_name in tqdm(layers, leave=True, desc="Layers: "):
        attrs = []
        for xs, ys in tqdm(dl, leave=False, desc="Eval dset: "):
            xs = xs.to(device)
            ys = ys.to(device)

            attr = tcav.attribute(xs, layer_name, args.mode, target=ys)
            attr = attr.detach().cpu().numpy()
            attrs.append(attr)

        attrs = np.concatenate(attrs, axis=0)
        attr_path = os.path.join(args.base_save_path, f'cav_{args.arch_name}_{layer_name}.npy')
        np.save(attr_path, attrs)
        
if __name__ == '__main__':
    main()