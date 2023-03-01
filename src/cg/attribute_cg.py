import os
import argparse
import yaml
import numpy as np
from tqdm.auto import tqdm, trange

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from .dataset import CUBADataset
from .concept_gradients import ConceptGradients
from .loader import load_dataset_and_model

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dset_name', type=str)
    parser.add_argument('x2y_model_path', type=str, help="Should be the checkpoint file.")
    parser.add_argument('x2c_model_path', type=str, help="Should be the checkpoint file.")
    parser.add_argument('save_path', type=str, help="Path to save the attributed numpy file.")
    parser.add_argument('--mode', type=str, default='chain_rule_independent', help="Mode to calculate CG.",
                        choices=['chain_rule_joint', 'chain_rule_independent', 'cav', 'inner_prod', 'cosine_similarity'])
    parser.add_argument('--x2y_arch_name', type=str, default='inception_v3', 
                        choices=['inception_v3', 'resnet50', 'dup-resnet50', 'vgg11_bn', 'vgg16_bn'])
    parser.add_argument('--x2c_arch_name', type=str, default=None, 
                        choices=['inception_v3', 'resnet50', 'dup-resnet50', 'vgg11_bn', 'vgg16_bn', None])
    parser.add_argument('--layer', type=str, default=None)
    parser.add_argument('--x2c_layer', type=str, default=None)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--data_root_dir', type=str, default='/home/andrewbai/data/')
    # parser.add_argument('--attr_root_dir', type=str, default='/home/andrewbai/attrs/')
    # parser.add_argument('--save_fname_suffix', type=str, default=None)
    
    parser.add_argument('--bsize', type=int, default=32,
                        help="Does not affect result. Only affects speed. Reduce when OOM.")
    parser.add_argument('--x2y_cfg_path', type=str, default=None)
    parser.add_argument('--x2c_cfg_path', type=str, default=None)
    return parser.parse_args()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_arguments()
    
    x2y_model_kwargs, x2c_model_kwargs = {}, {} 
    if args.x2y_cfg_path is not None:
        with open(args.x2y_cfg_path, 'r') as f:
            x2y_model_kwargs = yaml.safe_load(f)
    if args.x2c_cfg_path is not None:
        with open(args.x2c_cfg_path, 'r') as f:
            x2c_model_kwargs = yaml.safe_load(f)
    
    # default using x2y.arch_name as x2c.arch_name
    if args.x2c_arch_name is None:
        args.x2c_arch_name = args.x2y_arch_name
    if args.x2c_layer is None:
        args.x2c_layer = args.layer
        
    x2y_dl_train, x2y_dl_valid, x2y_dl_test, x2y_model = load_dataset_and_model(
        dset_name=args.dset_name, task='x2y', data_root_dir=args.data_root_dir, 
        use_all_data=False, arch_name=args.x2y_arch_name, bsize=args.bsize, model_kwargs=x2y_model_kwargs)
    x2y_model.load_state_dict(torch.load(args.x2y_model_path))
    x2y_model = x2y_model.to(device).eval()
    
    x2c_dl_train, x2c_dl_valid, x2c_dl_test, x2c_model = load_dataset_and_model(
        dset_name=args.dset_name, task='x2c', data_root_dir=args.data_root_dir, 
        use_all_data=False, arch_name=args.x2c_arch_name, bsize=args.bsize, model_kwargs=x2c_model_kwargs)
    x2c_model.load_state_dict(torch.load(args.x2c_model_path))
    x2c_model = x2c_model.to(device).eval()
    
    # hack to get number of concepts
    for xs, cs in x2c_dl_test:
        n_concepts = cs.shape[1]
        break
    
    cg = ConceptGradients(forward_func=x2y_model, concept_forward_func=x2c_model)
    
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

        attr = cg.attribute(x, mode=args.mode, target=y, n_concepts=n_concepts,
                            target_layer_name=args.layer, concept_layer_name=args.x2c_layer)
        attr = attr.detach().cpu().numpy()
        attrs.append(attr)

    attrs = np.concatenate(attrs, axis=0)
    
    '''
    layer_str = '' if args.layer is None else '-' + args.layer
    suffix = '' if args.save_fname_suffix is None else '_' + args.save_fname_suffix
    attr_path = os.path.join(args.attr_root_dir, f"{args.dset_name}_cg-{args.mode}{layer_str}_{args.x2y_arch_name}{suffix}.npy")
    np.save(attr_path, attrs)
    '''
    
    np.save(args.save_path, attrs)

if __name__ == '__main__':
    main()
