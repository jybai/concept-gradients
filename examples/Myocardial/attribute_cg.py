import os
import argparse
import yaml
import numpy as np
from tqdm.auto import tqdm, trange

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from cg.concept_gradients import ConceptGradients

from data import MIComplications
from model import ClassifierHead

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('x2y_model_path', type=str, help="Should be the checkpoint file.")
    parser.add_argument('x2c_model_path', type=str, help="Should be the checkpoint file.")
    parser.add_argument('--mode', type=str, default='chain_rule_independent', help="Mode to calculate CG.",
                        choices=['chain_rule_joint', 'chain_rule_independent', 'cav', 'inner_prod', 'cosine_similarity'])
    parser.add_argument('--layer', type=str, default=None)
    parser.add_argument('--x2c_layer', type=str, default=None)
    parser.add_argument('--data_root_dir', type=str, default='/home/andrewbai/data/')
    
    parser.add_argument('--bsize', type=int, default=32,
                        help="Does not affect result. Only affects speed. Reduce when OOM.")
    return parser.parse_args()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_arguments()
    
    x2c_dset = MIComplications(args.data_root_dir, return_attribute=True)
    dl = DataLoader(x2c_dset, batch_size=args.bsize, num_workers=8)
    
    n_concepts = x2c_dset.C.shape[1]
    
    if args.x2c_layer is None:
        args.x2c_layer = args.layer
    
    x2y_model = ClassifierHead(x2c_dset.X.shape[1], 1, squeeze_final=True)
    x2y_model.load_state_dict(torch.load(args.x2y_model_path))
    x2y_model = x2y_model.to(device).eval()
    
    x2c_model = ClassifierHead(x2c_dset.X.shape[1], n_concepts, squeeze_final=True)
    x2c_model.load_state_dict(torch.load(args.x2c_model_path))
    x2c_model = x2c_model.to(device).eval()
    
    cg = ConceptGradients(forward_func=x2y_model, concept_forward_func=x2c_model)
    
    attrs = []

    for x, y in tqdm(dl, leave=True):
        x = x.to(device)
        x.requires_grad = True

        attr = cg.attribute(x, mode=args.mode, n_concepts=n_concepts, target_layer_name=args.layer,
                            concept_layer_name=args.x2c_layer)
        attr = attr.detach().cpu().numpy()
        attrs.append(attr)

    attrs = np.concatenate(attrs, axis=0)
    
    print(x2c_dset.C_labels)
    print((attrs > 0).mean(0))

if __name__ == '__main__':
    main()
