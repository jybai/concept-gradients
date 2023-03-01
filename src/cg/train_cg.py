import argparse
import os
import sys
import yaml
import glob
import random
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL
import seaborn as sns

from copy import deepcopy
from importlib import reload
from scipy.stats import ttest_1samp
from time import sleep
from tqdm import tqdm, trange

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CelebA, ImageFolder
import torchmetrics
from pytorch_lightning.loggers import CSVLogger

from .loader import load_dataset_and_model
from .train_utils import seed_everything, calculate_pos_weight

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dset_name', metavar='D', type=str, choices=['cuba', 'cuba-unvoted'])
    parser.add_argument('task', metavar='T', type=str, choices=['x2y', 'c2y', 'x2c'])
    parser.add_argument('--exp_name_prefix', type=str, default=None)
    parser.add_argument('--arch_name', type=str, default='inception_v3', 
                        choices=['inception_v3', 'resnet50', 'vgg11_bn', 'vgg16_bn'])
    parser.add_argument('--lr', type=float, default=1e-2, help='search range: [1e-2, 1e-3]')
    parser.add_argument('--scheduler_step', type=int, default=15, help='search range: [15, 20, 25]')
    parser.add_argument('--weight_decay', type=float, default=4e-4, help='search range: [4e-4, 4e-5]')
    parser.add_argument('--data_root_dir', type=str, default='/home/andrewbai/data/')
    parser.add_argument('--save_dir', type=str, default='./models/')
    parser.add_argument('--x2c_from_x2y_ckpt', type=str, default=None)
    parser.add_argument('--finetune_layer_start', type=str, default=None,
                        help='Default finetunes entire network.')
    parser.add_argument('--use_all_data', action='store_true')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--bsize', type=int, default=64)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--nepochs', type=int, default=200)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--min_lr', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    return parser.parse_args()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args = parse_arguments()
    
    if args.seed is not None:
        seed_everything(args.seed)
        print(f"Using random seed {args.seed}.")
    
    '''Fixed hparams
    '''
    if args.exp_name_prefix is not None:
        exp_name = args.exp_name_prefix
    else:
        exp_name = f"{args.dset_name}_{args.task}_{args.arch_name}"
    if args.use_all_data:
        exp_name += '_all-data'
    if args.x2c_from_x2y_ckpt is not None:
        exp_name += '_ft-x2y'
    if args.finetune_layer_start is not None:
        exp_name += f'_ft-{args.finetune_layer_start}+'
    if args.seed is not None:
        exp_name += f'_seed{args.seed}'
    print(exp_name)
    os.makedirs(os.path.join(args.save_dir, exp_name), exist_ok=True)
    
    logger = CSVLogger(args.save_dir, name=exp_name, flush_logs_every_n_steps=1)
    logger.log_hyperparams(vars(args))
    
    dl_train, dl_valid, dl_test, model = load_dataset_and_model(
        args.dset_name, args.task, args.data_root_dir, args.use_all_data, 
        args.arch_name, args.x2c_from_x2y_ckpt, args.bsize)
    
    if args.task == 'x2y':
        loss_fn = nn.CrossEntropyLoss()
    elif args.task == 'c2y':
        loss_fn = nn.CrossEntropyLoss()
    elif args.task == 'x2c':
        pos_weight = calculate_pos_weight(dl_train)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    
    if args.finetune_layer_start is not None:
        finetune_flag = False
        for name, p in model.named_parameters():
            if name.startswith(args.finetune_layer_start):
                finetune_flag = True
            p.requires_grad = finetune_flag
    
        print('Trainable parameters:')
        for name, p in model.named_parameters():
            if p.requires_grad:
                print(name)
        
    model = model.to(device)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.lr_decay_rate)
    
    patience_cnt = 0
    with trange(args.nepochs, leave=False, desc='Epochs') as tepochs:
        best_metric = 0
        best_loss = np.inf
        
        for epoch in tepochs:
            model.train()
            for x, y in tqdm(dl_train, leave=False, desc='Training steps'):
                optimizer.zero_grad()

                x = x.to(device)
                y = y.to(device)

                if args.task == 'x2y' or args.task == 'x2c':
                    if args.arch_name == 'inception_v3':
                        pred_y, aux_pred_y = model(x)
                        if args.task == 'x2c':
                            loss = loss_fn(pred_y, y.float()) + 0.4 * loss_fn(aux_pred_y, y.float())
                        else:
                            loss = loss_fn(pred_y, y) + 0.4 * loss_fn(aux_pred_y, y)
                    else:
                        pred_y = model(x)
                        if args.task == 'x2c':
                            loss = loss_fn(pred_y, y.float())
                        else:
                            loss = loss_fn(pred_y, y)
                elif args.task == 'c2y':
                    pred_y = model(x)
                    loss = loss_fn(pred_y, y)
                loss.backward()
                optimizer.step()

            model.eval()
            eval_fn = torchmetrics.Accuracy().to(device)
            losses = []
            with torch.no_grad():
                for x, y in tqdm(dl_valid, leave=False, desc='Eval steps'):
                    x = x.to(device)
                    y = y.to(device)
                    
                    pred_y = model(x)
                    if args.task == 'x2c':
                        eval_fn(pred_y, y)
                        loss = loss_fn(pred_y, y.float())
                    else:
                        eval_fn(pred_y.argmax(-1), y)
                        loss = loss_fn(pred_y, y)
                    losses.append(loss.item())
                    
            eval_metric = eval_fn.compute().item()
            
            if best_metric < eval_metric:
                best_metric = eval_metric
                if args.save_model:
                    torch.save(model.state_dict(), os.path.join(args.save_dir, exp_name, 
                                                                f'version_{logger.version}', 'model.ckpt'))
                patience_cnt = 0
            else:
                patience_cnt += 1
            
            best_loss = np.min([np.mean(losses), best_loss])
            
            logger.log_metrics(dict(acc=eval_metric, loss=np.mean(losses)))
            
            tepochs.set_postfix(loss=f"{np.mean(losses):.3f}/{best_loss:.3f}", 
                                acc=f"{eval_metric:.3f}/{best_metric:.3f}", 
                                patience=f"{patience_cnt}/{args.patience}",)
            sleep(0.1)
            
            if scheduler.get_last_lr()[0] > args.min_lr:
                scheduler.step()
                
            if patience_cnt >= args.patience:
                break
    
    logger.save()
    print(f"best val acc = {best_metric}")

if __name__ == '__main__':
    main()
