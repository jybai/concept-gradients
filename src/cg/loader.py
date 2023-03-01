import argparse
import os
import numpy as np
from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import CelebA, ImageFolder

from .dataset import CUBADataset, CUBC2YDataset
from .models.duplicate_resblock_resnet import duplicate_resblock_resnet50

def load_dataset_and_model(dset_name, task, data_root_dir, use_all_data, arch_name, 
                           x2c_from_x2y_ckpt=None, bsize=64, num_workers=8, return_dataloader=True, 
                           model_kwargs={}):
    
    if dset_name == 'cuba':
        NUM_CLASSES = 200
        NUM_CONCEPTS = 112
        dset_class = lambda *args, **kwargs: CUBADataset(*args, voted_concept_labels=True, **kwargs)
    elif dset_name == 'cuba-unvoted':
        NUM_CLASSES = 200
        NUM_CONCEPTS = 112
        dset_class = lambda *args, **kwargs: CUBADataset(*args, voted_concept_labels=False, **kwargs)
    else:
        raise NotImplementedError
    
    if arch_name == 'inception_v3':
        model = torch.hub.load('pytorch/vision:v0.9.0', arch_name, pretrained=True, verbose=False)
        img_size = 299

        if task == 'x2y':
            model.AuxLogits.fc = nn.Linear(768, NUM_CLASSES)
            model.fc = nn.Linear(2048, NUM_CLASSES)

        elif task == 'x2c':
            if x2c_from_x2y_ckpt is not None:
                model.AuxLogits.fc = nn.Linear(768, NUM_CLASSES)
                model.fc = nn.Linear(2048, NUM_CLASSES)
                model.load_state_dict(torch.load(x2c_from_x2y_ckpt))
                print(f"Loaded pretrained weights from {x2c_from_x2y_ckpt}.")
            model.AuxLogits.fc = nn.Linear(768, NUM_CONCEPTS)
            model.fc = nn.Linear(2048, NUM_CONCEPTS)

    elif arch_name == 'resnet50':
        model = torch.hub.load('pytorch/vision:v0.9.0', arch_name, pretrained=True, verbose=False)
        img_size = 224

        if task == 'x2y':
            model.fc = nn.Linear(2048, NUM_CLASSES)

        elif task == 'x2c':
            if x2c_from_x2y_ckpt is not None:
                model.fc = nn.Linear(2048, NUM_CLASSES)
                model.load_state_dict(torch.load(x2c_from_x2y_ckpt))
                print(f"Loaded pretrained weights from {x2c_from_x2y_ckpt}.")
            model.fc = nn.Linear(2048, NUM_CONCEPTS)
            
    elif arch_name == 'dup-resnet50':
        model = duplicate_resblock_resnet50(model_kwargs['duplicate_layer'],
                                            model_kwargs['duplicate_block'],
                                            model_kwargs['duplicate_copies'])
        img_size = 224

        if task == 'x2y':
            model.fc = nn.Linear(2048, NUM_CLASSES)

        elif task == 'x2c':
            if x2c_from_x2y_ckpt is not None:
                model.fc = nn.Linear(2048, NUM_CLASSES)
                model.load_state_dict_from_nonduplicate(torch.load(x2c_from_x2y_ckpt))
                print(f"Loaded pretrained weights from {x2c_from_x2y_ckpt}.")
            model.fc = nn.Linear(2048, NUM_CONCEPTS)

    elif arch_name == 'vgg11_bn' or arch_name == 'vgg16_bn':
        model = torch.hub.load('pytorch/vision:v0.9.0', arch_name, pretrained=True, verbose=False)
        img_size = 224

        if task == 'x2y':
            model.classifier[6] = nn.Linear(4096, NUM_CLASSES)

        elif task == 'x2c':
            if x2c_from_x2y_ckpt is not None:
                model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
                model.load_state_dict(torch.load(x2c_from_x2y_ckpt))
                print(f"Loaded pretrained weights from {x2c_from_x2y_ckpt}.")
            model.classifier[6] = nn.Linear(4096, NUM_CONCEPTS)

    else:
        raise NotImplementedError
        
    # dataset

    train_transform = transforms.Compose([ 
        transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    valid_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if not use_all_data:
        train_split = 'train'
        valid_split = 'val'
    else:
        train_split = 'train_val'
        valid_split = 'train_val'
    test_split = 'test'
    
    if task == 'x2y':
        dset_train = dset_class(root_dir=data_root_dir, split=train_split, 
                                transform=train_transform, return_attribute=False)
        dset_valid = dset_class(root_dir=data_root_dir, split=valid_split, 
                                transform=valid_transform, return_attribute=False)
        dset_test = dset_class(root_dir=data_root_dir, split=test_split, 
                               transform=valid_transform, return_attribute=False)
    elif task == 'x2c':
        dset_train = dset_class(root_dir=data_root_dir, split=train_split, 
                                transform=train_transform, return_attribute=True)
        dset_valid = dset_class(root_dir=data_root_dir, split=valid_split, 
                                transform=valid_transform, return_attribute=True)
        dset_test = dset_class(root_dir=data_root_dir, split=test_split, 
                               transform=valid_transform, return_attribute=True)
    
    if not return_dataloader:
        return dset_train, dset_valid, dset_test, model
    
    dl_train = DataLoader(dset_train, batch_size=bsize, shuffle=True,
                          drop_last=True, num_workers=num_workers)
    dl_valid = DataLoader(dset_valid, batch_size=bsize, shuffle=False,
                          drop_last=False, num_workers=num_workers)
    dl_test = DataLoader(dset_test, batch_size=bsize, shuffle=False,
                         drop_last=False, num_workers=num_workers)
    
    return dl_train, dl_valid, dl_test, model
