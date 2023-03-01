import os
import json 
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# reference: https://github.com/Deepwise-AILab/ChestX-Det-Dataset/blob/be718a08a220c26cb468000e88e7df7737e1b93b/pre-trained_PSPNet/seg.py#L10

chestx_transform = transforms.Compose([
    transforms.Lambda(lambda x: cv2.resize(x, (512, 512), interpolation=cv2.INTER_CUBIC)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class ChestX(Dataset):
    def __init__(self, root_dir, split, transform=chestx_transform):
        
        dset_dir = os.path.join(root_dir, 'ChestX_Det')
        assert split in ['trn', 'tst']
        
        self.all_syms = ['Atelectasis', 'Calcification', 'Cardiomegaly', 'Consolidation', 
                         'Diffuse Nodule', 'Effusion', 'Emphysema', 'Fibrosis', 'Fracture', 
                         'Mass', 'Nodule', 'Pleural Thickening', 'Pneumothorax']
        
        def one_hot_encode(syms, all_syms):
            encoding = [int(s in syms) for s in all_syms]
            return encoding
        
        if split == 'trn':
            self.images_dir = os.path.join(dset_dir, 'train')
            with open(os.path.join(dset_dir, 'ChestX_Det_train.json')) as f:
                self.data = json.load(f)
            for i in range(len(self.data)):
                self.data[i]['syms_encoding'] = one_hot_encode(self.data[i]['syms'], self.all_syms)
        elif split == 'tst':
            self.images_dir = os.path.join(dset_dir, 'test')
            with open(os.path.join(dset_dir, 'ChestX_Det_test.json')) as f:
                self.data = json.load(f)
            for i in range(len(self.data)):
                self.data[i]['syms_encoding'] = one_hot_encode(self.data[i]['syms'], self.all_syms)
        else:
            raise NotImplementedError
            
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.data[idx]['file_name'])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) # if set `cv2.IMREAD_GRAYSCALE` flag, returns 1 channel instead of 3
        
        if self.transform is not None:
            img = self.transform(img)
        
        labels = self.data[idx]['syms_encoding']
        labels = torch.as_tensor(labels)
        
        return img, labels
