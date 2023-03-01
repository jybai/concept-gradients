import os
import pickle
import numpy as np
import pandas as pd
import PIL
import json
import glob
import shutil
import tarfile
import random
import urllib.request

import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import pad
from torch.utils.data import DataLoader, Dataset

from .CUB.data_processing import extract_driver
from .CUB.generate_new_data import get_class_attributes_data

class CellExtrusionDataset(Dataset):
    def __init__(self, root_dir, split, split_ratio=0.8, transform=None, return_attribute=False, seed=1126):
        
        assert split in ['train', 'val']
        
        pos_img_paths = glob.glob(os.path.join(root_dir, 'cell_extrusion_identification/es/*.tif'))
        neg_img_paths = glob.glob(os.path.join(root_dir, 'cell_extrusion_identification/nes/*.tif'))
        
        self.data = [(path, 1) for path in pos_img_paths] + [(path, 0) for path in neg_img_paths]
        
        random.seed(seed)
        random.shuffle(self.data)
        
        trn_size = int(split_ratio * len(self.data))
        if split == 'train':
            self.data = self.data[:trn_size]
        elif split == 'val':
            self.data = self.data[trn_size:]
        else:
            raise NotImplementedError
        
        self.transform = transform
        assert (not return_attribute)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        matrix = (0, 1, 0, 0)
        with PIL.Image.open(img_path) as image:
            image = image.resize((128, 128)).convert("L", matrix)

        if self.transform:
            image = self.transform(image)
        return image, label
    
    def calculate_normalization_stats(self):
        assert self.transform is None
        
        images = []
        for i in range(self.__len__()):
            image, label = self.__getitem__(i)
            image = np.array(image) / 255.
            images.append(image)
        images = np.stack(images, 0)
        return images.mean(), images.std()

class AwA2Dataset(Dataset):
    
    attr_groups = {
        'color': np.arange(0, 8),
        'surface': np.arange(8, 14),
        'size': np.arange(14, 18),
        'torso': np.arange(18, 26),
        'teeth': np.arange(26, 30),
        'exo': np.arange(30, 33),
        'terrain': np.arange(64, 78)
    }
    
    def __init__(self, root_dir, split, return_attribute=False, attr_binarize=True, attr_group=None, 
                 use_cc_128=False, transform=None):
        assert(split in ['train', 'valid', 'all'])
        split_ratio = 0.8
        
        data_dir = os.path.join(root_dir, 'Animals_with_Attributes2')
        if use_cc_128:
            img_dir = os.path.join(data_dir, 'JPEG_centercrop_128')
        else:
            img_dir = os.path.join(data_dir, 'JPEGImages')
        
        if attr_group is not None:
            assert(attr_group in list(self.attr_groups.keys()))
        self.attr_group = attr_group
        
        with open(os.path.join(data_dir, 'classes.txt')) as f:
            self.class_names = np.array([r.split('\t')[-1].rstrip("\n") for r in f.readlines()])
        
        with open(os.path.join(data_dir, 'predicates.txt')) as f:
            self.attr_names = np.array([r.split('\t')[-1].rstrip("\n") for r in f.readlines()])
            if self.attr_group is not None:
                self.attr_names = self.attr_names[self.attr_groups[self.attr_group]]
        
        if return_attribute:
            if attr_binarize:
                attr_fname = 'predicate-matrix-binary.txt'
                with open(os.path.join(data_dir, attr_fname)) as f:
                    labels = np.stack([np.array(r.split(' ')).astype(int) for r in f.readlines()], axis=0)
            else:
                attr_fname = 'predicate-matrix-continuous.txt'
                
                with open(os.path.join(data_dir, attr_fname)) as f:
                    arr = [np.array([e.strip() for e in r.split(' ') if e.strip() != '']).astype(float) 
                           for r in f.read().splitlines()]
                    labels = np.stack(arr, axis=0)
                
            if attr_group is not None:
                labels = labels[:, self.attr_groups[self.attr_group]]
        else:
            labels = np.arange(len(self.class_names))
        
        self.data = []
        for class_name, label in zip(self.class_names, labels):
            class_dir = os.path.join(img_dir, class_name)
            image_fnames = np.sort(glob.glob(os.path.join(class_dir, '*.jpg')))
            if split == 'train':
                image_fnames = image_fnames[:int(len(image_fnames) * split_ratio)]
            elif split == 'valid':
                image_fnames = image_fnames[int(len(image_fnames) * split_ratio):]
            elif split == 'all':
                pass
            else:
                raise NotImplementedError
            
            for image_fname in image_fnames:
                self.data.append((image_fname, label))
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_fname, label = self.data[idx]
        image = PIL.Image.open(img_fname).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = torch.as_tensor(label)
        return image, label
    

class BAMSceneDataset(Dataset):
    def _check_valid_exist(self, root_dir):
        if os.path.isdir(os.path.join(root_dir, 'valid')):
            return
        
        val_label_fname = os.path.join(root_dir, 'val.txt')
        with open(val_label_fname, 'r') as f:
            instances = f.readlines()
        class_names = set([instance.split('-')[1] for instance in instances])

        # create dirs
        for class_name in class_names:
            os.makedirs(os.path.join(root_dir, f'valid/{class_name}'), exist_ok=True)

        # copy each instance to their corresponding folder
        for instance in instances:
            shutil.copyfile(os.path.join(root_dir, f'val/{instance.split(" ")[0]}'), # src
                            os.path.join(root_dir, f'valid/{instance.split("-")[1]}/{instance.split(" ")[0]}'))
        
    def __init__(self, root_dir, split, transform=None):
        
        assert(split in ['train', 'valid'])
        root_dir = os.path.join(root_dir, 'bam/data/scene_only/')
        if split == 'valid':
            self._check_valid_exist(root_dir)
        root_dir = os.path.join(root_dir, f'{split}')
        
        img_fnames = glob.glob(os.path.join(root_dir, "*", "*.jpg"))
        scene_concepts = np.array(['bamboo_forest', 'bedroom', 'bowling_alley', 'bus_interior', 'cockpit', 
                                   'corn_field', 'laundromat', 'runway', 'ski_slope', 'track'])
        self.num_scene_concepts = len(scene_concepts)
        
        fname_concepts = []
        for img_fname in img_fnames:
            scene_index = np.where(scene_concepts == img_fname.split('/')[-1].split('-')[1])[0][0]
            concept_one_hot = self._one_hot(scene_index)
            fname_concepts.append((img_fname, concept_one_hot))
        
        self.data = fname_concepts
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_fname, attr = self.data[idx]
        image = PIL.Image.open(img_fname)
        if self.transform is not None:
            image = self.transform(image)
        attr = torch.as_tensor(attr)
        return image, attr
        
    def _one_hot(self, scene_index):
        one_hot_encoding = np.zeros(self.num_scene_concepts).astype(np.int32)
        one_hot_encoding[scene_index] = 1
        
        return one_hot_encoding

class BAMMixedDataset(Dataset):
    def _check_valid_exist(self, root_dir):
        if os.path.isdir(os.path.join(root_dir, 'valid')):
            return
        
        val_label_fname = os.path.join(root_dir, 'val.txt')
        with open(val_label_fname, 'r') as f:
            instances = f.readlines()
        class_names = set([instance.split('-')[1] for instance in instances])

        # create dirs
        for class_name in class_names:
            os.makedirs(os.path.join(root_dir, f'valid/{class_name}'), exist_ok=True)

        # copy each instance to their corresponding folder
        for instance in instances:
            shutil.copyfile(os.path.join(root_dir, f'val/{instance.split(" ")[0]}'), # src
                            os.path.join(root_dir, f'valid/{instance.split("-")[1]}/{instance.split(" ")[0]}'))
        
    def __init__(self, root_dir, split, transform=None, return_type='all', return_one_hot=True):
        
        assert(split in ['train', 'valid'])
        assert(return_type in ['all', 'scene', 'object'])
        
        root_dir = os.path.join(root_dir, 'bam/data/scene/')
        if split == 'valid':
            self._check_valid_exist(root_dir)
        root_dir = os.path.join(root_dir, f'{split}')
        
        img_fnames = glob.glob(os.path.join(root_dir, "*", "*.jpg"))
        obj_concepts = np.array(['backpack', 'bird', 'dog', 'elephant', 'kite', 
                                 'pizza', 'stop_sign', 'toilet', 'truck', 'zebra'])
        scene_concepts = np.array(['bamboo_forest', 'bedroom', 'bowling_alley', 'bus_interior', 'cockpit', 
                                   'corn_field', 'laundromat', 'runway', 'ski_slope', 'track'])

        self.num_obj_concepts = len(obj_concepts)
        self.num_scene_concepts = len(scene_concepts)
        self.return_type = return_type
        
        data = []
        for img_fname in img_fnames:
            obj_index = np.where(obj_concepts == img_fname.split('/')[-1].split('-')[0])[0][0]
            scene_index = np.where(scene_concepts == img_fname.split('/')[-1].split('-')[1])[0][0]
            if return_one_hot:
                concept_one_hot = self._one_hot(obj_index, scene_index)
                data.append((img_fname, concept_one_hot))
            else:
                if self.return_type == 'scene':
                    data.append((img_fname, scene_index))
                elif self.return_type == 'object':
                    data.append((img_fname, obj_index))
                else:
                    raise NotImplementedError
        
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_fname, attr = self.data[idx]
        image = PIL.Image.open(img_fname)
        if self.transform is not None:
            image = self.transform(image)
        attr = torch.as_tensor(attr)
        return image, attr
        
    def _one_hot(self, obj_index, scene_index):
        if self.return_type == 'all':
            one_hot_encoding = np.zeros(self.num_obj_concepts + self.num_scene_concepts).astype(np.int32)
            one_hot_encoding[obj_index] = 1
            one_hot_encoding[self.num_obj_concepts + scene_index] = 1
        elif self.return_type == 'scene':
            one_hot_encoding = np.zeros(self.num_scene_concepts).astype(np.int32)
            one_hot_encoding[scene_index] = 1
        elif self.return_type == 'scene':
            one_hot_encoding = np.zeros(self.num_obj_concepts).astype(np.int32)
            one_hot_encoding[obj_index] = 1
        else:
            raise NotImplementedError
        
        return one_hot_encoding

class CocoA(Dataset):
    def __init__(self, root_dir, split, transform=None):
        assert(split in ['train', 'val'])
        
        self.images_dir = os.path.join(root_dir, f'mscoco/{split}2014')
        
        attr_pkl_path = os.path.join(root_dir, f'mscocoa/cocottributes_eccv_version.pkl')
        with open(attr_pkl_path, 'rb') as f:
            attr = pickle.load(f, encoding='latin1')
            self.ann_ids = [attr['patch_id_to_ann_id'][attr_id] 
                            for attr_id in attr['ann_vecs'].keys()
                            if attr['split'][attr_id] == f'{split}2014']
            self.attr_anns = [v for k, v in attr['ann_vecs'].items()
                              if attr['split'][k] == f'{split}2014']
            del attr
        
        meta_json_path = os.path.join(root_dir, f'mscoco/annotations/instances_{split}2014.json')
        with open(meta_json_path, 'r') as f:
            meta_data = json.load(f)
            image_names = {d['id']: d['file_name'] for d in meta_data['images']}
            self.meta_data = {d['id']: {'bbox': d['bbox'], 'img_path': image_names[d['image_id']]} 
                              for d in meta_data['annotations']}
            del meta_data, image_names
        
        self.transform = transform
        
    def __len__(self):
        return len(self.ann_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        ann_id = self.ann_ids[idx]

        img_fname = os.path.join(self.images_dir, 
                                 self.meta_data[ann_id]['img_path'])
        left, upper, width, height = self.meta_data[ann_id]['bbox']
        right, lower = left + width, upper + height
        
        X = PIL.Image.open(img_fname).crop(box=(left, upper, right, lower)).convert("RGB")

        attr = torch.as_tensor(self.attr_anns[idx])

        if self.transform:
            X = self.transform(X)

        return X, attr

class CUBADataset(Dataset):
    
    def __init__(self, root_dir, split, transform=None, return_attribute=True, download=True, 
                 voted_concept_labels=True):
        '''https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/dataset.py
        '''
        assert(split in ['train', 'val', 'test', 'train_test', 'train_val'])
        
        src_dir = os.path.join(root_dir, 'CUB_200_2011')
        self.images_dir = os.path.join(src_dir, 'images')
        
        processed_dir = os.path.join(root_dir, 'CUB_processed')
        unvoted_dir = os.path.join(root_dir, 'CUB_unvoted')
        
        if (not os.path.isdir(self.images_dir)) or (not os.path.isdir(processed_dir)):
            print('Downloading data...')
            self.download(root_dir)
        
        if not os.path.isdir(unvoted_dir):
            print('Creating unvoted data...')
            self.create_unvoted(root_dir)
        
        self.meta_data = []
        for split_ in split.split('_'):
            if voted_concept_labels:
                meta_pkl_path = os.path.join(processed_dir, f'class_attr_data_10/{split_}.pkl')
            else:
                meta_pkl_path = os.path.join(unvoted_dir, f'{split_}.pkl')
            with open(meta_pkl_path, 'rb') as f:
                self.meta_data += pickle.load(f)
        
        self.transform = transform
        self.return_attribute = return_attribute
        
        # load attr_names
        with open(os.path.join(src_dir, 'attributes/attributes.txt'), 'r') as f:
            all_attr_names = np.array([r.split(' ')[1] for r in f.read().splitlines()])
        # https://github.com/yewsiang/ConceptBottleneck/blob/a2fd8184ad609bf0fb258c0b1c7a0cc44989f68f/CUB/generate_new_data.py#L65
        selected_attr_indices = np.array([1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 
                                          44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 
                                          84, 90, 91, 93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 
                                          131, 132, 134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 
                                          172, 178, 179, 181, 183, 187, 188, 193, 194, 196, 198, 202, 203, 
                                          208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 
                                          240, 242, 243, 244, 249, 253, 254, 259, 260, 262, 268, 274, 277, 
                                          283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311])
        self.attr_names = all_attr_names[selected_attr_indices]
        
        # load class_names
        with open(os.path.join(src_dir, 'classes.txt'), 'r') as f:
            self.class_names = np.array([r.split(' ')[1] for r in f.read().splitlines()])
    
    def download(self, root_dir):
        ''' https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/README.md
        '''
        fname_url_pairs = [
            ("CUB_200_2011", "https://worksheets.codalab.org/rest/bundles/0xd013a7ba2e88481bbc07e787f73109f5/contents/blob/"),
            ("CUB_processed", "https://worksheets.codalab.org/rest/bundles/0x5b9d528d2101418b87212db92fea6683/contents/blob/"),
        ]
        
        for fname, url in fname_url_pairs:
            expand_dir = os.path.join(root_dir, fname)
            tar_path = os.path.join(root_dir, f"{fname}.tar.gz")
            os.makedirs(expand_dir, exist_ok=True)
        
            urllib.request.urlretrieve(url, tar_path)
            with tarfile.open(tar_path) as f:
                f.extractall(expand_dir)
            
            os.remove(tar_path)
    
    def create_unvoted(self, root_dir, min_class_count=10):
        src_dir = os.path.join(root_dir, 'CUB_200_2011')
        processed_dir = os.path.join(root_dir, 'CUB_processed/class_attr_data_10')
        raw_dir = os.path.join(root_dir, 'CUB_raw')
        unvoted_dir = os.path.join(root_dir, 'CUB_unvoted')
        
        assert os.path.isdir(src_dir) and os.path.isdir(processed_dir)
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(unvoted_dir, exist_ok=True)
        
        extract_driver(raw_dir, src_dir, ref_data_dir=processed_dir)
        get_class_attributes_data(min_class_count, unvoted_dir, modify_data_dir=raw_dir, keep_instance_data=True)
    
    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_fname = os.path.join(self.images_dir, 
                                 *self.meta_data[idx]['img_path'].split('/')[-2:])
        X = PIL.Image.open(img_fname).convert("RGB")
        
        if self.return_attribute:
            Y = torch.as_tensor(self.meta_data[idx]['attribute_label'])
        else:
            Y = torch.as_tensor(self.meta_data[idx]['class_label'])
            
        if self.transform:
            X = self.transform(X)

        return X, Y
    
class CUBC2YDataset(CUBADataset):
    def __init__(self, root_dir, split, download=True, voted_concept_labels=True):
        super().__init__(root_dir, split, download=download, 
                         voted_concept_labels=voted_concept_labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        X = torch.as_tensor(self.meta_data[idx]['attribute_label']).float()
        Y = torch.as_tensor(self.meta_data[idx]['class_label'])
        
        return X, Y

class SCUTFBP5500v2Dataset(Dataset):

    def __init__(self, root_dir, split, transform=None):
        assert(split in ['train', 'test'])
        label_path = os.path.join(root_dir, f'SCUT-FBP5500_v2/train_test_files/split_of_60%training and 40%testing/{split}.txt')
        df = pd.read_csv(label_path, delim_whitespace=True, header=None)
        df.columns = ['img_fname', 'label']
        
        self.img_dir = os.path.join(root_dir, 'SCUT-FBP5500_v2/Images')
        self.transform = transform
        self.meta_data = df

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_fname = os.path.join(self.img_dir,
                                 self.meta_data.iloc[idx]['img_fname'])
        X = PIL.Image.open(img_fname)
        beauty_rank = self.meta_data.iloc[idx]['label'].astype(np.float32)
        beauty_rank = np.expand_dims(beauty_rank, axis=-1)
        
        if self.transform:
            X = self.transform(X)

        return X, beauty_rank