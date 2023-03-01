"""
Make train, val, test datasets based on train_test_split.txt, and by sampling val_ratio of the official train data to make a validation set 
Each dataset is a list of metadata, each includes official image id, full image path, class label, attribute labels, attribute certainty scores, and attribute labels calibrated for uncertainty
https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/data_processing.py
"""
import os
import random
import pickle
import argparse
from os import listdir
from os.path import isfile, isdir, join
from collections import defaultdict as ddict
import numpy as np


def extract_data(data_dir, train_val_split_ids=None):
    '''
    train_val_split_ids = {
        'train': [ids belonging to the train split],
        'val': [ids belonging to the val split],
    }
    '''
    cwd = os.getcwd()
    data_path = join(cwd,data_dir + '/images')
    val_ratio = 0.2

    path_to_id_map = dict() #map from full image path to image id
    with open(data_path.replace('images', 'images.txt'), 'r') as f:
        for line in f:
            items = line.strip().split()
            path_to_id_map[join(data_path, items[1])] = int(items[0])

    attribute_labels_all = ddict(list) #map from image id to a list of attribute labels
    attribute_certainties_all = ddict(list) #map from image id to a list of attribute certainties
    attribute_uncertain_labels_all = ddict(list) #map from image id to a list of attribute labels calibrated for uncertainty
    # 1 = not visible, 2 = guessing, 3 = probably, 4 = definitely
    uncertainty_map = {1: {1: 0, 2: 0.5, 3: 0.75, 4:1}, #calibrate main label based on uncertainty label
                        0: {1: 0, 2: 0.5, 3: 0.25, 4: 0}}
    with open(join(cwd, data_dir + '/attributes/image_attribute_labels.txt'), 'r') as f:
        for line in f:
            file_idx, attribute_idx, attribute_label, attribute_certainty = line.strip().split()[:4]
            attribute_label = int(attribute_label)
            attribute_certainty = int(attribute_certainty)
            uncertain_label = uncertainty_map[attribute_label][attribute_certainty]
            attribute_labels_all[int(file_idx)].append(attribute_label)
            attribute_uncertain_labels_all[int(file_idx)].append(uncertain_label)
            attribute_certainties_all[int(file_idx)].append(attribute_certainty)

    is_train_test = dict() #map from image id to 0 / 1 (1 = train)
    with open(join(cwd, data_dir + '/train_test_split.txt'), 'r') as f:
        for line in f:
            idx, is_train = line.strip().split()
            is_train_test[int(idx)] = int(is_train)
    print("Number of train images from official train test split:", sum(list(is_train_test.values())))

    train_val_data, test_data = [], []

    folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
    folder_list.sort() #sort by class index
    for i, folder in enumerate(folder_list):
        folder_path = join(data_path, folder)
        classfile_list = [cf for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.')]
        #classfile_list.sort()
        for cf in classfile_list:
            img_id = path_to_id_map[join(folder_path, cf)]
            img_path = join(folder_path, cf)
            metadata = {'id': img_id, 'img_path': img_path, 'class_label': i,
                      'attribute_label': attribute_labels_all[img_id], 'attribute_certainty': attribute_certainties_all[img_id],
                      'uncertain_attribute_label': attribute_uncertain_labels_all[img_id]}
            if is_train_test[img_id]:
                train_val_data.append(metadata)
            else:
                test_data.append(metadata)

    train_data, val_data = [], []
    if train_val_split_ids is not None:
        assert 'train' in train_val_split_ids
        assert 'val' in train_val_split_ids
        assert np.array_equal(np.sort([d['id'] for d in train_val_data]),
                              np.sort(train_val_split_ids['train'] + train_val_split_ids['val']))
        for id_ in train_val_split_ids['train']:
            d = [d_ for d_ in train_val_data if d_['id'] == id_][0]
            train_data.append(d)
        for id_ in train_val_split_ids['val']:
            d = [d_ for d_ in train_val_data if d_['id'] == id_][0]
            val_data.append(d)
    else:
        random.shuffle(train_val_data)
        split = int(val_ratio * len(train_val_data))
        train_data = train_val_data[split :]
        val_data = train_val_data[: split]
    print('Size of train set:', len(train_data))
    return train_data, val_data, test_data

def extract_driver(save_dir, data_dir, ref_data_dir=None):
    
    train_val_split_ids = None
    if ref_data_dir is not None:
        with open(os.path.join(ref_data_dir, 'train.pkl'), 'rb') as f:
            ref_train_data = pickle.load(f)
        with open(os.path.join(ref_data_dir, 'val.pkl'), 'rb') as f:
            ref_val_data = pickle.load(f)
        train_val_split_ids = {'train': [d['id'] for d in ref_train_data],
                               'val': [d['id'] for d in ref_val_data],
                              }
        del ref_train_data, ref_val_data
    
    train_data, val_data, test_data = extract_data(data_dir, train_val_split_ids)

    for dataset in ['train','val','test']:
        print("Processing %s set" % dataset)
        with open(os.path.join(save_dir, dataset + '.pkl'), 'wb') as f:
            if 'train' == dataset:
                pickle.dump(train_data, f)
            elif 'val' == dataset:
                pickle.dump(val_data, f)
            else:
                pickle.dump(test_data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset preparation')
    parser.add_argument('-save_dir', '-d', help='Where to save the new datasets')
    parser.add_argument('-data_dir', help='Where to load the datasets')
    parser.add_argument('--ref_data_dir', help='Reference dataset to provide the indices for train/val split', default=None)
    args = parser.parse_args()
    
    extract_driver(args.save_dir, args.data_dir, args.ref_data_dir)
