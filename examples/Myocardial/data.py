import os
import json
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

class MIComplications(Dataset):
    def __init__(self, root_dir, return_attribute=True):
        self.return_attribute = return_attribute
        data_path = os.path.join(root_dir, 'myocardial/data.npz')
        
        if not os.path.isfile(data_path):
            preprocess_and_cache(root_dir)

        with np.load(data_path, allow_pickle=True) as f:
            self.X = torch.from_numpy(f['X']).float()
            self.C = torch.from_numpy(f['C']).int()
            self.Y = torch.from_numpy(f['Y']).int()
            self.X_labels = f['feat_labels']
            self.C_labels = f['concept_labels']
            self.Y_label = f['target_label']
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):

        if self.return_attribute:
            return self.X[idx], self.C[idx]
        else:
            return self.X[idx], self.Y[idx]
    
    def preprocess_and_cache(self, root_dir):
        raw_path = os.path.join(root_dir, 'myocardial/myocardial_infarction_complicatiions_database.csv')
        df = pd.read_csv(raw_path)
        df_X = df.iloc[:, :112]
        df_C = df.iloc[:, 112:123]
        df_Y = df.iloc[:, 123]
        
        # remove features with NaN rate greater then t
        t = 0.25
        df_X = df_X.drop(labels=[col for col in df_X.columns 
                                 if df_X[col].isna().mean() > t], axis=1)
        df_X = df_X.drop('ID', axis=1)
        
        # fill NaN with mode
        df_X = df_X.fillna(df_X.mean().iloc[0])
        # normalize
        df_X = (df_X - df_X.mean(0)) / df_X.std(0)
        
        df_Y = (df_Y != 0).astype(int)
        
        np.savez(os.path.join(root_dir, 'myocardial/data.npz'), 
                 X=df_X.values, feat_labels=df_X.columns, 
                 C=df_C.values, concept_labels=df_C.columns,
                 Y=df_Y.values, target_label=df_Y.name)
