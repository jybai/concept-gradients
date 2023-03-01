import os
import sys
import yaml
import torch
import glob
import numpy as np
import pandas as pd
from time import sleep
from scipy.stats import ttest_ind
from captum.attr import LayerActivation
from captum._utils.gradient import compute_layer_gradients_and_eval
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision.datasets import CelebA, ImageFolder
from torch.utils.data import DataLoader, Dataset
import torchmetrics
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm, trange
import PIL
import seaborn

def save_tcav_results(trials, tcavs, save_npz_fname=None, force=False):
    
    stacked_tcavs = np.stack([np.stack(list(tcavs_.values()), axis=0) for tcavs_ in tcavs], axis=0)
    stacked_accs = np.stack([np.stack(list(trial[1].values()), axis=0)
                             for trial in trials], axis=0)
    
    if save_npz_fname is not None:
        if os.path.exists(save_npz_fname) and not force:
            print(f"{save_npz_fname} already exists.")
        else:
            np.savez(save_npz_fname, tcavs=stacked_tcavs, accs=stacked_accs)
    
    return stacked_tcavs, stacked_accs

class LinearModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.model = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.model(x)
    
class NoReduceMSE():
    def __init__(self):
        self.se = 0
        self.total = 0
    def __call__(self, pred, gt):
        self.se += ((pred - gt)**2).sum(0)
        self.total += pred.shape[0]
    def compute(self):
        return self.se / self.total

class TCAVScore(torchmetrics.Metric):
    def __init__(self, CAV, signed=True, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        assert len(CAV.shape) == 2
        self.CAV = CAV
        
        self.signed = signed
        
        self.add_state("sum", default=torch.zeros([self.CAV.shape[0]]), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, grads: torch.Tensor):
        
        with torch.no_grad():
            assert len(grads.shape) == 2
            assert grads.shape[-1] == self.CAV.shape[-1]

            grads = grads.unsqueeze(1)

            cos = F.cosine_similarity(grads, self.CAV, dim=-1)
            if self.signed:
                score = (cos > 0).sum(0)
            else:
                score = cos.sum(0)

            self.sum += score
            self.total += cos.shape[0]

    def compute(self):
        return self.sum.float() / self.total

class TCAV(nn.Module):
    def __init__(self, target_model, layer_names=None, cache_dir=None):
        super().__init__()
        
        self.target_model = target_model.eval()
        
        self.CAVs = None
        self.random_CAVs = None
        self.metrics = None
        self.cache_dir = cache_dir
        
        assert (layer_names is not None) or (cache_dir is not None)
        
        # reload from cache
        if self.cache_dir is not None and \
            os.path.exists(os.path.join(self.cache_dir, 'random_CAVs.npz')) and \
            os.path.exists(os.path.join(self.cache_dir, 'CAVs.npz')) and \
            os.path.exists(os.path.join(self.cache_dir, 'metrics.npz')):

            print("Loading `random_CAVs.npz`, `CAVs.npz`, and `metrics.npz` from cache...")
            with np.load(os.path.join(self.cache_dir, 'random_CAVs.npz')) as f:
                random_CAVs = {k: v for k, v in f.items()}
            assert all([len(v) > 0 for v in random_CAVs.values()])
            self.random_CAVs = random_CAVs

            with np.load(os.path.join(self.cache_dir, 'CAVs.npz')) as f:
                CAVs = {k: v for k, v in f.items()}
            assert all([len(v) > 0 for v in CAVs.values()])
            self.CAVs = CAVs
            
            with np.load(os.path.join(self.cache_dir, 'metrics.npz')) as f:
                metrics = {k: v for k, v in f.items()}
            assert all([len(v) > 0 for v in metrics.values()])
            self.metrics = metrics

            assert list(self.random_CAVs.keys()) == list(self.CAVs.keys())
            assert list(self.metrics.keys()) == list(self.CAVs.keys())

            self.layer_names = list(self.random_CAVs.keys())
            print(f"Using cached layer names: {self.layer_names}")
        else:
            self.layer_names = layer_names
        
        # searching for layers in target_model
        self.layers = {}
        for name, layer in target_model.named_modules():
            if name in self.layer_names:
                self.layers[name] = layer
        if sorted(self.layer_names) != sorted(list(self.layers.keys())):
            raise ValueError(f"Keys {sorted(self.layer_names)} and {sorted(list(self.layers.keys()))} don't match.")
    
    @staticmethod            
    def get_class_balanced_sampler(ys, y_index):
        ys = ys[:, y_index]
        pos_ratio = ys.sum() / ys.shape[0]
        weights = ys * (1 - pos_ratio) + (1 - ys).abs() * pos_ratio
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        return sampler

    def _generate_CAVs(self, dset_train, dset_valid, hparams=None, verbose=True):
        
        default_hparams = dict(task='classification', n_epochs=100, lr=1e-4, weight_decay=1e-2, 
                               batch_size=128, patience=10, pos_weight=None, num_workers=8)
        
        if hparams is None:
            hparams = default_hparams
        else:
            default_hparams.update(hparams)
            hparams = default_hparams
        
        dl_train = DataLoader(dset_train, batch_size=hparams['batch_size'], drop_last=False, 
                              num_workers=hparams['num_workers'], shuffle=False)
        
        dl_valid = DataLoader(dset_valid, batch_size=hparams['batch_size'], drop_last=False, 
                              num_workers=hparams['num_workers'], shuffle=False)
        
        device = next(self.target_model.parameters())
        cs_train = torch.cat([y.detach().clone() for x, y in tqdm(dl_train, leave=False)], dim=0)
        cs_valid = torch.cat([y.detach().clone() for x, y in tqdm(dl_valid, leave=False)], dim=0)
        
        CAVs, metrics = {}, {}
        
        for layer_name, layer in tqdm(self.layers.items(), leave=False, desc="Layers: "):
            layer_act = LayerActivation(self.target_model, layer)
            
            # extract activations
            acts_train = []
            for x, y in tqdm(dl_train, leave=False):
                act = layer_act.attribute(x.to(device), attribute_to_layer_input=True).flatten(start_dim=1)
                in_dim, out_dim = act.shape[1], y.shape[1]
                acts_train.append(act.detach().clone().cpu())
            acts_train = torch.cat(acts_train, dim=0)
            layer_dset_train = torch.utils.data.TensorDataset(acts_train, cs_train)
            
            acts_valid = []
            for x, y in tqdm(dl_valid, leave=False):
                act = layer_act.attribute(x.to(device), attribute_to_layer_input=True).flatten(start_dim=1)
                in_dim, out_dim = act.shape[1], y.shape[1]
                acts_valid.append(act.detach().clone().cpu())
            acts_valid = torch.cat(acts_valid, dim=0)
            layer_dset_valid = torch.utils.data.TensorDataset(acts_valid, cs_valid)
            
            if hparams['pos_weight'] is not None:
                if isinstance(hparams['pos_weight'], torch.Tensor):
                    pos_weight = hparams['pos_weight'].to(device)
                else:
                    pos_weight = hparams['pos_weight'] * torch.ones([out_dim]).to(device)
            else:
                pos_weight = None

            # sampler = get_class_balanced_sampler(all_cs, nc)
            layer_dl_train = DataLoader(layer_dset_train, batch_size=hparams['batch_size'], drop_last=False, 
                                        num_workers=hparams['num_workers'], shuffle=True)
            layer_dl_valid = DataLoader(layer_dset_valid, batch_size=hparams['batch_size'], drop_last=False, 
                                        num_workers=hparams['num_workers'], shuffle=False)

            # define model and optimizer
            linear_model = LinearModel(in_dim, out_dim).to(device)
            optimizer = optim.Adam(linear_model.parameters(), lr=hparams['lr'], 
                                   weight_decay=hparams['weight_decay'])
            if hparams['task'] == 'classification':
                loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                metric = torchmetrics.Accuracy(threshold=0, num_classes=out_dim, average=None).to(device)
            elif hparams['task'] == 'regression':
                loss_fn = torch.nn.MSELoss()
                metric = NoReduceMSE()
            else:
                raise NotImplementedError

            # train
            patience = 0
            min_loss = np.inf
            linear_model.train()
            with trange(hparams['n_epochs'], leave=False, desc="Epochs: ") as tepochs:
                for epoch in tepochs:
                    losses = []
                    for xs, cs in layer_dl_train:
                        xs = xs.to(device)
                        cs = cs.to(device)
                        optimizer.zero_grad()

                        logits_cs = linear_model(xs)
                        loss = loss_fn(logits_cs, cs.float())

                        loss.backward()
                        optimizer.step()
                        
                        losses.append(loss.item())
                    
                    if min_loss > np.mean(losses):
                        min_loss = np.mean(losses)
                        patience = 0
                    else:
                        patience += 1
                    tepochs.set_postfix(loss=f"{np.mean(losses):.4f}/{min_loss:.4f}")
                    sleep(0.1)
                    if patience > hparams['patience']:
                        tepochs.update(n=hparams['n_epochs'] - epoch)
                        tepochs.close()
                        break

            # eval
            linear_model.eval()
            for xs, cs in layer_dl_valid:
                xs = xs.to(device)
                cs = cs.to(device).to(cs_valid.dtype)
                
                with torch.no_grad():
                    pred_cs = linear_model(xs)
                    metric(pred_cs, cs)
            
            CAV = linear_model.model.weight.detach().clone()
            CAV = CAV / torch.norm(CAV, dim=1, keepdim=True)
            CAVs[layer_name] = CAV.cpu().numpy()
            metrics[layer_name] = metric.compute().detach().cpu().numpy()
            
        return CAVs, metrics
    
    def _generate_random_CAVs(self, dset_train, dset_valid):
        dl = DataLoader(dset_train, batch_size=128, drop_last=False, 
                        num_workers=8, shuffle=False)
        device = next(self.target_model.parameters())
        
        CAVs = {}
        for layer_name, layer in tqdm(self.layers.items(), leave=False):
            layer_act = LayerActivation(self.target_model, layer)
            # get in_dim
            for x, y in dl:
                act = layer_act.attribute(x.to(device), attribute_to_layer_input=True).flatten(start_dim=1)
                in_dim, out_dim = act.shape[1], y.shape[1]
                break
            CAV = (torch.rand(out_dim, in_dim) - 1)
            CAV = CAV / torch.norm(CAV, dim=1, keepdim=True)
            CAVs[layer_name] = CAV.cpu().numpy()
        
        return CAVs
    
    def generate_CAVs(self, dset_train, dset_valid, n_repeat=5, hparams=None, force_rewrite_cache=False):
        
        self.CAVs = {layer_name: [] for layer_name in self.layer_names}
        metrics = {layer_name: [] for layer_name in self.layer_names}
        
        # reload from cache
        if (self.cache_dir is not None) and (not force_rewrite_cache) and \
           (os.path.exists(os.path.join(self.cache_dir, 'CAVs.npz'))) and \
           (os.path.exists(os.path.join(self.cache_dir, 'metrics.npz'))):
            
            raise ValueError("Cached directory already exist. Use `force_rewrite_cache = True` to overwrite.")
            '''
            print("Loading from cache...")
            
            with np.load(os.path.join(self.cache_dir, 'CAVs.npz')) as f:
                self.CAVs.update({k: v for k, v in f.items()})
            with np.load(os.path.join(self.cache_dir, 'metrics.npz')) as f:
                metrics.update({k: v for k, v in f.items()})
            
            if all([len(v) > 0 for v in self.CAVs.values()]):
                return self.CAVs, metrics
            '''
        
        update_layer_names = [k for k, v in self.CAVs.items() if len(v) == 0]
        print(f"Generating TCAV for layers: {update_layer_names}")
        
        # generate
        for _ in trange(n_repeat, desc="#repeats: "):
            CAVs_, metrics_ = self._generate_CAVs(dset_train, dset_valid, hparams=hparams)
            for layer_name in update_layer_names:
                self.CAVs[layer_name].append(CAVs_[layer_name])
                metrics[layer_name].append(metrics_[layer_name])
                
        for layer_name in update_layer_names:
            self.CAVs[layer_name] = np.stack(self.CAVs[layer_name], axis=0)
            metrics[layer_name] = np.stack(metrics[layer_name], axis=0)

        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.savez_compressed(os.path.join(self.cache_dir, 'CAVs.npz'), **self.CAVs)
            np.savez_compressed(os.path.join(self.cache_dir, 'metrics.npz'), **metrics)
            
        return self.CAVs, metrics
    
    def generate_random_CAVs(self, dset_train, dset_valid, n_repeat=5, force_rewrite_cache=False):
        
        random_CAVs = {layer_name: [] for layer_name in self.layer_names}
        
        # reload from cache
        if (self.cache_dir is not None) and (not force_rewrite_cache) and \
           (os.path.exists(os.path.join(self.cache_dir, 'random_CAVs.npz'))):
            
            raise ValueError("Cached directory already exist. Use `force_rewrite_cache = True` to overwrite.")
            
            '''
            print("Loading from cache...")
            
            with np.load(os.path.join(self.cache_dir, 'random_CAVs.npz')) as f:
                random_CAVs.update({k: v for k, v in f.items()})
            
            if all([len(v) > 0 for v in random_CAVs.values()]):
                return random_CAVs
            '''
        
        update_layer_names = [k for k, v in random_CAVs.items() if len(v) == 0]
        print(f"Generating random TCAV for layers: {update_layer_names}")
        
        for _ in trange(n_repeat):
            random_CAVs_ = self._generate_random_CAVs(dset_train, dset_valid)
            for layer_name in update_layer_names:
                random_CAVs[layer_name].append(random_CAVs_[layer_name])
                
        for layer_name in update_layer_names:
            random_CAVs[layer_name] = np.stack(random_CAVs[layer_name], axis=0)

        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True) 
            np.savez_compressed(os.path.join(self.cache_dir, 'random_CAVs.npz'), **random_CAVs)
        
        self.random_CAVs = random_CAVs
        return self.random_CAVs
    
    def generate_TCAVs(self, dset_valid, layer_name, target_index=None, score_signed=True, 
                       return_ttest_results=False, ttest_threshold=0.05):
        
        assert len(self.CAVs[layer_name]) == len(self.random_CAVs[layer_name])
        n_repeat = len(self.CAVs[layer_name])
        
        device = next(self.target_model.parameters())
        tcavs, random_tcavs = [], []
        
        concept_dl = DataLoader(dset_valid, batch_size=16, shuffle=False, 
                                drop_last=False, num_workers=8)
        
        for i in trange(n_repeat, leave=False):
            
            CAV = torch.from_numpy(self.CAVs[layer_name][i]).float().to(device)
            random_CAV = torch.from_numpy(self.random_CAVs[layer_name][i]).float().to(device)

            tcavs_ = TCAVScore(CAV, signed=score_signed).to(device)
            random_tcavs_ = TCAVScore(random_CAV, signed=score_signed).to(device)

            for xs, cs in concept_dl:
                xs = xs.to(device)
                cs = cs.to(device)

                layer_grads_, _ = compute_layer_gradients_and_eval(
                    self.target_model, layer, xs, target_ind=target_index, 
                    attribute_to_layer_input=True)
                del _
                layer_grads_ = layer_grads_[0].flatten(start_dim=1)

                tcavs_(layer_grads_)
                random_tcavs_(layer_grads_)

            tcavs.append(tcavs_.compute().detach().cpu().numpy())
            random_tcavs.append(random_tcavs_.compute().detach().cpu().numpy())
        
        random_tcavs = np.stack(random_tcavs, axis=0)
        tcavs = np.stack(tcavs, axis=0)
        
        print('random_tcavs:', random_tcavs.mean(0))
        print('tcavs:', tcavs.mean(0))
        
        # run two-sided test
        ttest_results = []
        for i in range(tcavs.shape[1]):
            ttest_result = ttest_ind(tcavs[:, i], random_tcavs[:, i])
            ttest_results.append(ttest_result.pvalue)
        ttest_results = np.array(ttest_results)
        
        avg_tcav_scores = tcavs.mean(0)
        avg_tcav_scores[~(ttest_results < ttest_threshold)] = np.nan
        
        if return_ttest_results:
            return avg_tcav_scores, ttest_results
        else:
            return avg_tcav_scores

    def attribute(self, inputs, layer_name, mode, target=None, abs=False, use_random=False, select_index=None):
        
        assert mode in ['inner_prod', 'cosine_similarity']
        
        
        if use_random:
            assert self.random_CAVs is not None
            
            if select_index is None:
                CAV_ = self.random_CAVs[layer_name].mean(0)
            else:
                CAV_ = self.random_CAVs[layer_name][select_index]
        else:
            assert self.CAVs is not None
            
            if select_index is None:
                CAV_ = self.CAVs[layer_name].mean(0)
            else:
                CAV_ = self.CAVs[layer_name][select_index]
        CAV = torch.from_numpy(CAV_).float().to(inputs.device)
        
        with torch.no_grad():
            grads, _ = compute_layer_gradients_and_eval(
                self.target_model, self.layers[layer_name], inputs, 
                target_ind=target, attribute_to_layer_input=True)
            del _
            grads = grads[0].flatten(start_dim=1)
            
            if mode == 'inner_prod':
                attributions = grads @ CAV.T
            elif mode == 'cosine_similarity':
                attributions = grads @ CAV.T / (torch.norm(grads, dim=1, keepdims=True) * \
                                                torch.norm(CAV.T, dim=0, keepdims=True))
            else:
                raise NotImplementedError
            
            if abs:
                attributions = torch.abs(attributions)
                
        return attributions
