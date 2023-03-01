import argparse
import os
import numpy as np
from tqdm.auto import tqdm, trange
import cv2 # somehow importing this first before `pytorch_lightning` avoids crashing

import torch
from torch import optim
from torch.utils.data import DataLoader

import torchmetrics
from pytorch_lightning.loggers.csv_logs import CSVLogger

from model import PspDethead
from data import ChestX

def get_pos_weights(dset):
    lbls = []
    for xs, ys, in dset:
        lbls.append(ys.detach().clone())
    pos_ratios = torch.cat(lbls, dim=0).float().mean(dim=0)
    pos_weights = (1. - pos_ratios) / pos_ratios
    return pos_weights

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, help='Path to save model checkpoint and logs.')
    parser.add_argument('--load_ckpt_path', type=str, help='Path to pretrained pkl path.', default=None)
    parser.add_argument('--freeze_pretrained', action='store_true') # default to False

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-2)

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--bsize', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--data_dir', type=str, default='/home/andrewbai/data')

    parser.add_argument('--dry_run', action='store_true') # default to False

    return parser.parse_args()

def train():
    pass

def eval():
    pass

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_arguments()

    if not args.dry_run:
        os.makedirs(args.save_path, exist_ok=True)
        logger = CSVLogger(save_dir=os.path.dirname(args.save_path), name=os.path.basename(args.save_path), version='./')
        logger.log_hyperparams(vars(args))

    dset_trn = ChestX('/home/andrewbai/data/', 'trn')
    dset_tst = ChestX('/home/andrewbai/data/', 'tst')

    dl_trn = DataLoader(dset_trn, batch_size=args.bsize, shuffle=True, drop_last=True, num_workers=args.num_workers)
    dl_tst = DataLoader(dset_tst, batch_size=args.bsize, shuffle=False, drop_last=False, num_workers=args.num_workers)

    model = PspDethead()

    if args.load_ckpt_path is not None:
        model.load_from_pspnet(args.load_ckpt_path)

        if args.freeze_pretrained:

            for param in model.parameters():
                param.requires_grad = False

            model.classification.weight.requires_grad = True
            model.classification.bias.requires_grad = True

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=args.lr, weight_decay=args.weight_decay)

    pos_weights = get_pos_weights(dset_trn)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    best_tst_f1 = 0
    for epoch in trange(1, args.epochs + 1):

        trn_loss = 0
        model.train()
        for xs, ys in tqdm(dl_trn, leave=False):
            optimizer.zero_grad()

            xs = xs.to(device)
            ys = ys.to(device)

            pred_ys = model(xs)
            loss = loss_fn(pred_ys, ys.float())
            loss.backward()
            optimizer.step()

            trn_loss += loss.item()
        trn_loss /= len(dl_trn)

        tst_loss = 0
        tst_acc = torchmetrics.Accuracy().to(device)
        tst_f1 = torchmetrics.classification.MultilabelF1Score(num_labels=13).to(device)

        model.eval()
        with torch.no_grad():
            for xs, ys in tqdm(dl_tst, leave=False):
                xs = xs.to(device)
                ys = ys.to(device)

                pred_ys = model(xs)
                loss = loss_fn(pred_ys, ys.float())
                tst_loss += loss.item()
                tst_acc(torch.sigmoid(pred_ys), ys)
                tst_f1(torch.sigmoid(pred_ys), ys)
            tst_loss /= len(dl_tst)
            tst_acc = tst_acc.compute().item()
            tst_f1 = tst_f1.compute().item()

        print(f"Epoch {epoch}: trn_loss={trn_loss:.4f}, tst_loss={tst_loss:.4f}, tst_acc={tst_acc:.4f}, tst_f1={tst_f1:.4f}")

        if args.dry_run:
            break

        logger.log_metrics({'trn_loss': trn_loss, 'tst_loss': tst_loss, 'tst_acc': tst_acc, 'tst_f1': tst_f1}, step=epoch * len(dl_trn))
        logger.save()

        if best_tst_f1 < tst_f1:
            best_tst_f1 = tst_f1
            torch.save(model.state_dict(), os.path.join(args.save_path, 'best_f1_model.ckpt'))

if __name__ == '__main__':
    main()

