import torch
import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, in_dim, out_dim, p_drop=0.5, squeeze_final=False):
        super().__init__()
        self.squeeze_final = squeeze_final
        self.head = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=p_drop),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )
    def forward(self, x):
        x = self.head(x)
        if self.squeeze_final:
            x = x.squeeze(-1)
        return x
