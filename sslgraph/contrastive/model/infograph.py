import sys, torch
import torch.nn as nn
from .contrastive import Contrastive


class ProjHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)
    

class InfoGraph(Contrastive):
    
    def __init__(self, dim):
        '''
        Args:
            diffusion_type: String. Diffusion instantiation mode with two options:
                            'ppr': Personalized PageRank
                            'heat': heat kernel
            alpha: Float in (0,1). Teleport probability in a random walk.
            t: Integer. Diffusion time.
        '''
        views_fn = [lambda x: x]
        proj = ProjHead(dim)
        proj_n = ProjHead(dim)
        super(InfoGraph, self).__init__(objective='JSE',
                                      views_fn=views_fn,
                                      node_level=True,
                                      proj=proj,
                                      proj_n=proj_n,
                                      device=device)
        
    def train(self, encoders, data_loader, optimizer, epochs):
        encs, (proj, proj_n) = super().train(self, encoders, data_loader, optimizer, epochs)
        
        return encoder