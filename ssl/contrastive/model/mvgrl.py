import sys, torch
import torch.nn as nn
from .contrastive import Contrastive
from ssl.contrastive.views_fn import diffusion


class MVGRL_enc(nn.Module):
    '''
        MVGRL includes projection heads and combines two views and encoders
        when inferencing representation
    '''
    def __init__(self, encoder_0, encoder_1, proj, proj_n, views_fn):
        super(MVGRL_enc, self).__init__()
        self.encoder_0 = encoder_0
        self.encoder_1 = encoder_1
        self.proj = proj
        self.proj_n = proj_n
        self.views_fn = views_fn
        
    def forward(self, data):
        zg_1, zn_1 = self.encoder_0(views_fn[0](data))
        zg_1 = self.proj(zg_1)
        zn_1 = self.proj_n(zn_1)
        
        zg_2, zn_2 = self.encoder_1(views_fn[1](data))
        zg_2 = self.proj(zg_2)
        zn_2 = self.proj_n(zn_2)
        
        return (zg_1 + zg_2), (zn_1 + zn_2)

    
class MVGRL(Contrastive):
    
    def __init__(self, diffusion_type='ppr', alpha, t):        
        '''
        Args:
            diffusion_type: String. Diffusion instantiation mode with two options:
                            'ppr': Personalized PageRank
                            'heat': heat kernel
            alpha: Float in (0,1). Teleport probability in a random walk.
            t: Integer. Diffusion time.
        '''
        views_fn = [lambda x: x,
                    diffusion(mode=diffusion_type, alpha=alpha, t=t)
                   ]
        super(GraphCL, self).__init__(objective='JSE',
                                      views_fn=views_fn,
                                      node_level=True,
                                      proj='MLP',
                                      proj_n='MLP',
                                      device=device)
        
    def train(self, encoders, data_loader, optimizer, epochs):
        encs, (proj, proj_n) = super().train(self, encoders, data_loader, optimizer, epochs)
        encoder = MVGRL_enc(encs[0], enc[1], proj, proj_n, self.view_fn)
        
        return encoder