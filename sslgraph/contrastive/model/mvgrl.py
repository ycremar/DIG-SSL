import sys, torch
import torch.nn as nn
from .contrastive import Contrastive
from sslgraph.contrastive.views_fn import diffusion, diffusion_with_sample


class MVGRL_enc(nn.Module):
    '''
        MVGRL includes projection heads and combines two views and encoders
        when inferencing graph-level representation.
    '''
    def __init__(self, encoder_0, encoder_1, 
                 proj, proj_n, views_fn, 
                 graph_level=True, node_level=True):
        
        super(MVGRL_enc, self).__init__()
        self.encoder_0 = encoder_0
        self.encoder_1 = encoder_1
        self.proj = proj
        self.proj_n = proj_n
        self.views_fn = views_fn
        self.graph_level = graph_level
        self.node_level = node_level
        
    def forward(self, data):
        zg_1, zn_1 = self.encoder_0(views_fn[0](data))
        zg_1 = self.proj(zg_1)
        zn_1 = self.proj_n(zn_1)
        
        zg_2, zn_2 = self.encoder_1(views_fn[1](data))
        zg_2 = self.proj(zg_2)
        zn_2 = self.proj_n(zn_2)
        
        if self.graph_level and self.node_level:
            return (zg_1 + zg_2), (zn_1 + zn_2)
        elif self.graph_level:
            return zg_1 + zg_2
        elif self.node_level:
            return zn_1 + zn_2
        else:
            return None

    
class MVGRL(Contrastive):
    
    def __init__(self, dim, diffusion_type='ppr', alpha=None, t=None, 
                 graph_level_output=True, node_level_output=False):
        '''
        Args:
            diffusion_type: String. Diffusion instantiation mode with two options:
                'ppr': Personalized PageRank
                'heat': heat kernel
            alpha: Float in (0,1). Teleport probability in a random walk.
            t: Integer. Diffusion time.
            subgraph: Boolean. Whether to sample subgraph from a large graph. 
                Set to True for node-level tasks on large graphs.
        '''
        views_fn = [lambda x: x,
                    diffusion(mode=diffusion_type, alpha=alpha, t=t)]
        self.graph_level = graph_level
        self.node_level = node_level
        super(MVGRL, self).__init__(objective='JSE',
                                      views_fn=views_fn,
                                      node_level=True,
                                      dim=dim,
                                      proj='MLP',
                                      proj_n='MLP',
                                      device=device)
        
    def train(self, encoders, data_loader, optimizer, epochs):
        encs, (proj, proj_n) = super(MVGRL, self).train(encoders, data_loader, optimizer, epochs)
        encoder = MVGRL_enc(encs[0], enc[1], proj, proj_n, self.view_fn, self.graph_level, self.node_level)
        
        return encoder
    
    
    
class NodeMVGRL(Contrastive):
    
    def __init__(self, dim, diffusion_type='ppr', alpha=None, t=None, 
                 graph_level_output=False, node_level_output=True):
        '''
        Args:
            diffusion_type: String. Diffusion instantiation mode with two options:
                'ppr': Personalized PageRank
                'heat': heat kernel
            alpha: Float in (0,1). Teleport probability in a random walk.
            t: Integer. Diffusion time.
            subgraph: Boolean. Whether to sample subgraph from a large graph. 
                Set to True for node-level tasks on large graphs.
        '''
        views_gn = [diffusion_with_sample, None]
        self.graph_level = graph_level
        self.node_level = node_level
        
        super(MVGRL, self).__init__(objective='JSE',
                                    views_fn=views_fn,
                                    node_level=True,
                                    dim=dim,
                                    proj=nn.Sigmoid(),
                                    proj_n='Linear',
                                    neg_by_crpt=True,
                                    device=device)
        
    def train(self, encoders, data_loader, optimizer, epochs):
        encs, (proj, proj_n) = super(MVGRL, self).train(encoders, data_loader, optimizer, epochs)
        encoder = MVGRL_enc(encs[0], enc[1], (lambda x: x), (lambda x: x), 
                            self.view_fn, self.graph_level, self.node_level)
        
        return encoder