import sys, torch
import torch.nn as nn
from .contrastive import Contrastive
from sslgraph.contrastive.views_fn import node_attr_mask, edge_perturbation, combine

class GRACE(Contrastive):
    
    def __init__(self, dim, dropE_rate_1, dropE_rate_2, maskN_rate_1, maskN_rate_2, device=None):
        '''
        dim: Integer. Embedding dimension.
        aug1, aug2: String. Should be in ['dropN', 'permE', 'subgraph', 
                    'maskN', 'random2', 'random3', 'random4'].
        aug_ratio: Float between (0,1).
        '''
        view_fn_1 = combine([edge_perturbation(ratio=dropE_rate_1),
                          node_attr_mask(mask_ratio=maskN_rate_1)])
        view_fn_2 = combine([edge_perturbation(ratio=dropE_rate_2),
                          node_attr_mask(mask_ratio=maskN_rate_2)])
        views_fn = [view_fn_1, view_fn_2]
        
        super(GraphCL, self).__init__(objective='NCE',
                                      views_fn=views_fn,
                                      dim=dim,
                                      proj='MLP',
                                      graph_level=False,
                                      node_level=True,
                                      device=device)
        
    def train(self, encoders, data_loader, optimizer, epochs):
        # GRACE removes projection heads after pre-training
        for enc, proj in super().train(self, encoders, data_loader, optimizer, epochs):
            yield enc