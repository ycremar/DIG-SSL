import sys, torch
import torch.nn as nn
from ssl.contrastive.views_fn import *
from ssl.contrastive.objectives import NCE_loss, JSE_loss

class Contrastive(nn.Module):
    def __init__(self, 
                 objective,
                 views_fn,
                 optimizer,
                 proj_head = 'MLP',
                 node2graph = False,
                 device = None):
        """
        Args:
            objective: String or function. If string, should be one of 'NCE' and 'JSE'.
            proj_head: String, function or None. If string, should be one of 'linear' and 'MLP'.
            views_fn: List of functions. 
            optimizer: Pytorch optimizer object.
        """

        self.loss_fn = self._get_loss(objective)
        self.optimizer = optimizer
        
        if proj is not None:
            self.proj_head = self._get_proj(self.proj_head, z_dim)
            self.optimizer.add_param_group({"params": self.proj_head})
        else:
            self.proj_head = lambda x: x
        
        self.views_fn = views_fn # fn: graph -> graph
        self.device = device
        
    def train(self, encoders, data_loader):
        """
        Args:
            encoder: Trainable pytorch model or list of models. Callable with inputs (X, edge_index, batch).
            dataloader: Dataloader.
        """
        
        if isinstance(encoders, list):
            assert len(encoders)==len(self.views_fn)
            return self.train_mul_encoders(encoders, data_loader)
        elif node2graph:
            pass
        else:
            return self.train_single_encoder(encoders, data_loader)

    def train_single_encoder(self, encoder, data_loader):
        
        data = list(data_loader)[0].to(torch.device(self.device))
        _, z_dim = encoder(data).size()

        encoder.train()
        self.proj_head.train()
        for data in data_loader:
            self.optimizer.zero_grad()
            
            views = [v_fn(data) for v_fn in self.views_fn]
            zs = [encoder(*view) for view in views]
            zs = [self.proj_head(z) for z in zs]
            
            loss = self.loss_fn(zs)
            loss.backward()
            self.optimizer.step()

        return encoder
    
    def train_mul_encoders(self, encoders, data_loader):
        pass
    
    def _get_proj(self, proj_head, z_dim):
        
        if callable(proj_head):
            return proj_head
        
        assert proj_head in ['linear', 'MLP']
        
        if proj_head == 'linear':
            return nn.Linear(z_dim, z_dim)
        elif proj_head == 'MLP':
            return nn.Sequential(nn.Linear(z_dim, z_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(z_dim, z_dim))
        
    def _get_loss(self, objective):
        
        if callable(objective):
            return objective
        
        assert objective in ['JSE', 'NCE']
        
        return {'JSE':JSE_loss, 'NCE':NCE_loss}[objective]