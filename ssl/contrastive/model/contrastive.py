import sys, torch
import torch.nn as nn
from ssl.contrastive.views_fn import *
from ssl.contrastive.objectives import NCE_loss, JSE_loss

class Contrastive():
    def __init__(self, 
                 objective,
                 views_fn,
                 optimizer,
                 proj = None,
                 dim = None,
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
        self.views_fn = views_fn # fn: graph -> graph
        self.device = device
        self.proj = proj
        
        
    def train(self, encoders, data_loader):
        """
        Args:
            encoder: Trainable pytorch model or list of models. Callable with inputs (X, edge_index, batch).
                    If node2graph is False, return tensor of shape [n_batch, z_dim]. Else, return tuple of
                    shape ([n_batch, z_dim], [n_batch, z'_dim]) representing graph-level and node-level embeddings.
            dataloader: Dataloader.
        """
        
        if isinstance(encoders, list):
            assert len(encoders)==len(self.views_fn)
        else:
            encoders = [encoders]*len(self.views_fn)
        
        if node2graph:
            return self.train_encoder_node2graph(encoders, data_loader)
        else:
            return self.train_encoder(encoders, data_loader)

        
    def train_encoder(self, encoders, data_loader):
        
        data = list(data_loader)[0].to(torch.device(self.device))
        z_dim = encoders[0](data).shape[1]
        
        if self.proj is not None:
            self.proj_head = self._get_proj(self.proj, z_dim)
            self.optimizer.add_param_group({"params": self.proj_head})
        else:
            self.proj_head = lambda x: x

        [encoder.train() for encoder in encoders]
        self.proj_head.train()
        for data in data_loader:
            self.optimizer.zero_grad()
            zs = []
            for v_fn, encoder in zip(self.views_fn, encoders):
                view = v_fn(data)
                z = encoder(view)
                zs.append(self.proj_head(z))
            
            loss = self.loss_fn(zs)
            loss.backward()
            self.optimizer.step()

        return encoder, self.proj_head

    
    def train_encoder_node2graph(self, encoders, data_loader, sigma):
        
        # output of each encoder should be tuple of (node_embed, graph_embed)
        data = list(data_loader)[0].to(torch.device(self.device))
        z_n, z_g = encoders[0](data)
        z_n_dim, z_g_dim = z_n.shape[1], z_g.shape[1]
        
        if self.proj is not None:
            self.proj_head_n = self._get_proj(self.proj, z_n_dim)
            self.proj_head_g = self._get_proj(self.proj, z_g_dim)
            self.optimizer.add_param_group([{"params": self.proj_head_n}, 
                                            {"params": self.proj_head_g}])
        else:
            self.proj_head_n = lambda x: x
            self.proj_head_g = lambda x: x

        [encoder.train() for encoder in encoders]
        self.proj_head_n.train()
        self.proj_head_g.train()
        for data in data_loader:
            self.optimizer.zero_grad()
            zs_n, zs_g = [], []
            for v_fn, encoder in zip(self.views_fn, encoders):            
                view = v_fn(data)
                z_n, z_g = encoder(view)
                zs_n.append(self.proj_head_n(z_n))
                zs_g.append(self.proj_head_g(z_g))
            
            loss = self.loss_fn(zs_g, zs_n=zs_n)
            loss.backward()
            self.optimizer.step()

        return encoder, (self.proj_head_n, self.proj_head_g)
    
    
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