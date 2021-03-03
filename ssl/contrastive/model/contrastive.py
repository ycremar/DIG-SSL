import sys, torch
import torch.nn as nn
from ssl.contrastive.views_fn import *
from ssl.contrastive.objectives import NCE_loss, JSE_loss

class Contrastive():
    def __init__(self, 
                 objective,
                 views_fn,
                 node_level = False,
                 proj = None,
                 proj_n = None,
                 device = None):
        """
        Args:
            objective: String or function. If string, should be one of 'NCE' and 'JSE'.
            views_fn: List of functions. Functions to perform view transformation.
            node_level: Boolean. Whether to perform nodel level contrast.
            proj: String, function or None. Projection head for graph-level representation. 
                  If string, should be one of 'linear' and 'MLP'.
            proj_n: String, function or None. Projection head for node-level representations. 
                    If string, should be one of 'linear' and 'MLP'. Required when node_level
                    is True.
        """

        self.loss_fn = self._get_loss(objective)
        self.views_fn = views_fn # fn: (batched) graph -> graph
        self.device = device
        self.proj = proj
        self.node_level = node_level
        
        
    def train(self, encoders, data_loader, optimizer, epochs):
        """
        Args:
            encoder: Trainable pytorch model or list of models. Callable with inputs (X, edge_index, batch).
                    If node_level is False, return tensor of shape [n_graphs, z_dim]. Else, return tuple of
                    shape ([n_graphs, z_dim], [n_nodes, z'_dim]) representing graph-level and node-level embeddings.
            dataloader: Dataloader.
        """
        
        if isinstance(encoders, list):
            assert len(encoders)==len(self.views_fn)
            single_enc = False
        else:
            single_enc = True
            encoders = [encoders]*len(self.views_fn)
        
        if self.node_level:
            return self.train_encoder_node_level(single_enc, encoders, data_loader, optimizer, epochs)
        else:
            return self.train_encoder(single_enc, encoders, data_loader, optimizer, epochs)

        
    def train_encoder(self, single_enc, encoders, data_loader, optimizer, epochs):
        
        data = list(data_loader)[0].to(torch.device(self.device))
        z_dim = encoders[0](data).shape[1]
        
        if self.proj is not None:
            self.proj_head = self._get_proj(self.proj, z_dim)
            optimizer.add_param_group({"params": self.proj_head})
        else:
            self.proj_head = lambda x: x

        [encoder.train() for encoder in encoders]
        self.proj_head.train()
        for epoch in range(epochs):
            for data in data_loader:
                optimizer.zero_grad()
                zs = []
                for v_fn, encoder in zip(self.views_fn, encoders):
                    view = v_fn(data).to(torch.device(self.device))
                    z = encoder(view)
                    zs.append(self.proj_head(z))

                loss = self.loss_fn(zs)
                loss.backward()
                optimizer.step()
                
        if single_enc:
            encoders = encoders[0]

        return encoders, self.proj_head

    
    def train_encoder_node_level(self, single_enc, encoders, data_loader, optimizer, epochs):
        
        # output of each encoder should be tuple of (node_embed, graph_embed)
        data = list(data_loader)[0].to(torch.device(self.device))
        z_n, z_g = encoders[0](data)
        z_n_dim, z_g_dim = z_n.shape[1], z_g.shape[1]
        
        if self.proj is not None:
            self.proj_head_g = self._get_proj(self.proj[0], z_g_dim)
            optimizer.add_param_group([{"params": self.proj_head_g}])
        else:
            self.proj_head_g = lambda x: x
            
        if self.proj_n is not None:
            self.proj_head_n = self._get_proj(self.proj[1], z_n_dim)
            optimizer.add_param_group([{"params": self.proj_head_n}])
        else:
            self.proj_head_n = lambda x: x

        [encoder.train() for encoder in encoders]
        self.proj_head_n.train()
        self.proj_head_g.train()
        for epoch in epochs:
            for data in data_loader:
                optimizer.zero_grad()
                zs_n, zs_g = [], []
                for v_fn, encoder in zip(self.views_fn, encoders):            
                    view = v_fn(data).to(torch.device(self.device))
                    z_g, z_n = encoder(view)
                    zs_n.append(self.proj_head_n(z_n))
                    zs_g.append(self.proj_head_g(z_g))

                loss = self.loss_fn(zs_g, zs_n=zs_n, batch=data.batch)
                loss.backward()
                optimizer.step()

        if single_enc:
            encoders = encoders[0]
            
        return encoders, (self.proj_head_n, self.proj_head_g)
    
    
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