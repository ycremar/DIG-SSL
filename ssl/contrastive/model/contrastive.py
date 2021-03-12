import sys, torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from ssl.contrastive.objectives import NCE_loss, JSE_loss

class Contrastive(nn.Module):
    def __init__(self, 
                 objective,
                 views_fn,
                 z_dim,
                 graph_level = True,
                 node_level = False,
                 proj = None,
                 proj_n = None,
                 neg_by_crpt = False,
                 device = None):
        """
        Args:
            objective: String or function. If string, should be one of 'NCE' and 'JSE'.
            views_fn: List of functions. Functions to perform view transformation.
            graph_level: Boolean. Whether to include graph-level embedding for contrast.
            node_level: Boolean. Whether to include node-level embedding for contrast.
            proj: String, function or None. Projection head for graph-level representation. 
                If string, should be one of 'linear' and 'MLP'.
            proj_n: String, function or None. Projection head for node-level representations. 
                If string, should be one of 'linear' and 'MLP'. Required when node_level
                is True.
            neg_by_crpt: Boolean. If True, obtain negative samples by performing corruption.
                Otherwise, consider pairs of different graph samples as negative pairs.
                Should always be False when using InfoNCE objective.
        """
        assert node_level is not None or graph_level is not None
        assert not (objective=='NCE' and neg_by_crpt)

        self.loss_fn = self._get_loss(objective)
        self.views_fn = views_fn # fn: (batched) graph -> graph
        self.device = device
        self.node_level = node_level
        self.graph_level = graph_level
        self.neg_by_crpt = neg_by_crpt
        self.z_dim = z_dim
        
        if graph_level and self.proj is not None:
            self.proj_head_g = self._get_proj(proj, z_dim)
            optimizer.add_param_group({"params": self.proj_head_g})
        elif graph_level:
            self.proj_head_g = lambda x: x
        else:
            self.proj_head_g = None
            
        if node_level and self.proj_n is not None:
            self.proj_head_n = self._get_proj(self.proj[1], z_n_dim)
            optimizer.add_param_group([{"params": self.proj_head_n}])
        elif node_level:
            self.proj_head_n = lambda x: x
        else:
            self.proj_head_n = None
        
        
    def train(self, encoder, data_loader, optimizer, epochs):
        """
        Args:
            encoder: Trainable pytorch model or list of models. Callable with inputs (X, edge_index, batch).
                If node_level is False, return tensor of shape [n_graphs, z_dim]. Else, return tuple of
                shape ([n_graphs, z_dim], [n_nodes, z'_dim]) representing graph-level and node-level embeddings.
            dataloader: Dataloader for contrastive training.
            optimizer: Torch optimizer.
            epochs: Integer.
        """
        
        if self.node_level and self.graph_level:
            return self.train_encoder_node_level(encoder, data_loader, optimizer, epochs)
        elif self.graph_level:
            return self.train_encoder_graph(encoder, data_loader, optimizer, epochs)
        else:
            return self.train_encoder_node(encoder, data_loader, optimizer, epochs)

        
    def train_encoder_graph(self, encoder, data_loader, optimizer, epochs):
        
        # output of each encoder should be Tensor for graph-level embedding
        if isinstance(encoder, list):
            assert len(encoder)==len(self.views_fn)
            encoders = encoder
        else:
            encoders = [encoder]*len(self.views_fn)

        [enc.train() for enc in encoders]
        self.proj_head_g.train()
        for epoch in range(epochs):
            for data in data_loader:
                optimizer.zero_grad()
                if None in self.views_fn: 
                    # For view fn that returns multiple views
                    views = []
                    for v_fn in self.views_fn:
                        if v_fn is not None
                        views += [*v_fn(data)]
                else:
                    views = [v_fn(data) for for v_fn in self.views_fn]
                
                zs = []
                for v, enc in zip(views, encoders):
                    z = self._get_embed(enc, view.to(torch.device(self.device)))
                    zs.append(self.proj_head_g(z))

                loss = self.loss_fn(zs, neg_by_crpt=self.neg_by_crpt)
                loss.backward()
                optimizer.step()

        return encoder, self.proj_head

    
    def train_encoder_node(self, encoder, data_loader, optimizer, epochs):
        
        # output of each encoder should be Tensor for node-level embedding
        if isinstance(encoder, list):
            assert len(encoder)==len(self.views_fn)
            encoders = encoder
        else:
            encoders = [encoder]*len(self.views_fn)
        
        [encoder.train() for encoder in encoders]
        self.proj_head_n.train()
        for epoch in epochs:
            for data in data_loader:
                optimizer.zero_grad()
                if None in self.views_fn:
                    # For view fn that returns multiple views
                    views = []
                    for v_fn in self.views_fn:
                        if v_fn is not None
                        views += [*v_fn(data)]
                else:
                    views = [v_fn(data) for for v_fn in self.views_fn]
                
                zs_n = []
                for v, enc in zip(views, encoders):
                    z_n = self._get_embed(enc, view.to(torch.device(self.device)))
                    zs_n.append(self.proj_head_g(z_n))

                loss = self.loss_fn(zs_g=None, zs_n=zs_n, batch=data.batch, 
                                    neg_by_crpt=self.neg_by_crpt)
                loss.backward()
                optimizer.step()
            
        return encoder, self.proj_head_n
    
    
    def train_encoder_node_graph(self, encoder, data_loader, optimizer, epochs):
        
        # output of each encoder should be tuple of (node_embed, graph_embed)
        if isinstance(encoder, list):
            assert len(encoder)==len(self.views_fn)
            encoders = encoder
        else:
            encoders = [encoder]*len(self.views_fn)
        
        [encoder.train() for encoder in encoders]
        self.proj_head_n.train()
        self.proj_head_g.train()
        for epoch in epochs:
            for data in data_loader:
                optimizer.zero_grad()
                if None in self.views_fn:
                    views = []
                    for v_fn in self.views_fn:
                        # For view fn that returns multiple views
                        if v_fn is not None
                        views += [*v_fn(data)]
                    assert len(views)==len(encoders)
                else:
                    views = [v_fn(data) for for v_fn in self.views_fn]
                
                zs_n, zs_g = [], []
                for v, enc in zip(views, encoders):
                    z_g, z_n = self._get_embed(enc, view.to(torch.device(self.device)))
                    zs_n.append(self.proj_head_n(z_n))
                    zs_g.append(self.proj_head_g(z_g))

                loss = self.loss_fn(zs_g, zs_n=zs_n, batch=data.batch, 
                                    neg_by_crpt=self.neg_by_crpt)
                loss.backward()
                optimizer.step()
            
        return encoder, (self.proj_head_n, self.proj_head_g)
    
    def _get_embed(self, enc, view):
        
        if self.neg_by_crpt:
            view_crpt = self._corrupt_graph(view)
            if self.node_level and self.graph_level:
                z_g, z_n = enc(view)
                z_g_crpt, z_n_crpt = enc(view_crpt)
                z = (torch.cat([z_g, z_g_crpt], 0),
                     torch.cat([z_n, z_n_crpt], 0))
            else:
                z = enc(view)
                z_crpt = enc(view_crpt)
                z = torch.cat([z, z_crpt], 0)
        else:
            z = enc(view)
        
        return z
                
    
    def _corrupt_graph(self, view):
        
        data_list = view.to_data_list()
        crpt_list = []
        for data in data_list:
            n_nodes = view.x.shape[0]
            perm = torch.randperm(n_nodes).long()
            crpt_x = view.x[perm]
            crpt_list.append(Data(x=crpt_x, edge_index=view.edge_index))
        view_crpt = Batch.from_data_list(crpt_list)

        return view_crpt
        
    
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