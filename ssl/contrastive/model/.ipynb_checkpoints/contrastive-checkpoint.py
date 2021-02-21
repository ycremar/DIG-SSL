import sys
sys.path.append('..')
sys.path.append('../../utils')

import pytorch
from views_fn import *
from objectives import NCE, JSE

class Contrastive():
    def __init__(self, 
                 objective, 
                 proj_head,
                 views_fn,
                 optimizer):
        """
        Args:
            objective: String or function. If string, should be one of 'NCE' and 'JSE'.
            proj_head: String or function. If string, should be one of 'linear' and 'MLP'.
            views_fn: List of functions. 
            optimizer: Pytorch optimizer object.
        """

        self.objective = objective # NCE, JSE, DV, ...
        self.proj_head = proj_head # Linear, MLP, ...
        self.views_fn = views_fn # fn: graph -> graph
        self.optimizer = optimizer

    def train(self, encoder, data_loader):
        """
        Args:
            encoder: Pytorch module. Callable with inputs (X, edge_index, batch).
            dataloader: Dataloader.
        """

        proj = self._get_proj(self.proj_head)
        loss_fn = self._get_loss(self.objective)

        encoder.train()
        proj.train()
        for x in data_loader:
            views = [v_fn(x) for v_fn in self.views_fn]
            zs = [encoder(*view) for view in views]
            zs = [proj(z) for z in zs]
            loss = loss_fn(zs)
            loss.backward()

        return encoder
    
    def _get_proj(self, proj_head):
        
        if callable(proj_head):
            return proj_head
        
        