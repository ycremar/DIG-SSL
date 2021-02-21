import torch
import numpy as np
import itertools

def NCE_loss(zs, sigma=None, args=None):
    '''
    Args:
        zs: List of tensors of shape [batch_size, z_dim].
        sigma: 2D-array of shape [n_views, n_views] with boolean values.
            Only required when n_views > 2. If sigma_ij = True, then compute
            infoNCE between view_i and view_j.
    '''
    if args is None:
        tau = 0.5
        norm = True
    else:
        tau = args.tau
        norm = args.norm
    
    if len(zs)==2:
        return infoNCE(zs[0], zs[1], tau, norm)
    elif len(zs)>2:
        assert sigma is not None
        assert len(zs)==len(sigma)
        loss = 0
        for (i, j) in itertools.combinations(range(len(zs)), 2):
            if sigma[i][j]:
                loss += infoNCE(zs[i], zs[j], tau, norm)


def infoNCE(z1, z2, tau=0.5, norm=True):
    '''
    Args:
        z1, z2: Tensor of shape [batch_size, z_dim]
        tau: Float. Usually in (0,1].
        norm: Boolean. Whether to apply 
    '''
    
    batch_size, _ = x.size()
    sim_matrix = torch.einsum('ik,jk->ij', z1, z2)
    
    if norm:
        z1_abs = z1.norm(dim=1)
        z2_abs = z2.norm(dim=1)
        sim_matrix = sim_matrix / torch.einsum('i,j->ij', z1_abs, z2_abs)
        
    sim_matrix = torch.exp(sim_matrix / tau)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()
    return loss