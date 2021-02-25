import torch
import torch.nn.functional as F
import itertools


def JSE_loss(zs, zs_n=None, batch=None, sigma=None):
    '''
    Args:
        zs: List of tensors of shape [n_views, batch_size, z_dim].
        zs_n: List of tensors of shape [n_views, nodes, z_dim].
        sigma: 2D-array of shape [n_views, n_views] with boolean values.
            Only required when n_views > 2. If sigma_ij = True, then compute
            infoNCE between view_i and view_j.
    '''
    if zs_n is not None:
        assert len(zs_n) == len(zs)
        assert batch is not None
        if len(zs) == 1:
            return JSE_local_global(zs_n[0], zs[0], batch)
        elif len(zs) == 2:
            return (JSE_local_global(zs_n[0], zs[1], batch) +
                    JSE_local_global(zs_n[1], zs[0], batch))
        else:
            assert len(zs) == len(sigma)
            loss = 0
            for (i, j) in itertools.combinations(range(len(zs)), 2):
                if sigma[i][j]:
                    loss += (JSE_local_global(zs_n[i], zs[j], batch) +
                             JSE_local_global(zs_n[j], zs[i], batch))
            return loss

    if len(zs) == 2:
        return JSE_global_global(zs[0], zs[1])
    elif len(zs) > 2:
        assert len(zs) == len(sigma)
        loss = 0
        for (i, j) in itertools.combinations(range(len(zs)), 2):
            if sigma[i][j]:
                loss += JSE_global_global(zs[i], zs[j])
        return loss


def JSE_local_global(z_n, z_g, batch):
    '''
    Args:
        z_n: Tensor of shape [n_nodes, z_dim].
        z_g: Tensor of shape [n_graphs, z_dim].
        batch: Tensor of shape [n_graphs].
    '''
    num_graphs = z_g.shape[0]
    num_nodes = z_n.shape[0]

    pos_mask = torch.zeros((num_nodes, num_graphs))
    neg_mask = torch.ones((num_nodes, num_graphs))
    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.

    d_prime = torch.matmul(z_n, z_g)

    E_pos = get_expectation(d_prime * pos_mask, positive=True).sum()
    E_pos = E_pos / num_nodes
    E_neg = get_expectation(d_prime * neg_mask, positive=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))
    return E_neg - E_pos


def JSE_global_global(z1, z2):
    '''
    Args:
        z1, z2: Tensor of shape [batch_size, z_dim].
    '''
    num_graphs = z1.shape[0]

    pos_mask = torch.zeros((num_graphs, num_graphs))
    neg_mask = torch.ones((num_graphs, num_graphs))
    for graphidx in range(num_graphs):
        pos_mask[graphidx][graphidx] = 1.
        neg_mask[graphidx][graphidx] = 0.

    d_prime = torch.matmul(z1, z2)

    E_pos = get_expectation(d_prime * pos_mask, positive=True).sum()
    E_pos = E_pos / num_graphs
    E_neg = get_expectation(d_prime * neg_mask, positive=False).sum()
    E_neg = E_neg / (num_graphs * (num_graphs - 1))
    return E_neg - E_pos


def get_expectation(masked_d_prime, positive=True):
    '''
    Args:
        masked_d_prime: Tensor of shape [n_graphs, n_graphs] for global_global,
                        tensor of shape [n_nodes, n_graphs] for local_global.
        positive (bool): Set True if the d_prime is masked for positive pairs,
                        set False for negative pairs.
    '''
    if positive:
        score = - F.softplus(-masked_d_prime)
    else:
        score = F.softplus(-masked_d_prime) + masked_d_prime
    return score

