import torch
import numpy as np
import random


def node_attr_mask(data, mode='whole', mask_ratio=0.1, mask_mean=0.5, mask_std=0.5):
    '''
    Args:
        data: A graph data object containing:
                original x tensor with shape [num_nodes, num_node_features];
                y tensor with arbitrary shape;
                edge_attr tensor with shape [num_edges, num_edge_features];
                edge_index tensor with shape [2, num_edges].
        mode: Masking mode with three options:
                whole masking: mask all feature dimensions of the selected node with a Gaussian distribution;
                partial masking: mask only selected feature dimensions with a Gaussian distribution;
                onehot masking: mask all feature dimensions of the selected node with a one-hot vector.
        mask_ratio: Percentage of masking feature dimensions.
        mask_mean: Mean of the Gaussian distribution.
        mask_std: Standard deviation of the distribution. Must be non-negative.

    Returns:
        x tensor with shape [num_nodes, num_node_features];
        edge_index tensor with shape [2, num_edges];
        batch tensor with shape [num_nodes].
    '''
    node_num, feat_dim = data.x.size()
    x = data.x

    if mode == 'whole':
        mask_num = int(node_num * mask_ratio)
        idx_mask = np.random.choice(node_num, mask_num, replace=False)
        x[idx_mask] = torch.tensor(np.random.normal(loc=mask_mean, scale=mask_std, size=(mask_num, feat_dim)), dtype=torch.float32)

    elif mode == 'partial':
        for i in range(node_num):
            for j in range(feat_dim):
                if random.random() < mask_ratio:
                    x[i][j] = torch.tensor(np.random.normal(loc=mask_mean, scale=mask_std), dtype=torch.float32)

    elif mode == 'onehot':
        mask_num = int(node_num * mask_ratio)
        idx_mask = np.random.choice(node_num, mask_num, replace=False)
        x[idx_mask] = torch.tensor(np.eye(feat_dim)[np.random.randint(0, feat_dim, size=(mask_num))], dtype=torch.float32)

    else:
        raise Exception("Masking mode option '{0:s}' is not available!".format(mode))

    return (x, data.edge_index, data.batch)

