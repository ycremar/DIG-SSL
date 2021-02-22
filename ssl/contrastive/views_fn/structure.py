import torch
import numpy as np


def edge_perturbation(data, add=True, drop=False, ratio=0.1):
    '''
    Args:
        data: A graph data object containing:
                batch tensor with shape [num_nodes];
                x tensor with shape [num_nodes, num_node_features];
                y tensor with arbitrary shape;
                edge_attr tensor with shape [num_edges, num_edge_features];
                original edge_index tensor with shape [2, num_edges].
        add (bool): Set True if randomly add edges in a given graph.
        drop (bool): Set True if randomly drop edges in a given graph.
        ratio: Percentage of edges to add or drop.

    Returns:
        x tensor with shape [num_nodes, num_node_features];
        edge_index tensor with shape [2, num_perturb_edges];
        batch tensor with shape [num_nodes].
    '''

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    perturb_num = int(edge_num * ratio)
    edge_index = data.edge_index.transpose(0, 1).numpy()
    idx_remain = edge_index
    idx_add = np.array([]).reshape(-1, 2)

    if drop:
        idx_remain = edge_index[np.random.choice(edge_num, edge_num-perturb_num, replace=False)]

    if add:
        idx_add = np.random.choice(node_num, (perturb_num, 2))
        # idx_add = [idx_add[n] for n in range(perturb_num) if not list(idx_add[n]) in [list(pair) for pair in edge_index]]

    new_edge_index = np.concatenate((idx_remain, idx_add), axis=0)
    new_edge_index = np.unique(new_edge_index, axis=0)

    return (data.x, torch.tensor(new_edge_index).transpose_(0, 1), data.batch)
