import torch
import numpy as np
from torch_geometric.utils import to_dense_adj, dense_to_sparse


def uniform_sample(data, ratio=0.1):
    '''
    Args:
        data: A graph data object containing:
                batch tensor with shape [num_nodes];
                x tensor with shape [num_nodes, num_node_features];
                y tensor with arbitrary shape;
                edge_attr tensor with shape [num_edges, num_edge_features];
                edge_index tensor with shape [2, num_edges].
        ratio: Percentage of nodes to drop.

    Returns:
        x tensor with shape [num_nondrop_nodes, num_node_features];
        edge_index tensor with shape [2, num_nondrop_edges];
        batch tensor with shape [num_nondrop_nodes].
    '''

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()

    drop_num = int(node_num * ratio)
    idx_drop = np.random.choice(node_num, drop_num, replace=False)
    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    # edge_index = data.edge_index.numpy()
    # adj = torch.zeros((node_num, node_num))
    # adj[edge_index[0], edge_index[1]] = 1
    adj = to_dense_adj(data.edge_index)[0]
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    # new_edge_index = adj.nonzero().t()

    return (data.x[idx_nondrop], dense_to_sparse(adj), data.batch[idx_nondrop])

