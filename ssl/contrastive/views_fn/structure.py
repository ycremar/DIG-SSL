import torch
import numpy as np
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from scipy.linalg import fractional_matrix_power, inv


def edge_perturbation(add=True, drop=False, ratio=0.1, add_self_loop=True):
    '''
    Args:
        add (bool): Set True if randomly add edges in a given graph.
        drop (bool): Set True if randomly drop edges in a given graph.
        ratio: Percentage of edges to add or drop.
        add_self_loop (bool): Set True if add self-loop in edge_index.
    '''
    def views_fn(data):
        '''
        Args:
            data: A graph data object containing:
                    batch tensor with shape [num_nodes];
                    x tensor with shape [num_nodes, num_node_features];
                    y tensor with arbitrary shape;
                    edge_attr tensor with shape [num_edges, num_edge_features];
                    original edge_index tensor with shape [2, num_edges].

        Returns:
            x tensor with shape [num_nodes, num_node_features];
            edge_index tensor with shape [2, num_perturb_edges];
            batch tensor with shape [num_nodes].
        '''
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        perturb_num = int(edge_num * ratio)

        if add_self_loop:
            sl = torch.tensor([[n, n] for n in range(node_num)]).t()
            edge_index = torch.cat((data.edge_index, sl), dim=1)
        else:
            edge_index = data.edge_index.detach().clone()

        edge_index = edge_index.numpy()
        idx_remain = edge_index
        idx_add = np.array([]).reshape(-1, 2)

        if drop:
            idx_remain = edge_index[:, np.random.choice(edge_num, edge_num-perturb_num, replace=False)]

        if add:
            idx_add = np.random.choice(node_num, (2, perturb_num))
            # idx_add = [idx_add[n] for n in range(perturb_num) if not list(idx_add[n]) in [list(pair) for pair in edge_index]]

        new_edge_index = np.concatenate((idx_remain, idx_add), axis=1)
        new_edge_index = np.unique(new_edge_index, axis=1)

        return (data.x, torch.tensor(new_edge_index), data.batch)

    return views_fn


def diffusion(mode='ppr', alpha=0.2, t=5, add_self_loop=True):
    '''
    Args:
        mode: Diffusion instantiation mode with two options:
                'ppr': Personalized PageRank
                'heat': heat kernel
        alpha: Teleport probability in a random walk.
        t: Diffusion time.
        add_self_loop (bool): Set True if add self-loop in edge_index.
    '''
    def views_fn(data):
        '''
        Args:
            data: A graph data object containing:
                    batch tensor with shape [num_nodes];
                    x tensor with shape [num_nodes, num_node_features];
                    y tensor with arbitrary shape;
                    edge_attr tensor with shape [num_edges, num_edge_features];
                    original edge_index tensor with shape [2, num_edges].

        Returns:
            x tensor with shape [num_nodes, num_node_features];
            edge_index tensor with shape [2, num_diff_edges];
            batch tensor with shape [num_nodes].
        '''
        node_num, _ = data.x.size()

        if add_self_loop:
            sl = torch.tensor([[n, n] for n in range(node_num)]).t()
            edge_index = torch.cat((data.edge_index, sl), dim=1)
        else:
            edge_index = data.edge_index.detach().clone()

        adj = to_dense_adj(edge_index)[0].numpy()
        d = np.diag(np.sum(adj, 1))

        if mode == 'ppr':
            dinv = fractional_matrix_power(d, -0.5)
            at = np.matmul(np.matmul(dinv, adj), dinv)
            new_adj = torch.tensor(alpha * inv((np.eye(adj.shape[0]) - (1 - alpha) * at)))

        elif mode == 'heat':
            new_adj = torch.tensor(np.exp(t * (np.matmul(adj, inv(d)) - 1)))

        else:
            raise Exception("Must choose one diffusion instantiation mode from 'ppr' and 'heat'!")

        return (data.x, dense_to_sparse(new_adj)[0], data.batch)

    return views_fn
