import torch
import numpy as np
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from scipy.linalg import fractional_matrix_power, inv
from torch_geometric.data import Batch, Data


def edge_perturbation(add=True, drop=False, ratio=0.1):
    '''
    Args:
        add (bool): Set True if randomly add edges in a given graph.
        drop (bool): Set True if randomly drop edges in a given graph.
        ratio: Percentage of edges to add or drop.
    '''
    def do_trans(data):
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        perturb_num = int(edge_num * ratio)

        edge_index = data.edge_index.detach().clone()
        idx_remain = edge_index
        idx_add = torch.tensor([]).reshape(-1, 2)

        if drop:
            idx_remain = edge_index[:, np.random.choice(edge_num, edge_num-perturb_num, replace=False)]

        if add:
            idx_add = torch.randint(node_num, (2, perturb_num))

        new_edge_index = torch.cat((idx_remain, idx_add), dim=1)
        new_edge_index = torch.unique(new_edge_index, dim=1)

        return Data(x=data.x, edge_index=new_edge_index)

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
        if isinstance(data, Batch):
            dlist = [do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        elif isinstance(data, Data):
            return do_trans(data)

    return views_fn


def diffusion(mode='ppr', alpha=0.2, t=5):
    '''
    Args:
        mode: Diffusion instantiation mode with two options:
                'ppr': Personalized PageRank
                'heat': heat kernel
        alpha: Teleport probability in a random walk.
        t: Diffusion time.
    '''
    def do_trans(data):
        node_num, _ = data.x.size()
        adj = to_dense_adj(data.edge_index)[0]
        d = torch.diag(torch.sum(adj, 1))

        if mode == 'ppr':
            dinv = torch.inverse(torch.sqrt(d))
            at = torch.matmul(torch.matmul(dinv, adj), dinv)
            new_adj = alpha * torch.inverse((torch.eye(adj.shape[0]) - (1 - alpha) * at))

        elif mode == 'heat':
            new_adj = torch.exp(t * (torch.matmul(adj, torch.inverse(d)) - 1))

        else:
            raise Exception("Must choose one diffusion instantiation mode from 'ppr' and 'heat'!")

        return Data(x=data.x, edge_index=dense_to_sparse(new_adj)[0])

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
        if isinstance(data, Batch):
            dlist = [do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        elif isinstance(data, Data):
            return do_trans(data)

    return views_fn


def diffusion_with_sample(sample_size=2000, batch_size=4, mode='ppr', alpha=0.2, t=5):
    '''
    Args:
        sample_size: Number of nodes in the sampled subgraoh from a large graph.
        batch_size: Number of subgraphs to sample.
        mode: Diffusion instantiation mode with two options:
                'ppr': Personalized PageRank
                'heat': heat kernel
        alpha: Teleport probability in a random walk.
        t: Diffusion time.
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
        orig_adj = to_dense_adj(data.edge_index)[0]
        d = torch.diag(torch.sum(orig_adj, 1))

        if mode == 'ppr':
            dinv = torch.inverse(torch.sqrt(d))
            at = torch.matmul(torch.matmul(dinv, orig_adj), dinv)
            diff_adj = alpha * torch.inverse((torch.eye(orig_adj.shape[0]) - (1 - alpha) * at))

        elif mode == 'heat':
            diff_adj = torch.exp(t * (torch.matmul(orig_adj, torch.inverse(d)) - 1))

        else:
            raise Exception("Must choose one diffusion instantiation mode from 'ppr' and 'heat'!")

        dlist_orig_x = []
        dlist_diff_x = []
        # dlist_orig_shuf = []
        # dlist_diff_shuf = []
        drop_num = node_num - sample_size
        # idx_shuffle = np.random.permutation(sample_size)
        for b in range(batch_size):
            idx_drop = np.random.choice(node_num, drop_num, replace=False)
            idx_nondrop = [n for n in range(node_num) if not n in idx_drop]

            sample_orig_adj = orig_adj.copy()
            sample_orig_adj = sample_orig_adj[idx_nondrop, :][:, idx_nondrop]

            sample_diff_adj = diff_adj.copy()
            sample_diff_adj = sample_diff_adj[idx_nondrop, :][:, idx_nondrop]

            sample_orig_x = data.x[idx_nondrop]
            # sample_shuffle_x = sample_orig_x[idx_shuffle, :]

            dlist_orig_x.append(Data(x=sample_orig_x, edge_index=dense_to_sparse(sample_orig_adj)[0]))
            dlist_diff_x.append(Data(x=sample_orig_x, edge_index=dense_to_sparse(sample_diff_adj)[0]))
            # dlist_orig_shuf.append(Data(x=sample_shuffle_x, edge_index=dense_to_sparse(sample_orig_adj)[0]))
            # dlist_diff_shuf.append(Data(x=sample_shuffle_x, edge_index=dense_to_sparse(sample_diff_adj)[0]))

        return (Batch.from_data_list(dlist_orig_x), Batch.from_data_list(dlist_diff_x))

    return views_fn
