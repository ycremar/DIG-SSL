import torch
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, global_add_pool, global_mean_pool


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=5,
                 pool='sum', gnn='gin', node_level=False):
        super(Encoder, self).__init__()

        if gnn == 'gin':
            self.encoder = GIN(input_dim, hidden_dim, n_layers, pool)
        elif gnn == 'gcn':
            self.encoder = GCN(input_dim, hidden_dim, n_layers, pool)
        self.node_level = node_level

    def forward(self, x, edge_index, batch):
        z_g, z_n = self.encoder(x, edge_index, batch)
        if self.node_level:
            return z_n
        else:
            return z_g, z_n


class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=5,
                 pool='sum'):
        super(GIN, self).__init__()

        self.bns = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        self.n_layers = n_layers
        self.pool = pool

        for i in range(n_layers):
            start_dim = hidden_dim if i else input_dim
            nn = Sequential(Linear(start_dim, hidden_dim),
                            ReLU(),
                            Linear(hidden_dim, hidden_dim))
            conv = GINConv(nn)
            self.convs.append(conv)
            self.bns.append(BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch):
        xs = []
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            xs.append(x)

        if self.pool == 'sum':
            xpool = [global_add_pool(x, batch) for x in xs]
        else:
            xpool = [global_mean_pool(x, batch) for x in xs]
        global_rep = torch.cat(xpool, 1)

        return global_rep, x


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=5,
                 pool='sum'):
        super(GCN, self).__init__()

        self.bns = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        self.n_layers = n_layers
        self.pool = pool

        for i in range(n_layers):
            start_dim = hidden_dim if i else input_dim
            conv = GCNConv(start_dim, hidden_dim)
            self.convs.append(conv)
            self.bns.append(BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch):
        xs = []
        for i in range(self.n_layers):
            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)

        if self.pool == 'sum':
            xpool = [global_add_pool(x, batch) for x in xs]
        else:
            xpool = [global_mean_pool(x, batch) for x in xs]
        global_rep = torch.cat(xpool, 1)

        return global_rep, x

