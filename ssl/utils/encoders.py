import torch
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, global_add_pool, global_mean_pool


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=5,
                 pool='sum', gnn='gin', bn=False, act='relu',
                 bias=True, xavier=True, node_level=False, graph_level=True):
        super(Encoder, self).__init__()

        if gnn == 'gin':
            self.encoder = GIN(input_dim, hidden_dim, n_layers, pool, bn, act)
        elif gnn == 'gcn':
            self.encoder = GCN(input_dim, hidden_dim, n_layers, pool, bn, act, bias, xavier)
        self.node_level = node_level
        self.graph_level = graph_level

    def forward(self, x, edge_index, batch):
        z_g, z_n = self.encoder(x, edge_index, batch)
        if self.node_level and self.graph_level:
            return z_g, z_n
        elif self.graph_level:
            return z_g
        else:
            return z_n


class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=5,
                 pool='sum', bn=False, act='relu', bias=True, xavier=True):
        super(GIN, self).__init__()

        if bn:
            self.bns = torch.nn.ModuleList()
        else:
            self.bns = None
        self.convs = torch.nn.ModuleList()
        self.acts = torch.nn.ModuleList()
        self.n_layers = n_layers
        self.pool = pool

        if act == 'prelu':
            a = torch.nn.PReLU()
        else:
            a = torch.nn.ReLU()

        for i in range(n_layers):
            start_dim = hidden_dim if i else input_dim
            nn = Sequential(Linear(start_dim, hidden_dim, bias=bias),
                            a,
                            Linear(hidden_dim, hidden_dim, bias=bias))
            conv = GINConv(nn)
            if xavier:
                self.weights_init(conv)
            self.convs.append(conv)
            self.acts.append(a)
            if bn:
                self.bns.append(BatchNorm1d(hidden_dim))

    def weights_init(self, m):
        if isinstance(m, GINConv):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch):
        xs = []
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            x = self.acts[i](x)
            if self.bn is not None:
                x = self.bns[i](x)
            xs.append(x)

        if self.pool == 'sum':
            xpool = [global_add_pool(x, batch) for x in xs]
        else:
            xpool = [global_mean_pool(x, batch) for x in xs]
        global_rep = torch.cat(xpool, 1)

        return global_rep, x


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=5,
                 pool='sum', bn=False, act='relu', bias=True, xavier=True):
        super(GCN, self).__init__()

        if bn:
            self.bns = torch.nn.ModuleList()
        else:
            self.bns = None
        self.convs = torch.nn.ModuleList()
        self.acts = torch.nn.ModuleList()
        self.n_layers = n_layers
        self.pool = pool

        if act == 'prelu':
            a = torch.nn.PReLU()
        else:
            a = torch.nn.ReLU()

        for i in range(n_layers):
            start_dim = hidden_dim if i else input_dim
            conv = GCNConv(start_dim, hidden_dim, bias=bias)
            if xavier:
                self.weights_init(conv)
            self.convs.append(conv)
            self.acts.append(a)
            if bn:
                self.bns.append(BatchNorm1d(hidden_dim))

    def weights_init(self, m):
        if isinstance(m, GCNConv):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch):
        xs = []
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            x = self.acts[i](x)
            if self.bns is not None:
                x = self.bns[i](x)
            xs.append(x)

        if self.pool == 'sum':
            xpool = [global_add_pool(x, batch) for x in xs]
        else:
            xpool = [global_mean_pool(x, batch) for x in xs]
        global_rep = torch.cat(xpool, 1)

        return global_rep, x

