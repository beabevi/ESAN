"""
Code taken from ogb examples and adapted
"""

import torch
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import BondEncoder
from torch_geometric.nn import GINConv as PyGINConv
from torch_geometric.nn import GraphConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, in_dim, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                       torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim=in_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class ZINCGINConv(MessagePassing):
    def __init__(self, in_dim, emb_dim):
        super(ZINCGINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = torch.nn.Embedding(4, in_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr.squeeze())
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class OriginalGINConv(torch.nn.Module):
    def __init__(self, in_dim, emb_dim):
        super(OriginalGINConv, self).__init__()
        mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, emb_dim),
            torch.nn.BatchNorm1d(emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim)
        )
        self.layer = PyGINConv(nn=mlp, train_eps=False)

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index)


### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, in_dim, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(in_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm) + \
               F.relu(x + self.root_emb.weight) * 1. / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layer, in_dim, emb_dim, drop_ratio=0.5, JK="last", residual=False, gnn_type='gin',
                 num_random_features=0, feature_encoder=lambda x: x):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual
        self.gnn_type = gnn_type

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = feature_encoder
        self.num_random_features = num_random_features

        if num_random_features > 0:
            assert gnn_type == 'graphconv'

            self.initial_layers = torch.nn.ModuleList(
                [GraphConv(in_dim, emb_dim // 2), GraphConv(emb_dim // 2, emb_dim - num_random_features)]
            )
            # now the next layers will have dimension emb_dim
            in_dim = emb_dim

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim if layer != 0 else in_dim, emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim if layer != 0 else in_dim, emb_dim))
            elif gnn_type == 'originalgin':
                self.convs.append(OriginalGINConv(emb_dim if layer != 0 else in_dim, emb_dim))
            elif gnn_type == 'zincgin':
                self.convs.append(ZINCGINConv(emb_dim if layer != 0 else in_dim, emb_dim))
            elif gnn_type == 'graphconv':
                self.convs.append(GraphConv(emb_dim if layer != 0 else in_dim, emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        if self.num_random_features > 0:
            for layer in self.initial_layers:
                x = F.elu(layer(x, edge_index, edge_attr))

            # Implementation of RNI
            random_dims = torch.empty(x.shape[0], self.num_random_features).to(x.device)
            torch.nn.init.normal_(random_dims)
            x = torch.cat([x, random_dims], dim=1)

        ### computing input node embedding
        h_list = [self.atom_encoder(x)]

        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)

            if self.gnn_type not in ['gin', 'gcn'] or layer != self.num_layer - 1:
                h = F.relu(h)  # remove last relu for ogb

            if self.drop_ratio > 0.:
                h = F.dropout(h, self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]
        elif self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)

        return node_representation
