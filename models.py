import torch
import torch.nn.functional as F
import torch_scatter
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention
import numpy as np
from conv import GNN_node

def subgraph_pool(h_node, batched_data, pool):
    # Represent each subgraph as the pool of its node representations
    num_subgraphs = batched_data.num_subgraphs
    tmp = torch.cat([torch.zeros(1, device=num_subgraphs.device, dtype=num_subgraphs.dtype),
                     torch.cumsum(num_subgraphs, dim=0)])
    graph_offset = tmp[batched_data.batch]

    subgraph_idx = graph_offset + batched_data.subgraph_batch

    return pool(h_node, subgraph_idx)


class GNN(torch.nn.Module):

    def __init__(self, num_tasks, num_layer=5, in_dim=300, emb_dim=300,
                 gnn_type='gin', num_random_features=0, residual=False, drop_ratio=0.5, JK="last", graph_pooling="mean",
                 feature_encoder=lambda x: x):

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.out_dim = self.emb_dim if self.JK == 'last' else self.emb_dim * self.num_layer + in_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        self.gnn_node = GNN_node(num_layer, in_dim, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual,
                                 gnn_type=gnn_type, num_random_features=num_random_features,
                                 feature_encoder=feature_encoder)

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                            torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, 1)))
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        return subgraph_pool(h_node, batched_data, self.pool)


class GNNComplete(GNN):
    def __init__(self, num_tasks, num_layer=5, in_dim=300, emb_dim=300,
                 gnn_type='gin', num_random_features=0, residual=False, drop_ratio=0.5, JK="last", graph_pooling="mean",
                 feature_encoder=lambda x: x):

        super(GNNComplete, self).__init__(num_tasks, num_layer, in_dim, emb_dim, gnn_type, num_random_features,
                                          residual, drop_ratio, JK, graph_pooling, feature_encoder)

        if gnn_type == 'graphconv':
            self.final_layers = torch.nn.Sequential(
                torch.nn.Linear(in_features=self.out_dim, out_features=self.out_dim),
                torch.nn.ELU(),
                torch.nn.Linear(in_features=self.out_dim, out_features=self.out_dim // 2),
                torch.nn.ELU(),
                torch.nn.Linear(in_features=self.out_dim // 2, out_features=num_tasks)
            )
        else:
            self.final_layers = torch.nn.Sequential(
                torch.nn.Linear(in_features=self.out_dim, out_features=num_tasks),
            )

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)

        return self.final_layers(h_graph)


class DSnetwork(torch.nn.Module):
    def __init__(self, subgraph_gnn, channels, num_tasks, invariant):
        super(DSnetwork, self).__init__()
        self.subgraph_gnn = subgraph_gnn
        self.invariant = invariant

        fc_list = []
        fc_sum_list = []
        for i in range(len(channels)):
            fc_list.append(torch.nn.Linear(in_features=channels[i - 1] if i > 0 else subgraph_gnn.out_dim,
                                           out_features=channels[i]))
            if self.invariant:
                fc_sum_list.append(torch.nn.Linear(in_features=channels[i],
                                                   out_features=channels[i]))
            else:
                fc_sum_list.append(torch.nn.Linear(in_features=channels[i - 1] if i > 0 else subgraph_gnn.out_dim,
                                                   out_features=channels[i]))

        self.fc_list = torch.nn.ModuleList(fc_list)
        self.fc_sum_list = torch.nn.ModuleList(fc_sum_list)

        self.final_layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=channels[-1], out_features=2 * channels[-1]),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2 * channels[-1], out_features=num_tasks)
        )

    def forward(self, batched_data):
        h_subgraph = self.subgraph_gnn(batched_data)

        if self.invariant:
            for layer_idx, (fc, fc_sum) in enumerate(zip(self.fc_list, self.fc_sum_list)):
                x1 = fc(h_subgraph)

                h_subgraph = F.elu(x1)

            # aggregate to obtain a representation of the graph given the representations of the subgraphs
            h_graph = torch_scatter.scatter(src=h_subgraph, index=batched_data.subgraph_idx_batch, dim=0, reduce="mean")
            for layer_idx, fc_sum in enumerate(self.fc_sum_list):
                h_graph = F.elu(fc_sum(h_graph))
        else:
            for layer_idx, (fc, fc_sum) in enumerate(zip(self.fc_list, self.fc_sum_list)):
                x1 = fc(h_subgraph)
                x2 = fc_sum(
                    torch_scatter.scatter(src=h_subgraph, index=batched_data.subgraph_idx_batch, dim=0, reduce="mean")
                )

                h_subgraph = F.elu(x1 + x2[batched_data.subgraph_idx_batch])

            # aggregate to obtain a representation of the graph given the representations of the subgraphs
            h_graph = torch_scatter.scatter(src=h_subgraph, index=batched_data.subgraph_idx_batch, dim=0, reduce="mean")

        return self.final_layers(h_graph)


class DSSnetwork(torch.nn.Module):
    def __init__(self, num_layers, in_dim, emb_dim, num_tasks, feature_encoder, GNNConv):
        super(DSSnetwork, self).__init__()

        self.emb_dim = emb_dim

        self.feature_encoder = feature_encoder

        gnn_list = []
        gnn_sum_list = []
        bn_list = []
        bn_sum_list = []
        for i in range(num_layers):
            gnn_list.append(GNNConv(emb_dim if i != 0 else in_dim, emb_dim))
            bn_list.append(torch.nn.BatchNorm1d(emb_dim))

            gnn_sum_list.append(GNNConv(emb_dim if i != 0 else in_dim, emb_dim))
            bn_sum_list.append(torch.nn.BatchNorm1d(emb_dim))

        self.gnn_list = torch.nn.ModuleList(gnn_list)
        self.gnn_sum_list = torch.nn.ModuleList(gnn_sum_list)

        self.bn_list = torch.nn.ModuleList(bn_list)
        self.bn_sum_list = torch.nn.ModuleList(bn_sum_list)

        self.final_layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=emb_dim, out_features=2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2 * emb_dim, out_features=num_tasks)
        )

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        x = self.feature_encoder(x)
        for i in range(len(self.gnn_list)):
            gnn, bn, gnn_sum, bn_sum = self.gnn_list[i], self.bn_list[i], self.gnn_sum_list[i], self.bn_sum_list[i]

            h1 = bn(gnn(x, edge_index, edge_attr))

            num_nodes_per_subgraph = batched_data.num_nodes_per_subgraph
            tmp = torch.cat([torch.zeros(1, device=num_nodes_per_subgraph.device, dtype=num_nodes_per_subgraph.dtype),
                             torch.cumsum(num_nodes_per_subgraph, dim=0)])
            graph_offset = tmp[batch]

            # Same idx for a node appearing in different subgraphs of the same graph
            node_idx = graph_offset + batched_data.subgraph_node_idx
            x_sum = torch_scatter.scatter(src=x, index=node_idx, dim=0, reduce="mean")

            h2 = bn_sum(gnn_sum(x_sum, batched_data.original_edge_index,
                                batched_data.original_edge_attr if edge_attr is not None else edge_attr))

            x = F.relu(h1 + h2[node_idx])

        h_subgraph = subgraph_pool(x, batched_data, global_mean_pool)
        # aggregate to obtain a representation of the graph given the representations of the subgraphs
        h_graph = torch_scatter.scatter(src=h_subgraph, index=batched_data.subgraph_idx_batch, dim=0, reduce="mean")

        return self.final_layers(h_graph)




# ==================== DSS with Attention ==================== #
class SelfAttentionLayer(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, batch_first=True):
        """
        Initializes a self-attention layer.

        Args:
            embed_dim (int): The input embedding dimension.
            num_heads (int): The number of attention heads.
            batch_first (bool, optional): If True, input and output tensors are in batch-first format.
                Defaults to True.
        """
        super(SelfAttentionLayer, self).__init__()
        self.embed_dim = embed_dim # d
        self.num_heads = num_heads
        self.multihead_attention = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=batch_first)

    def forward(self, key_features, query_features, attn_mask=None):
        """
        Forward pass of the self-attention layer.

        Args:
            key_features (torch.Tensor): The key features tensor with shape (batch_size, sequence_length, embed_dim).
            query_features (torch.Tensor): The query features tensor with shape (batch_size, sequence_length, embed_dim).

        Returns:
            torch.Tensor: The output tensor of the self-attention layer with shape (batch_size, sequence_length, embed_dim).
            torch.Tensor: The attention weights tensor with shape (batch_size, num_heads, sequence_length, sequence_length).

        Example:
            batch_size = 1
            embed_dim = 128
            num_subgraphs = 8
            num_heads = 8

            key_features = torch.randn(batch_size, num_subgraphs, embed_dim)
            query_features = torch.randn(batch_size, num_subgraphs, embed_dim)

            attention_layer = SelfAttentionLayer(embed_dim, num_heads, batch_first=True)
            attention_output, attention_weights = attention_layer(key_features, query_features)
        """

        if attn_mask == None:
            attention_output, attention_weights = self.multihead_attention(
                query_features, key_features, key_features, need_weights=True
            )
        else:
            attention_output, attention_weights = self.multihead_attention(
                query_features, key_features, key_features, need_weights=True, attn_mask=attn_mask
            )

        return attention_output, attention_weights

class DSSnetwork_Atten(torch.nn.Module):
    def __init__(self, num_layers, in_dim, emb_dim, num_tasks, feature_encoder, GNNConv, num_heads_attn=1):
        super(DSSnetwork_Atten, self).__init__()

        self.emb_dim = emb_dim
        self.feature_encoder = feature_encoder
        self.num_heads_attn = num_heads_attn

        gnn_list = []
        bn_list = []

        attn_list = []
        gnn_sum_list = []

        bn_sum_list = []
        for i in range(num_layers):
            gnn_list.append(GNNConv(emb_dim if i != 0 else in_dim, emb_dim))
            bn_list.append(torch.nn.BatchNorm1d(emb_dim))

            gnn_sum_list.append(GNNConv(emb_dim if i != 0 else in_dim, emb_dim))
            # Adding the attention layer
            attn_list.append(SelfAttentionLayer(emb_dim if i != 0 else in_dim, num_heads=self.num_heads_attn), )
            bn_sum_list.append(torch.nn.BatchNorm1d(emb_dim))

        self.gnn_list = torch.nn.ModuleList(gnn_list)
        self.gnn_sum_list = torch.nn.ModuleList(gnn_sum_list)
        # Adding the attention layers
        self.attn_list = torch.nn.ModuleList(attn_list)
        self.bn_list = torch.nn.ModuleList(bn_list)
        self.bn_sum_list = torch.nn.ModuleList(bn_sum_list)

        self.final_layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=emb_dim, out_features=2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2 * emb_dim, out_features=num_tasks)
        )

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        x = self.feature_encoder(x)
        for i in range(len(self.gnn_list)):
            gnn, bn, attn, gnn_sum, bn_sum = self.gnn_list[i], self.bn_list[i], self.attn_list[i], self.gnn_sum_list[i], \
                                             self.bn_sum_list[i]

            h1 = bn(gnn(x, edge_index, edge_attr))
            # ===== beginning of my code ===== #

            # Getting global subgraph features
            index_for_subgraph_pool = self.get_subgraphs_by_indices(num_subgraphs=batched_data.num_subgraphs,
                                                                    num_nodes_per_subgraph=batched_data.num_nodes_per_subgraph)
            # Extracts global subgraph features (total_subgraphs_across_all_graphs, embs)
            global_subgraph_features = torch_scatter.scatter(src=x, index=index_for_subgraph_pool, dim=0, reduce="mean")
            # Creates a mask (total_subgraphs_across_all_graphs x total_subgraphs_across_all_graphs) to avoid attention across batches
            # i.e., stacks square matrices on the diagonal of size (num_subgraphs, num_subgraphs) for each graph
            mask = torch.block_diag(*[torch.ones(n, n) for n in batched_data.num_subgraphs])
            mask = torch.where(mask == 1, torch.tensor(0.), torch.tensor(float('-inf')))
            mask = mask.unsqueeze(0).repeat(self.num_heads_attn, 1, 1).to(global_subgraph_features.device)
            # mask_old = torch.block_diag(*[torch.ones(n, n) for n in batched_data.num_subgraphs]).unsqueeze(0).repeat(self.num_heads_attn, 1, 1).to(global_subgraph_features.device)
            # Computes the attention (total_subgraphs_across_all_graphs, total_subgraphs_across_all_graphs)
            _, attention_coefficients = attn(global_subgraph_features.unsqueeze(0), global_subgraph_features.unsqueeze(0), attn_mask=mask)
            attention_coefficients = attention_coefficients[0]
            attn_index = 0
            attn_previous_index = 0

            x_index = 0
            x_previous_index = 0
            attentions = []
            # Iterates over graphs
            for num_subgraph_per_graph, num_nodes_in_subgraphs in zip(batched_data.num_subgraphs, batched_data.num_nodes_per_subgraph):
                x_index += num_subgraph_per_graph * num_nodes_in_subgraphs
                attn_index += num_subgraph_per_graph
                # Slices the relevant attention weights for the subgraphs of this graph (num_subgraphs, num_subgraphs)
                relevant_attn_block = attention_coefficients[attn_previous_index:attn_index, attn_previous_index:attn_index]
                # Save to show as heat map in wandb
                attention_as_heatmap = relevant_attn_block.cpu().detach().numpy()
                # Reshapes to (num_subgraphs, num_subgraphs, 1)
                relevant_attn_block = relevant_attn_block.unsqueeze(-1)
                # Slices the relevant x features for the subgraphs of this graph (num_subgraphs^2 x embs)
                relevant_x = x[x_previous_index:x_index]
                # Reshapes to (num_subgraphs, num_subgraphs, embs)
                relevant_x = relevant_x.reshape(relevant_attn_block.shape[0], relevant_attn_block.shape[1], relevant_x.shape[-1])
                # Multiplies (automatic broadcast)
                curr_graph_weighted_sum = relevant_x * relevant_attn_block
                # Sums across the rows (the subgraphs)
                curr_graph_weighted_sum = curr_graph_weighted_sum.sum(dim=0)
                # Appends the relevant weighted sum for this batch
                attentions.append(curr_graph_weighted_sum)
                attn_previous_index = attn_index.clone()
                x_previous_index = x_index.clone()


            x_atten = torch.cat(attentions, dim=0) # shape is (sum of nodes across all graphs, embs)

            # x_attn = self.compute_sum(x, attention_coefficients[0])

            # batch_sizes = batched_data.num_subgraphs
            # Calculating the attention for each batch manually
            # start_idx = 0
            # x_atten = []
            # attention_coefficients = []
            # for batch_idx, batch_size in enumerate(batch_sizes):
            #     end_idx = start_idx + batch_size
            #
            #     # Get the current batch
            #     current_batch_features = global_subgraph_features[start_idx:end_idx]
            #     current_batch_features = current_batch_features.unsqueeze(0)
            #
            #     # Compute attention
            #     _, attention_coefficients_for_current_batch = attn(current_batch_features, current_batch_features)
            #     attention_coefficients_for_current_batch = attention_coefficients_for_current_batch.squeeze(0)
            #     attention_coefficients.append(attention_coefficients_for_current_batch)
            #
            #
            #     x_for_batch = x[:(batched_data.num_subgraphs * batched_data.num_nodes_per_subgraph)[batch_idx]]
            #     batch_result = self.compute_sum(x_for_batch, attention_coefficients_for_current_batch)
            #     x_atten.append(batch_result)
            # attention_as_heatmap = attention_coefficients[0].cpu().detach().numpy()
            #
            # x_atten = torch.cat(x_atten, dim=0) # num_subgraphs_across_all_graphs x emb_dim
            h2 = bn_sum(gnn_sum(x_atten, batched_data.original_edge_index,
                                batched_data.original_edge_attr if edge_attr is not None else edge_attr))

            # ===== end of my code ===== #
            num_nodes_per_subgraph = batched_data.num_nodes_per_subgraph
            tmp = torch.cat([torch.zeros(1, device=num_nodes_per_subgraph.device, dtype=num_nodes_per_subgraph.dtype),
                             torch.cumsum(num_nodes_per_subgraph, dim=0)])
            graph_offset = tmp[batch]

            # Same idx for a node appearing in different subgraphs of the same graph
            node_idx = graph_offset + batched_data.subgraph_node_idx
            # ==== beginning of old code ===== #

            # x_mean = torch_scatter.scatter(src=x, index=node_idx, dim=0, reduce="mean")
            #
            # h2 = bn_sum(gnn_sum(x_mean, batched_data.original_edge_index,
            #                     batched_data.original_edge_attr if edge_attr is not None else edge_attr))

            # ==== end of old code ===== #
            x = F.relu(h1 + h2[node_idx])

        h_subgraph = subgraph_pool(x, batched_data, global_mean_pool)
        # aggregate to obtain a representation of the graph given the representations of the subgraphs
        h_graph = torch_scatter.scatter(src=h_subgraph, index=batched_data.subgraph_idx_batch, dim=0, reduce="mean")

        return self.final_layers(h_graph), attention_as_heatmap

    def compute_sum(self, X, alpha):
        a, b = alpha.size()

        # Reshape X to (a, b, 300)
        X = X.view(a, b, X.shape[-1])

        # Reshape alpha to (a, b, 1)
        alpha = alpha.unsqueeze(2)

        # Element-wise multiplication between alpha and X (broadcasts automatically)
        alpha_X = alpha * X

        # Sum along the first dimension - which is the subgraphs dimension
        result = alpha_X.sum(dim=0)

        return result

    def get_subgraphs_by_indices(self, num_subgraphs, num_nodes_per_subgraph):
        """
        Generates subgraph indices based on the given number of subgraphs and nodes per subgraph.

        Args:
            num_subgraphs (list): A list of integers representing the number of subgraphs in each batch.
            num_nodes_per_subgraph (list): A list of integers representing the number of nodes per subgraph in each batch.

        Returns:
            list: A list of subgraph indices.

        Example:
            num_subgraphs = [2, 3]
            num_nodes_per_subgraph = [4, 5]
            get_subgraphs_by_indices(num_subgraphs, num_nodes_per_subgraph)
            Output: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]

        """
        subgraph_indices = []
        value = -1

        for num_subgraphs_current_batch, num_nodes_per_subgraph_current_batch in zip(num_subgraphs,
                                                                                     num_nodes_per_subgraph):
            for _ in range(num_subgraphs_current_batch.item()):
                value += 1
                subgraph_indices.extend([value] * num_nodes_per_subgraph_current_batch)

        subgraph_indices = torch.tensor(subgraph_indices, device=num_nodes_per_subgraph.device,
                                        dtype=num_nodes_per_subgraph.dtype)
        return subgraph_indices

# ============================================================ #
class EgoEncoder(torch.nn.Module):
    def __init__(self, encoder):
        super(EgoEncoder, self).__init__()
        self.num_added = 2
        self.enc = encoder

    def forward(self, x):
        return torch.hstack((x[:, :self.num_added], self.enc(x[:, self.num_added:])))


class ZincAtomEncoder(torch.nn.Module):
    def __init__(self, policy, emb_dim):
        super(ZincAtomEncoder, self).__init__()
        self.policy = policy
        self.num_added = 2
        self.enc = torch.nn.Embedding(21, emb_dim)

    def forward(self, x):
        if self.policy == 'ego_nets_plus':
            return torch.hstack((x[:, :self.num_added], self.enc(x[:, self.num_added:].squeeze())))
        else:
            return self.enc(x.squeeze())
