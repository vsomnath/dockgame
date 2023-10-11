import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from torch_scatter import scatter, scatter_mean, scatter_add
from torch_cluster import radius, radius_graph

from dockgame.utils.geometry import apply_rigid_transform
from dockgame.common.constants import DEVICE

from dockgame.models.dock_reward import (
    sample_rigid_body_transform, get_activation_layer,
    TensorProductConvLayer, GaussianSmearing
)


class RewardModelHetero(nn.Module):

    def __init__(self,
            node_fdim: int,
            edge_fdim: int,
            sh_lmax: int = 2,
            n_s: int = 16,
            n_v: int = 4,
            n_conv_layers: int = 2,
            max_radius: float = 10.0,
            cross_max_radius: float = 10.0,
            distance_emb_dim: int = 32,
            cross_dist_emb_dim: int = 32,
            dropout_p: float = 0.1,
            activation: str = "relu",  
            enforce_stability=False, n_deviations: int = 0, 
            deviation_eps: float = 0.01,**kwargs
        ):
            
            super().__init__(**kwargs)

            self.node_fdim = node_fdim
            self.edge_fdim = edge_fdim
            self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
            self.n_s, self.n_v = n_s, n_v
            self.n_conv_layers = n_conv_layers

            self.max_radius = max_radius
            self.cross_max_radius = cross_max_radius
            self.enforce_stability = enforce_stability

            self.n_deviations = n_deviations
            self.deviation_eps = deviation_eps
                                                
            irrep_seq = [
                f"{n_s}x0e",
                f"{n_s}x0e + {n_v}x1o",
                f"{n_s}x0e + {n_v}x1o + {n_v}x1e",
                f"{n_s}x0e + {n_v}x1o + {n_v}x1e + {n_s}x0o"
            ]

            act_layer = get_activation_layer(activation)

            self.node_embedding = nn.Sequential(
                nn.Linear(node_fdim, n_s),
                act_layer,
                nn.Dropout(dropout_p) if dropout_p else nn.Identity(),
                nn.Linear(n_s, n_s)
            )

            if cross_dist_emb_dim is None:
                cross_dist_emb_dim = distance_emb_dim

            self.edge_embedding = nn.Sequential(
                nn.Linear(edge_fdim + distance_emb_dim, n_s),
                act_layer,
                nn.Dropout(dropout_p) if dropout_p else nn.Identity(),
                nn.Linear(n_s, n_s)
            )

            self.cross_edge_embedding = nn.Sequential(
                nn.Linear(edge_fdim + cross_dist_emb_dim, n_s),
                act_layer,
                nn.Dropout(dropout_p) if dropout_p else nn.Identity(),
                nn.Linear(n_s, n_s)
            )

            self.dist_expansion = GaussianSmearing(
                start=0.0, stop=max_radius, 
                num_gaussians=distance_emb_dim
            )
            self.cross_dist_expansion = GaussianSmearing(
                start=0.0, stop=cross_max_radius, 
                num_gaussians=cross_dist_emb_dim
            )

            conv_layers, cross_conv_layers = [], []

            for i in range(n_conv_layers):
                in_irreps = irrep_seq[min(i, len(irrep_seq)-1)]
                out_irreps = irrep_seq[min(i+1, len(irrep_seq)-1)]

                parameters = {
                    "in_irreps": in_irreps,
                    "sh_irreps": self.sh_irreps,
                    "out_irreps": out_irreps,
                    "edge_fdim": 3 * n_s,
                    "h_dim": 3 * n_s,
                    "residual": False,
                    "dropout": dropout_p,
                }

                conv_layer = TensorProductConvLayer(**parameters)
                cross_conv_layer = TensorProductConvLayer(**parameters)

                conv_layers.append(conv_layer)
                cross_conv_layers.append(cross_conv_layer)

            self.conv_layers = nn.ModuleList(conv_layers)
            self.cross_conv_layers = nn.ModuleList(cross_conv_layers)

            self.energy_predictor_edges = nn.Sequential(
                #nn.Linear(4* self.n_s + distance_emb_dim if n_conv_layers >= 3 else 2 * self.n_s + distance_emb_dim, self.n_s),
                nn.Linear(5 * self.n_s if n_conv_layers >= 3 else 3 * self.n_s, self.n_s),
                act_layer,
                nn.Dropout(dropout_p),
                nn.Linear(self.n_s, self.n_s),
                act_layer,
                nn.Dropout(dropout_p),
                nn.Linear(self.n_s, 1),
            )

            self.energy_predictor_nodes = nn.Sequential(
                nn.Linear(2 * self.n_s if n_conv_layers >= 3 else self.n_s, 2 * self.n_s),
                act_layer,
                nn.Dropout(dropout_p),
                nn.Linear(2 * self.n_s, self.n_s),
                act_layer,
                nn.Dropout(dropout_p),
                nn.Linear(self.n_s, 1),
            )

    def forward(self, data):
        graph = self.setup_graph(data, pos_to_use="current")
        graph_ref = self.setup_graph(data, pos_to_use="ref")

        cross_graph = self.setup_cross_graph(data, pos_to_use="current")
        cross_graph_ref = self.setup_cross_graph(data, pos_to_use="ref")

        energy = self.compute_energy(graph, cross_graph, data.batch)
        energy_ref = self.compute_energy(graph_ref, cross_graph_ref, data.batch)
        energy_diff = energy - energy_ref

        energy_deviations = None
        energy_bound = None

        return (energy, energy_ref, energy_diff), (energy_deviations, energy_bound)

    def compute_energy(self, graph, cross_graph, batch):
        x, edge_index, edge_attr, edge_sh = graph

        src, dst = edge_index
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)

        cross_edge_index, cross_edge_attr, cross_edge_sh = cross_graph
        cross_src, cross_dst = cross_edge_index
        cross_edge_attr = self.cross_edge_embedding(cross_edge_attr)

        for i in range(self.n_conv_layers):
            edge_attr_ = torch.cat([edge_attr, x[src, :self.n_s], x[dst, :self.n_s]], dim=-1)
            x_intra_update = self.conv_layers[i](x, edge_index, edge_attr_, edge_sh)

            cross_edge_attr_ = torch.cat([cross_edge_attr, x[cross_src, :self.n_s], x[cross_dst, :self.n_s]], dim=-1)
            x_inter_update = self.cross_conv_layers[i](x, cross_edge_index, cross_edge_attr_, cross_edge_sh)

            x = F.pad(x, (0, x_intra_update.shape[-1] - x.shape[-1]))
            x = x + x_intra_update + x_inter_update

        x_src = torch.cat([x[src,:self.n_s], x[src,-self.n_s:]], dim=1) \
            if self.n_conv_layers >= 3 else x[src,:self.n_s] # (n_edges, emb_dim)
        x_dst = torch.cat([x[dst,:self.n_s], x[dst,-self.n_s:]], dim=1) \
            if self.n_conv_layers >= 3 else x[dst,:self.n_s]
        x_feats = torch.cat([x_src, x_dst], dim=-1)

        energy_inputs_edges = torch.cat([edge_attr, x_feats], dim=-1)
        energy_inputs_nodes = torch.cat([x[:, :self.n_s], x[:, -self.n_s:]], dim=1) \
            if self.n_conv_layers >= 3 else x[:, :self.n_s]
        
        energy_nodes = self.energy_predictor_nodes(energy_inputs_nodes)        
        energy_edges = self.energy_predictor_edges(energy_inputs_edges)

        edge_batch = batch[src]

        energy_nodes_agg = scatter_mean(energy_nodes, index=batch, dim=0)
        energy_edges_agg = scatter_mean(energy_edges, index=edge_batch, dim=0)
        
        energy = energy_edges_agg + energy_nodes_agg
        return energy
    
    def predict(self, data):
        graph = self.setup_graph(data=data, pos_to_use="current")
        cross_graph = self.setup_cross_graph(data=data, pos_to_use='current')
        energy = self.compute_energy(graph, cross_graph, data.batch)
        return energy

    def setup_graph(self, data, pos_to_use: str = 'current'):
        if pos_to_use == 'current':
            pos = data.pos
        elif pos_to_use == 'ref':
            pos = data.pos_ref

        edge_index = data.edge_index
        src, dst = edge_index
        edge_vec = pos[src.long()] - pos[dst.long()]

        edge_length_emb = self.dist_expansion(edge_vec.norm(dim=-1))
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalization='component', normalize=True)

        return data.x, edge_index, edge_length_emb, edge_sh
    
    def setup_cross_graph(self, data, pos_to_use: str = 'current'):

        if pos_to_use == "current":
            pos = data.pos
            cross_edge_index = data.cross_edge_index

        elif pos_to_use == "ref":
            pos = data.pos_ref
            cross_edge_index = data.ref_cross_edge_index

        cross_src, cross_dst = cross_edge_index
        edge_vec = pos[cross_src.long()] - pos[cross_dst.long()]

        edge_length_emb = self.cross_dist_expansion(edge_vec.norm(dim=-1))
        cross_edge_sh = o3.spherical_harmonics(
            self.sh_irreps, edge_vec, normalization='component', normalize=True)

        return cross_edge_index, edge_length_emb, cross_edge_sh
