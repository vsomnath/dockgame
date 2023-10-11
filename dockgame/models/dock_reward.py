import torch
import torch.nn as nn
import torch.nn.functional as F

from e3nn import o3
from torch_scatter import scatter, scatter_mean, scatter_add
from torch_cluster import radius, radius_graph
import numpy as np
from scipy.spatial.transform import Rotation

from dockgame.utils.geometry import apply_rigid_transform
from dockgame.common.constants import DEVICE


def sample_rigid_body_transform(max_tr: float = 10.0):

    def sample_translation(max_length):
        tr = np.random.randn(1, 3)
        tr = tr / np.sqrt( np.sum(tr * tr))
        tr_len = np.random.uniform(low=0.0, high=max_length)
        tr = tr * tr_len
        assert np.allclose(np.linalg.norm(tr), tr_len, atol=1e-5)
        tr = tr.squeeze()
        return tr

    def sample_rotation():
        rot_vec = Rotation.random(num=1).as_rotvec()
        return rot_vec

    rot_vec = torch.tensor(sample_rotation()).float()
    tr_vec = torch.tensor(0.5*sample_translation(max_length=max_tr)).float()

    return rot_vec, tr_vec


def get_activation_layer(activation):
    if activation == "relu":
        return nn.ReLU()
    
    elif activation == "silu":
        return nn.SiLU()

    elif activation == "leaky_relu":
        return nn.LeakyReLU()


class GaussianSmearing(nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        mu = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (mu[1] - mu[0]).item() ** 2
        self.register_buffer('mu', mu)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.mu.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class TensorProductConvLayer(nn.Module):

    def __init__(self, in_irreps, sh_irreps, out_irreps, edge_fdim, residual=True, dropout=0.0,
                 h_dim=None):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        if h_dim is None:
            h_dim = edge_fdim
        
        self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc_net = nn.Sequential(
            nn.Linear(edge_fdim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, tp.weight_numel)
        )

    def forward(self, x, edge_index, edge_attr, edge_sh, out_nodes=None, aggr='mean'):
        edge_src, edge_dst = edge_index
        tp_out = self.tp(x[edge_src], edge_sh, self.fc_net(edge_attr))

        out_nodes = out_nodes or x.shape[0]

        out = scatter(src=tp_out, index=edge_dst, dim=0, dim_size=out_nodes, reduce=aggr)

        if self.residual:
            padded = F.pad(x, (0, out.shape[-1] - x.shape[-1]))
            out = out + padded

        return out 


class RewardModel(nn.Module):

    def __init__(self, node_fdim: int, edge_fdim: int, sh_lmax: int = 2,
                 n_s: int = 16, n_v: int = 16, n_conv_layers: int = 2, 
                 max_radius: float = 10.0, max_neighbors: int = 24,
                 distance_emb_dim: int = 32, dropout_p: float = 0.2,
                 activation: str = "relu",  mode: str = "base", 
                 enforce_stability=False, n_deviations: int = 0,
                 deviation_eps: float = 0.01,
                 **kwargs
        ):

            super().__init__(**kwargs)

            self.node_fdim = node_fdim
            self.edge_fdim = edge_fdim
            self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
            self.n_s, self.n_v = n_s, n_v
            self.n_conv_layers = n_conv_layers

            self.max_radius = max_radius
            self.max_neighbors = max_neighbors
            self.mode = mode
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

            self.edge_embedding = nn.Sequential(
                nn.Linear(edge_fdim + distance_emb_dim, n_s),
                act_layer,
                nn.Dropout(dropout_p) if dropout_p else nn.Identity(),
                nn.Linear(n_s, n_s)
            )

            self.dist_expansion = GaussianSmearing(start=0.0, stop=max_radius, num_gaussians=distance_emb_dim)

            conv_layers = []
            for i in range(n_conv_layers):
                in_irreps = irrep_seq[min(i, len(irrep_seq)-1)]
                out_irreps = irrep_seq[min(i+1, len(irrep_seq)-1)]

                parameters = {
                    "in_irreps": in_irreps,
                    "sh_irreps": self.sh_irreps,
                    "out_irreps": out_irreps,
                    "edge_fdim": 2 * n_s + distance_emb_dim,
                    "h_dim": 3 * n_s,
                    "residual": False,
                    "dropout": dropout_p,
                }

                layer = TensorProductConvLayer(**parameters)
                conv_layers.append(layer)
            self.conv_layers = nn.ModuleList(conv_layers)


            if self.mode == "only_pos":
                self.score_predictor_edges = nn.Sequential(
                    nn.Linear(1 + distance_emb_dim, 2 * self.n_s),
                    act_layer,
                    nn.Dropout(dropout_p),
                    nn.Linear(2 * self.n_s, self.n_s),
                    act_layer,
                    nn.Dropout(dropout_p),
                    nn.Linear(self.n_s, 1)
                )

            else:
                self.score_predictor_edges = nn.Sequential(
                    nn.Linear(4* self.n_s + distance_emb_dim if n_conv_layers >= 3 else 2 * self.n_s + distance_emb_dim, self.n_s),
                    act_layer,
                    nn.Dropout(dropout_p),
                    nn.Linear(self.n_s, self.n_s),
                    act_layer,
                    nn.Dropout(dropout_p),
                    nn.Linear(self.n_s, 1),
                )

            self.score_predictor_nodes = nn.Sequential(
                nn.Linear(2 * self.n_s if n_conv_layers >= 3 else self.n_s, 2 * self.n_s),
                act_layer,
                nn.Dropout(dropout_p),
                nn.Linear(2 * self.n_s, self.n_s),
                act_layer,
                nn.Dropout(dropout_p),
                nn.Linear(self.n_s, 1),
            )

    def compute_energy(self, graph, batch):
        x, edge_index, edge_attr, edge_sh = graph
        src, dst = edge_index
        x = self.node_embedding(x)

        for i in range(self.n_conv_layers):
            edge_attr_ = torch.cat([edge_attr, x[dst, :self.n_s], x[src, :self.n_s]], dim=-1)
            x_update = self.conv_layers[i](x, edge_index, edge_attr_, edge_sh)

            x = F.pad(x, (0, x_update.shape[-1] - x.shape[-1]))
            x = x + x_update
        
        x_src = torch.cat([x[src,:self.n_s], x[src,-self.n_s:]], dim=1) if self.n_conv_layers >= 3 else x[src,:self.n_s] # (n_edges, emb_dim)
        x_dst = torch.cat([x[dst,:self.n_s], x[dst,-self.n_s:]], dim=1) if self.n_conv_layers >= 3 else x[dst,:self.n_s]
        x_feats = torch.cat([x_src, x_dst], dim=-1)

        score_inputs_edges = torch.cat([edge_attr, x_feats], dim=-1)
        score_inputs_nodes = torch.cat([x[:, :self.n_s], x[:, -self.n_s:]], dim=1) if self.n_conv_layers >= 3 else x[:, :self.n_s]
        
        scores_nodes = self.score_predictor_nodes(score_inputs_nodes)        
        scores_edges = self.score_predictor_edges(score_inputs_edges)

        edge_batch = batch[src]
        score = scatter_mean(scores_edges, index=edge_batch, dim=0) + scatter_mean(scores_nodes, index=batch, dim=0)

        return score

    def forward(self, data):
        graph = self.build_graph(data, pos_to_use="current")
        energy = self.compute_energy(graph, data.batch)
        graph_ref = self.build_graph(data, pos_to_use="ref")
        energy_ref = self.compute_energy(graph_ref, data.batch)
        energy_diff = energy - energy_ref
        
        energy_bound = None
        #graph = self.build_graph(data, pos_to_use="bound")
        #energy_bound = self.compute_energy(graph, data.batch)
        energy_deviations = None
        if self.n_deviations > 0:
            energy_deviations = self._compute_score_deviations(data=data)

        return (energy, energy_ref, energy_diff), (energy_deviations, energy_bound)

    def forward_no_embeddings(self, data, pos_to_use: str = "current"):
        assert pos_to_use in ["current", "ref", "bound"], f"{pos_to_use} not supported."

        if pos_to_use == "current":
            pos = data.pos
        elif pos_to_use == "ref":
            pos = data.pos_ref
        elif pos_to_use == "bound":
            pos = data.pos_bound

        radius_edges = radius_graph(pos, self.max_radius, data.batch)

        src, dst = radius_edges
        edge_vec = pos[dst.long()] - pos[src.long()]
        edge_length = edge_vec.norm(dim=-1, keepdim=True)
        edge_length_emb = self.dist_expansion(edge_vec.norm(dim=-1))

        score_inputs = torch.cat([edge_length, edge_length_emb], dim=-1)
        scores = self.score_predictor_edges(score_inputs)
        score = scatter_add(scores, index=data.batch[src], dim=0)
        return score

    def _compute_score_deviations(self, data):
        # Perturb the data.pos and compute energy for each perturbation according to the model
        energy_deviations = []
        for _ in range(self.n_deviations):
            data_dev_i = data.clone()
            tr_vec, rot_vec = sample_rigid_body_transform(max_tr=self.deviation_eps)
            rot_vec = rot_vec * self.deviation_eps
            
            data_dev_i.pos_bound = apply_rigid_transform(
                        pos=data_dev_i.pos_bound,
                        rot_vec=rot_vec.to(DEVICE),
                        tr_vec=tr_vec.to(DEVICE)
                    )

            graph = self.build_graph(data_dev_i, pos_to_use="bound")
            energy = self.compute_energy(graph, data_dev_i.batch)
            energy_deviations.append(energy)

        return energy_deviations

    def predict(self, data):
        graph = self.build_graph(data=data, pos_to_use="current")
        energy = self.compute_energy(graph, data.batch)
        return energy
    
    def build_graph(self, data, pos_to_use: str = "current"):
        assert pos_to_use in ["current", "ref", "bound"], f"{pos_to_use} not supported."

        if pos_to_use == "current":
            pos = data.pos
        elif pos_to_use == "ref":
            pos = data.pos_ref
        elif pos_to_use == "bound":
            pos = data.pos_bound
        
        radius_edges = radius_graph(pos, self.max_radius, data.batch)

        src, dst = radius_edges
        edge_vec = pos[dst.long()] - pos[src.long()] # (n_edges, 3)
        edge_length_emb = self.dist_expansion(edge_vec.norm(dim=-1)) # (n_edges, distance_dim)

        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return data.x, radius_edges, edge_length_emb, edge_sh
