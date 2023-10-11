from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter, scatter_add
from e3nn import o3

from dockgame.common.constants import amino_acid_types
import dockgame.utils.so3 as so3



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
    

class NodeEmbedding(nn.Module):

    def __init__(self, n_residues: int, residue_feats: int, sigma_embed_dim, n_s: int):
        super().__init__()
        self.residue_type_embed = nn.Embedding(n_residues, n_s)
        self.residue_feats_embed = nn.Linear(residue_feats, n_s)
        self.sigma_embed = nn.Linear(sigma_embed_dim, n_s)

        self.n_residues = n_residues
        self.residue_feats = residue_feats
        self.sigma_embed_dim = sigma_embed_dim

    def forward(self, x):
        x_restype = torch.argmax(x[:, :self.n_residues], dim=1)
        res_type_emb = self.residue_type_embed(x_restype)

        x_resfeats = x[:, self.n_residues: self.n_residues + self.residue_feats]
        res_feats_emb = self.residue_feats_embed(x_resfeats)

        x_sigma = x[:, self.n_residues + self.residue_feats:]
        sigma_emb = self.sigma_embed(x_sigma)

        node_emb = res_feats_emb + res_type_emb + sigma_emb
        return node_emb
    

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


class ScoreModel(nn.Module):

    def __init__(self, 
            node_fdim: int, 
            edge_fdim: int, 
            sh_lmax: int = 2,
            n_s: int = 16, 
            n_v: int = 16, 
            n_conv_layers: int = 2, 
            max_radius: float = 10.0, 
            cross_max_radius: float = 10.0,
            center_max_radius: float = 10.0,
            distance_emb_dim: int = 32,
            cross_dist_emb_dim: int = 32,
            center_dist_emb_dim: int = 32,
            timestep_emb_fn=None,
            sigma_emb_dim: int = 32,
            dropout_p: float = 0.2,
            activation: str = "relu",
            t_to_sigma_fn: Callable = None,
            scale_by_sigma: bool = False,
            node_encoder_type: str = 'base',
            **kwargs
        ):

        super().__init__(**kwargs)

        self.node_fdim = node_fdim
        self.edge_fdim = edge_fdim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.n_s, self.n_v = n_s, n_v
        self.n_conv_layers = n_conv_layers

        self.max_radius = max_radius
        self.cross_max_radius = cross_max_radius
        self.timestep_emb_fn = timestep_emb_fn

        self.scale_by_sigma = scale_by_sigma
        self.t_to_sigma_fn = t_to_sigma_fn
                                            
        irrep_seq = [
            f"{n_s}x0e",
            f"{n_s}x0e + {n_v}x1o",
            f"{n_s}x0e + {n_v}x1o + {n_v}x1e",
            f"{n_s}x0e + {n_v}x1o + {n_v}x1e + {n_s}x0o"
        ]

        act_layer = get_activation_layer(activation)

        if node_encoder_type == 'base':
            self.node_embedding = nn.Sequential(
                nn.Linear(node_fdim + sigma_emb_dim, n_s),
                act_layer,
                nn.Dropout(dropout_p) if dropout_p else nn.Identity(),
                nn.Linear(n_s, n_s)
            )
        else:
            self.node_embedding = NodeEmbedding(
                n_residues=len(amino_acid_types),
                residue_feats=node_fdim - len(amino_acid_types),
                sigma_embed_dim=sigma_emb_dim,
                n_s=n_s
            )

        if cross_dist_emb_dim is None:
            cross_dist_emb_dim = distance_emb_dim

        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_fdim + distance_emb_dim + sigma_emb_dim, n_s),
            act_layer,
            nn.Dropout(dropout_p) if dropout_p else nn.Identity(),
            nn.Linear(n_s, n_s)
        )

        self.cross_edge_embedding = nn.Sequential(
            nn.Linear(cross_dist_emb_dim + sigma_emb_dim, n_s),
            act_layer,
            nn.Dropout(dropout_p) if dropout_p else nn.Identity(),
            nn.Linear(n_s, n_s)
        )

        self.dist_expansion = GaussianSmearing(start=0.0, stop=max_radius, num_gaussians=distance_emb_dim)
        self.cross_dist_expansion = GaussianSmearing(start=0.0, stop=cross_max_radius, 
                                                        num_gaussians=cross_dist_emb_dim)
        self.center_dist_expansion = GaussianSmearing(start=0.0, stop=center_max_radius,
                                                      num_gaussians=center_dist_emb_dim)

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

        self.center_edge_embedding = nn.Sequential(
                nn.Linear(center_dist_emb_dim + sigma_emb_dim, n_s),
                act_layer,
                nn.Dropout(dropout_p),
                nn.Linear(n_s, n_s),
            )
        
        self.final_conv = TensorProductConvLayer(
            in_irreps=self.conv_layers[-1].out_irreps,
            sh_irreps=self.sh_irreps,
            out_irreps=f'2x1o + 2x1e',
            edge_fdim=2 * n_s,
            h_dim=2 * n_s,
            residual=False,
            dropout=dropout_p,
        )

        self.tr_final_layer = nn.Sequential(
            nn.Linear(1 + sigma_emb_dim, n_s),
            nn.Dropout(dropout_p),
            act_layer,
            nn.Linear(n_s, 1),
        )

        self.rot_final_layer = nn.Sequential(
            nn.Linear(1 + sigma_emb_dim, n_s),
            nn.Dropout(dropout_p),
            act_layer,
            nn.Linear(n_s, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, edge_sh = self.setup_graph(data)
        src, dst = edge_index
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)

        cross_edge_index, cross_edge_attr, cross_edge_sh = self.setup_cross_graph(data)
        cross_src, cross_dst = cross_edge_index
        cross_edge_attr = self.cross_edge_embedding(cross_edge_attr)

        for i in range(self.n_conv_layers):
            edge_attr_ = torch.cat([edge_attr, x[src, :self.n_s], x[dst, :self.n_s]], dim=-1)
            x_intra_update = self.conv_layers[i](x, edge_index, edge_attr_, edge_sh)

            cross_edge_attr_ = torch.cat([cross_edge_attr, x[cross_src, :self.n_s], x[cross_dst, :self.n_s]], dim=-1)
            x_inter_update = self.cross_conv_layers[i](x, cross_edge_index, cross_edge_attr_, cross_edge_sh)

            x = F.pad(x, (0, x_intra_update.shape[-1] - x.shape[-1]))
            x = x + x_intra_update + x_inter_update

        center_edge_index, center_edge_attr, center_sh = self.setup_center_graph(data)
        center_src, _ = center_edge_index
        center_edge_attr = self.center_edge_embedding(center_edge_attr)
        center_edge_attr_ = torch.cat([center_edge_attr, x[center_src, :self.n_s]], dim=-1)
        
        global_pred = self.final_conv(x, center_edge_index, center_edge_attr_, center_sh,
                                      out_nodes=data.agent_center_pos.size(0))
        
        tr_pred = global_pred[:, :3] + global_pred[:, 6:9]
        rot_pred = global_pred[:, 3:6] + global_pred[:, 9:]

        agent_sigma_emb = self.timestep_emb_fn(data.t_tr)
        
        tr_norm = torch.linalg.vector_norm(tr_pred, dim=1).unsqueeze(1)
        tr_scale = self.tr_final_layer(torch.cat([tr_norm, agent_sigma_emb], dim=1))
        tr_pred = (tr_pred / tr_norm) * tr_scale

        rot_norm = torch.linalg.vector_norm(rot_pred, dim=1).unsqueeze(1)
        rot_scale = self.rot_final_layer(torch.cat([rot_norm, agent_sigma_emb], dim=1))
        rot_pred = (rot_pred / rot_norm) * rot_scale

        if self.scale_by_sigma:
            tr_sigma, rot_sigma = self.t_to_sigma_fn(data.t_tr, data.t_rot)
            tr_pred = tr_pred / tr_sigma.unsqueeze(1)
            rot_pred = rot_pred * so3.score_norm(rot_sigma.cpu()).unsqueeze(1).to(data.agent_center_pos.device)

        return tr_pred, rot_pred

    def setup_graph(self, data):
        data.node_sigma_emb = self.timestep_emb_fn(data.node_t_tr)
        x = torch.cat([data.x, data.node_sigma_emb], dim=-1)

        edge_index = data.edge_index
        src, dst = edge_index
        edge_vec = data.pos[src.long()] - data.pos[dst.long()]

        edge_length_emb = self.dist_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data.node_sigma_emb[dst.long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], dim=-1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalization='component', normalize=True)

        return x, edge_index, edge_attr, edge_sh

    def setup_cross_graph(self, data):
        cross_edge_index = data.cross_edge_index
        cross_src, cross_dst = cross_edge_index
        edge_vec = data.pos[cross_src.long()] - data.pos[cross_dst.long()]

        edge_length_emb = self.cross_dist_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data.node_sigma_emb[cross_dst.long()]
        cross_edge_attr = torch.cat([edge_length_emb, edge_sigma_emb], dim=-1)
        cross_edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalization='component', normalize=True)

        return cross_edge_index, cross_edge_attr, cross_edge_sh
    
    def setup_center_graph(self, data):
        if hasattr(data, 'center_src'):
            center_src = data.center_src.unsqueeze(0)
        else:
            raise ValueError("center_src must have been computed before.")
        center_dst = data.agent_membership.unsqueeze(0)

        edge_index = torch.cat([center_src, center_dst], dim=0)
        src, dst = edge_index

        edge_vec = data.pos[src.long()] - data.agent_center_pos[dst.long()]
        edge_length_emb = self.center_dist_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data.node_sigma_emb[src.long()]

        edge_attr = torch.cat([edge_length_emb, edge_sigma_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return edge_index, edge_attr, edge_sh
