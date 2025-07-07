# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modified code of Openfold's IPA."""
import pdb

import numpy as np
import torch
import math
from scipy.stats import truncnorm
import torch.nn as nn
from typing import Optional, Callable, List, Sequence
from openfold.utils.rigid_utils import Rigid
from openfold.model.dropout import DropoutRowwise, DropoutColumnwise
from openfold.model.primitives import Attention
from openfold.model.pair_transition import PairTransition
from openfold.model.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
)

from data import all_atom


def sctrans_organize(sctrans_tiled: torch.Tensor):
    """

    Args:
        sctrans_tiled: [..., num_scatoms * 3]

    Returns:
        sctrans: [..., num_scatoms, 3]
    """
    sctrans = torch.split(sctrans_tiled, sctrans_tiled.shape[-1] // 3, dim=-1)
    sctrans = torch.stack(sctrans, dim=-1)  # (..., 9, 3)
    return sctrans


def sctrans_tile(sctrans_organize: torch.Tensor):
    return torch.cat([sctrans_organize[..., 0],
                      sctrans_organize[..., 1],
                      sctrans_organize[..., 2]], dim=-1)


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


def zero_init_(module: nn.Module):
    for name, param in module.named_parameters():
        if param.requires_grad:
            param.data.zero_()


def compute_angles(ca_pos, pts):
    batch_size, num_res, num_heads, num_pts, _ = pts.shape
    calpha_vecs = (ca_pos[:, :, None, :] - ca_pos[:, None, :, :]) + 1e-10
    calpha_vecs = torch.tile(calpha_vecs[:, :, :, None, None, :], (1, 1, 1, num_heads, num_pts, 1))
    ipa_pts = pts[:, :, None, :, :, :] - torch.tile(ca_pos[:, :, None, None, None, :],
                                                    (1, 1, num_res, num_heads, num_pts, 1))
    phi_angles = all_atom.calculate_neighbor_angles(
        calpha_vecs.reshape(-1, 3),
        ipa_pts.reshape(-1, 3)
    ).reshape(batch_size, num_res, num_res, num_heads, num_pts)
    return phi_angles


def ca_attn(r: Rigid, mask: torch.Tensor, offset_diag=False):
    L = mask.shape[-1]
    _pred_trans1 = r.get_trans()
    ca_dist = torch.cdist(_pred_trans1, _pred_trans1, p=2)
    ca_dist_out = ca_dist.clone()
    if offset_diag:
        ca_dist += 6 * torch.eye(L)[None, ...].expand_as(ca_dist).to(ca_dist.device)
    pre_attn = -ca_dist + 1e4 * (mask - 1)
    attn = torch.nn.functional.softmax(pre_attn, dim=-1)
    return (attn * mask).detach(), ca_dist_out.detach()



def sc_attn(r: Rigid, sc_trans: torch.Tensor,
            mask: torch.Tensor, offset_diag=False):
    sctrans_abs = r[..., None].apply(sc_trans)
    B, L, A, C = sctrans_abs.shape
    structure_flat = sctrans_abs.view(B, L * A, C)  # Shape: [B, L*A, 3]
    atom_distances = torch.cdist(structure_flat, structure_flat, p=2)  # Shape: [B, L*A, L*A]
    atom_distances = atom_distances.view(B, L, A, L, A)
    atom_distances = torch.min(atom_distances, dim=4).values
    atom_distances = torch.min(atom_distances, dim=2).values
    if offset_diag:
        atom_distances += 6 * torch.eye(L)[None, ...].expand_as(atom_distances).to(sc_trans.device)
    pre_attn = -atom_distances + 1e4 * (mask - 1)
    attn = torch.nn.functional.softmax(pre_attn, dim=-1)
    return (attn * mask).detach()


def bin_dist_mat(dist_mat: torch.tensor, mask: torch.tensor):
    # 22 bins. 20 bins between [2, 20], 1 bin is < 2, 1 bin is > 20
    bin_edges = torch.tensor([2] + list(range(3, 21)) + [20]).to(dist_mat.device)
    binned_matrix = torch.bucketize(dist_mat, bin_edges)
    binned_matrix = torch.where(mask.bool(), binned_matrix, 21)
    return binned_matrix



class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            bias: bool = True,
            init: str = "default",
            init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init_fn is not None:
            init_fn(self.weight, self.bias)
        else:
            if init == "default":
                lecun_normal_init_(self.weight)
            elif init == "relu":
                he_normal_init_(self.weight)
            elif init == "glorot":
                glorot_uniform_init_(self.weight)
            elif init == "gating":
                gating_init_(self.weight)
                if bias:
                    with torch.no_grad():
                        self.bias.fill_(1.0)
            elif init == "normal":
                normal_init_(self.weight)
            elif init == "final":
                final_init_(self.weight)
            else:
                raise ValueError("Invalid init string.")


class StructureModuleTransition(nn.Module):
    def __init__(self, c):
        super(StructureModuleTransition, self).__init__()

        self.c = c

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(self.c)

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)
        s = s + s_initial
        s = self.ln(s)

        return s


class EdgeTransition(nn.Module):
    def __init__(
            self,
            *,
            node_embed_size,
            edge_embed_in,
            edge_embed_out,
            num_layers=2,
            node_dilation=2
    ):
        super(EdgeTransition, self).__init__()

        bias_embed_size = node_embed_size // node_dilation
        self.initial_embed = Linear(
            node_embed_size, bias_embed_size, init="relu")
        hidden_size = bias_embed_size * 2 + edge_embed_in
        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers.append(Linear(hidden_size, hidden_size, init="relu"))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)
        self.final_layer = Linear(hidden_size, edge_embed_out, init="final")
        self.layer_norm = nn.LayerNorm(edge_embed_out)

    def forward(self, node_embed, edge_embed):
        node_embed = self.initial_embed(node_embed)
        batch_size, num_res, _ = node_embed.shape
        edge_bias = torch.cat([
            torch.tile(node_embed[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(node_embed[:, None, :, :], (1, num_res, 1, 1)),
        ], axis=-1)
        edge_embed = torch.cat(
            [edge_embed, edge_bias], axis=-1).reshape(
            batch_size * num_res ** 2, -1)
        edge_embed = self.final_layer(self.trunk(edge_embed) + edge_embed)
        edge_embed = self.layer_norm(edge_embed)
        edge_embed = edge_embed.reshape(
            batch_size, num_res, num_res, -1
        )
        return edge_embed


class GraphTriangleBlock(nn.Module):
    # Proteus
    def __init__(
            self,
            c_s,
            c_z,
            c_rbf,
            c_gate_s,
            c_hidden,
            c_hidden_mul,
            no_heads,
            transition_n,
            k_neighbour,
            k_linear,
            inf,
            pair_dropout,
            **kwargs,
    ):
        super(GraphTriangleBlock, self).__init__()
        self.embed_size = (c_s // 2) * 2 + c_z
        self.no_heads = no_heads
        self.c_hidden = c_hidden
        self.c_rbf = c_rbf
        self.k_neighbour = k_neighbour
        self.k_linear = k_linear
        self.inf = inf
        self.NM_TO_ANG_SCALE = 10.0

        self.proj_left = Linear(c_s, c_gate_s)
        self.proj_right = Linear(c_s, c_gate_s)
        self.to_gate = Linear(c_gate_s * c_gate_s, c_z, init="gating")

        self.emb_rbf = nn.Linear(c_rbf, c_z)
        self.to_bias = Linear(c_z, self.no_heads, bias=False, init="normal")

        self.tri_mul_out = TriangleMultiplicationOutgoing(c_z, c_hidden_mul, )
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z, c_hidden_mul, )

        self.mha_start = Attention(c_z, c_z, c_z, self.c_hidden, self.no_heads)
        self.mha_end = Attention(c_z, c_z, c_z, self.c_hidden, self.no_heads)
        self.pair_transition = PairTransition(
            c_z,
            transition_n,
        )

        self.dropout_row_layer = DropoutRowwise(pair_dropout)
        self.dropout_col_layer = DropoutColumnwise(pair_dropout)

        self.layer_norm = nn.LayerNorm(c_z)

    def local_mha(self, x, rigids, num_neighbour, num_linear, triangle_bias, mask, starting_node):
        '''
        Args:
            x: [batch_size, residue_num, residue_num, embed_size]
            rigids: [batch_size, residue_num, 3, 3]
            num_neighbour: int
            num_linear: int
            triangle_bias: [batch_size, residue_num, residue_num, num_heads]
            mask: [batch_size, residue_num, residue_num]
            starting_node: bool
        Returns:
            x: [batch_size, residue_num, residue_num, embed_size]
        '''
        B, N, _, D = x.size()
        B, N, _, H = triangle_bias.size()

        out_x = torch.zeros_like(x)
        coords = rigids.get_trans()

        if not starting_node:
            x = x.transpose(-2, -3)
            triangle_bias = triangle_bias.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        # [batch_size, residue_num, num_neighbour]
        indices = self.knn_indices(coords, num_neighbour, num_linear, pair_mask=mask)
        num_neighbour = num_neighbour + num_linear

        x = torch.gather(
            x, dim=2, index=indices.unsqueeze(-1).expand(B, N, num_neighbour, D)
        )
        x = self.layer_norm(x)
        # [B, I, K]
        mask = torch.gather(
            mask, dim=2, index=indices
        )
        # [B, I, 1, 1, K]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        # [B, I, J, K, H]
        triangle_bias = triangle_bias.unsqueeze(-2).expand(B, N, N, N, H, )
        # [B, I, k, K, H]
        triangle_bias = torch.gather(
            triangle_bias, dim=2, index=indices[..., None, None].expand((B, N, num_neighbour, N, H,))
        )
        # [B, I, k, k, H]
        triangle_bias = torch.gather(
            triangle_bias, dim=3, index=indices[..., None, :, None].expand((B, N, num_neighbour, num_neighbour, H,))
        )
        # [B, I, H, k, k]
        triangle_bias = permute_final_dims(triangle_bias, (2, 1, 0))

        biases = [mask_bias, triangle_bias]

        if starting_node:
            x = self.mha_start(q_x=x, kv_x=x, biases=biases)
        else:
            x = self.mha_end(q_x=x, kv_x=x, biases=biases)

        out_x = out_x.scatter(2, indices.unsqueeze(-1).expand(B, N, num_neighbour, D), x)

        if not starting_node:
            out_x = out_x.transpose(-2, -3)

        return out_x

    def knn_indices(self, x, num_neighbour, num_linear, pair_mask=None):
        _, nres = x.shape[:2]

        # Warning : we advise to use this commented no bug line for new model, we left the buggy line to keep with the original trained model.
        # distances = torch.norm(x.unsqueeze(2) - x.unsqueeze(1), dim=-1) * self.NM_TO_ANG_SCALE
        distances = torch.norm(x.unsqueeze(2) - x.unsqueeze(1), dim=-1)
        distances[:, torch.arange(0, nres, dtype=torch.long), torch.arange(0, nres, dtype=torch.long)] = self.inf

        # set distance between linear neighbour to 0
        for i in range(1, num_linear // 2 + 1):
            row_indices = torch.arange(0, nres, dtype=torch.long)
            indices = torch.arange(i, nres + i, dtype=torch.long)
            distances[:, row_indices[:nres - i], indices[:nres - i]] = 0
            indices = torch.arange(i * -1, nres - i, dtype=torch.long)
            distances[:, row_indices[i:], indices[i:]] = 0

        if pair_mask is not None:
            distances = distances + (self.inf * (pair_mask - 1))

        _, indices = torch.topk(distances, num_neighbour + num_linear, dim=-1, largest=False)  # Shape: [B, N, K]
        return indices

    def rbf(self, D, D_min=0.0, D_sigma=0.5):
        # Distance radial basis function
        D_max = D_min + (self.c_rbf - 1) * D_sigma
        D_mu = torch.linspace(D_min, D_max, self.c_rbf).to(D.device)
        D_mu = D_mu[None, :]
        D_expand = torch.unsqueeze(D, -1)
        rbf_feat = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        rbf_feat = self.emb_rbf(rbf_feat)
        return rbf_feat

    def forward(self, node_embed, edge_embed, rigids, edge_mask):

        batch_size, num_res, _ = node_embed.shape

        # get pair bias from rbf of distance
        coords = rigids.get_trans()
        distances = torch.norm(coords.unsqueeze(2) - coords.unsqueeze(1), dim=-1)
        bias = self.rbf(distances)

        # gate pair bias with sequence embedding
        left = self.proj_left(node_embed)
        right = self.proj_right(node_embed)
        gate = torch.einsum('bli,bmj->blmij', left, right).reshape(batch_size, num_res, num_res, -1)
        gate = torch.sigmoid(self.to_gate(gate))
        bias = bias * gate
        # pair bias shape : [B,N,N,h]
        bias = self.to_bias(bias)

        z = edge_embed
        z = z + self.dropout_row_layer(self.tri_mul_out(z, mask=edge_mask))
        z = z + self.dropout_row_layer(self.tri_mul_in(z, mask=edge_mask))
        z = z + self.dropout_row_layer(
            self.local_mha(z, rigids, self.k_neighbour, self.k_linear, triangle_bias=bias, mask=edge_mask,
                           starting_node=True))
        z = z + self.dropout_col_layer(
            self.local_mha(z, rigids, self.k_neighbour, self.k_linear, triangle_bias=bias, mask=edge_mask,
                           starting_node=False))

        return z


class InvariantPointAttention(nn.Module):
    """
    Implements Algorithm 22.
    """

    def __init__(
            self,
            ipa_conf,
            inf: float = 1e5,
            eps: float = 1e-8,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super(InvariantPointAttention, self).__init__()
        self._ipa_conf = ipa_conf

        self.c_s = ipa_conf.c_s
        self.c_z = ipa_conf.c_z
        self.c_hidden = ipa_conf.c_hidden
        self.no_heads = ipa_conf.no_heads
        self.no_qk_points = ipa_conf.no_qk_points
        self.no_v_points = ipa_conf.no_v_points
        self.inf = inf
        self.eps = eps

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        self.linear_b = Linear(self.c_z, self.no_heads)
        self.down_z = Linear(self.c_z, self.c_z // 4)

        self.head_weights = nn.Parameter(torch.zeros((ipa_conf.no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = (
                self.c_z // 4 + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
            self,
            s: torch.Tensor,
            z: Optional[torch.Tensor],
            r: Rigid,
            mask: torch.Tensor,
            _offload_inference: bool = False,
            _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        if _offload_inference:
            z = _z_reference_list
        else:
            z = [z]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        )

        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)

        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)

        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z[0])

        if (_offload_inference):
            z[0] = z[0].cpu()

        # [*, H, N_res, N_res], ALg 22 Line 7
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))

        # [*, N_res, N_res, H, P_q, 3]
        pt_displacement = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_displacement ** 2

        # [*, N_res, N_res, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )  # Alg 22, Line 5, w_C
        pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        a = a + pt_att
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)
        # DEBUG
        # attn_matrix = tmp.mean(dim=1)  # torch.topk(attn_matrix[0, 12, -9:], k=5) 3rm0_I [-7, -29, -11:]
        # pdb.set_trace()
        # DEBUG
        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(
            a, v.transpose(-2, -3)
        ).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, H, 3, N_res, P_v] 
        o_pt = torch.sum(
            (
                    a[..., None, :, :, None]
                    * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
            ),
            dim=-2,
        )

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(
            o_pt_dists, 2)

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        if (_offload_inference):
            z[0] = z[0].to(o_pt.device)

        # [*, N_res, H, C_z // 4]
        pair_z = self.down_z(z[0])
        o_pair = torch.matmul(a.transpose(-2, -3), pair_z)

        # [*, N_res, H * C_z // 4]
        o_pair = flatten_final_dims(o_pair, 2)

        o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair]

        # [*, N_res, C_s]
        s = self.linear_out(
            torch.cat(
                o_feats, dim=-1
            )
        )

        return s


class scIPA(nn.Module):
    """
    Implements Algorithm 22.
    """

    def __init__(
            self,
            ipa_conf,
            inf: float = 1e5,
            eps: float = 1e-8,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super(scIPA, self).__init__()
        self._ipa_conf = ipa_conf

        self.c_s = ipa_conf.c_s
        self.c_z = ipa_conf.c_z
        self.c_hidden = ipa_conf.c_hidden
        self.no_heads = ipa_conf.no_heads
        self.no_qk_points = ipa_conf.no_qk_points
        self.no_v_points = ipa_conf.no_v_points
        self.inf = inf
        self.eps = eps

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)
        # self.linear_sc = Linear(27, 27)  # Soften the sctrans
        self.linear_sc = nn.Sequential(
            nn.Linear(27, self.c_s), nn.ReLU(),
            nn.Linear(self.c_s, self.c_s), nn.ReLU(),
            nn.Linear(self.c_s, 27)
        )

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        self.linear_b = Linear(self.c_z, self.no_heads)
        self.down_z = Linear(self.c_z, self.c_z // 4)

        self.head_weights = nn.Parameter(torch.zeros((ipa_conf.no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = (
                self.c_z // 4 + self.c_hidden + self.no_v_points * 4
        )

        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s, init="final")
        # self.linear_out = Linear(self.no_heads * concat_out_dim + 72 * 3, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
            self,
            s: torch.Tensor,
            z: Optional[torch.Tensor],
            r: Rigid,
            mask: torch.Tensor,
            sctrans: torch.Tensor,
            _offload_inference: bool = False,
            _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        if _offload_inference:
            z = _z_reference_list
        else:
            z = [z]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        )

        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)

        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)

        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z[0])

        if (_offload_inference):
            z[0] = z[0].cpu()

        # [*, H, N_res, N_res], ALg 22 Line 7
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))

        # [*, N_res, N_res, H, P_q, 3]
        pt_displacement = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_displacement ** 2
        # pt_att = torch.sqrt(pt_displacement ** 2 + self.eps)

        # [*, N_res, N_res, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )  # Alg 22, Line 5, w_C
        pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))


        # Side-chain attention
        sctrans_soft = self.linear_sc(sctrans_tile(sctrans))
        sctrans_abs = r[..., None].apply(sctrans_organize(sctrans_soft))  # [*, N_res, 9, 3]
        # sctrans_abs = r[..., None].apply(sctrans)

        # sc_displacement = sctrans_abs.unsqueeze(-4) - sctrans_abs.unsqueeze(-3)
        # sc_att, _ = torch.sqrt(torch.sum(sc_displacement**2, dim=-1) + self.eps).min(dim=-1)
        # Calculate the minimum distance between two residues
        B, L, A, C = sctrans_abs.shape
        structure_flat = sctrans_abs.view(B, L * A, C)  # Shape: [B, L*A, 3]
        atom_distances = torch.cdist(structure_flat, structure_flat, p=2)  # Shape: [B, L*A, L*A]
        atom_distances = atom_distances.view(B, L, A, L, A)
        min_distances_step1 = torch.min(atom_distances, dim=4).values
        sc_att = torch.min(min_distances_step1, dim=2).values
        sc_att += 4.5 * torch.eye(sc_att.shape[-1])[None, ...].expand_as(sc_att).to(sc_att.device)
        # 4.5 A is a threshold of interaction
        sc_att = sc_att[:, None, ...].expand_as(a)
        """"""

        # TODO: Need to balance between them, or we will discard pt_att
        # a = a + pt_att
        a = a - sc_att * 0.1
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden], Alg 22, Line 9
        o = torch.matmul(
            a, v.transpose(-2, -3)
        ).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, H, 3, N_res, P_v], Alg 22, Line 10
        o_pt = torch.sum(
            (
                    a[..., None, :, :, None]
                    * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
            ),
            dim=-2,
        )

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(
            o_pt_dists, 2)

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        if (_offload_inference):
            z[0] = z[0].to(o_pt.device)

        # [*, N_res, H, C_z // 4]
        pair_z = self.down_z(z[0])
        o_pair = torch.matmul(a.transpose(-2, -3), pair_z)
        # Alg 22, Line 8

        # [*, N_res, H * C_z // 4]
        o_pair = flatten_final_dims(o_pair, 2)

        # TODO: Add sctrans into o
        """
        o_sc = torch.sum(
            (
                    a[..., None, :, :, None]
                    * permute_final_dims(sctrans[:, :, None].repeat(1, 1, self.no_heads, 1, 1), (1, 3, 0, 2))[..., None, :, :]
            ),
            dim=-2,
        )

        o_sc = permute_final_dims(o_sc, (2, 0, 3, 1))
        o_sc = r[..., None, None].invert_apply(o_sc)
        o_sc = o_sc.reshape(*o_sc.shape[:-3], -1, 3)
        o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair, *torch.unbind(o_sc, dim=-1)]
        """
        # TODO: Add sctrans into o

        o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair]

        # [*, N_res, C_s]
        s = self.linear_out(
            torch.cat(
                o_feats, dim=-1
            )
        )

        return s


class TorsionAngles(nn.Module):
    def __init__(self, c, num_torsions, eps=1e-8):
        super(TorsionAngles, self).__init__()

        self.c = c
        self.eps = eps
        self.num_torsions = num_torsions

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        # TODO: Remove after published checkpoint is updated without these weights.
        self.linear_3 = Linear(self.c, self.c, init="final")
        self.linear_final = Linear(
            self.c, self.num_torsions * 2, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)

        s = s + s_initial
        unnormalized_s = self.linear_final(s)
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(unnormalized_s ** 2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        normalized_s = unnormalized_s / norm_denom

        return unnormalized_s, normalized_s


class RotationVFLayer(nn.Module):
    def __init__(self, dim):
        super(RotationVFLayer, self).__init__()

        self.linear_1 = Linear(dim, dim, init="relu")
        self.linear_2 = Linear(dim, dim, init="relu")
        self.linear_3 = Linear(dim, dim)
        self.final_linear = Linear(dim, 6, init="final")
        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)
        s = s + s_initial
        return self.final_linear(s)


class BackboneUpdate(nn.Module):
    """
    Implements part of Algorithm 23.
    """

    def __init__(self, c_s, use_rot_updates):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super(BackboneUpdate, self).__init__()

        self.c_s = c_s
        self._use_rot_updates = use_rot_updates
        update_dim = 6 if use_rot_updates else 3
        self.linear = Linear(self.c_s, update_dim, init="final")

    def forward(self, s: torch.Tensor):
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector 
        """
        # [*, 6]
        update = self.linear(s)

        return update


class SideChainUpdate(nn.Module):
    def __init__(self, c):
        super(SideChainUpdate, self).__init__()

        self.c = c

        # self.linear_0 = Linear(27, self.c, init="relu")
        # self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_1 = Linear(self.c + 27, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, 27, init="relu")
        # self.linear_node = Linear(self.c, self.c, init="relu")
        self.relu = nn.ReLU()

    def forward(self, sctrans, s):
        """

        Args:
            sctrans: sctrans, [..., 9, 3]
            s:  [..., self.c]

        Returns:
            sctrans_updated: [..., 9, 3]
        """
        sctrans_t = sctrans_tile(sctrans)
        sctrans_t = torch.cat([s, sctrans_t], dim=-1)
        # sctrans_t = self.relu(sctrans_t)
        # sctrans_t = self.linear_2(sctrans_t)
        # sctrans_t = self.relu(sctrans_t)
        sctrans_t = self.linear_1(sctrans_t)
        sctrans_t = self.relu(sctrans_t)
        sctrans_t = self.linear_2(sctrans_t)
        sctrans_t = self.relu(sctrans_t)
        sctrans_t = self.linear_3(sctrans_t)

        return sctrans + sctrans_organize(sctrans_t)


class IpaScore(nn.Module):

    def __init__(self, model_conf, diffuser):
        super(IpaScore, self).__init__()
        self._model_conf = model_conf
        ipa_conf = model_conf.ipa
        self._ipa_conf = ipa_conf
        self.diffuser = diffuser

        self.scale_pos = lambda x: x * ipa_conf.coordinate_scaling
        self.scale_rigids = lambda x: x.apply_trans_fn(self.scale_pos)

        self.unscale_pos = lambda x: x / ipa_conf.coordinate_scaling
        self.unscale_rigids = lambda x: x.apply_trans_fn(self.unscale_pos)
        self.trunk = nn.ModuleDict()

        for b in range(ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = InvariantPointAttention(ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(ipa_conf.c_s)
            self.trunk[f'skip_embed_{b}'] = Linear(
                self._model_conf.node_embed_size,
                self._ipa_conf.c_skip,
                init="final"
            )
            tfmr_in = ipa_conf.c_s + self._ipa_conf.c_skip
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, ipa_conf.seq_tfmr_num_layers)
            self.trunk[f'post_tfmr_{b}'] = Linear(
                tfmr_in, ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = StructureModuleTransition(
                c=ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = BackboneUpdate(ipa_conf.c_s)

            if b < ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f'edge_transition_{b}'] = EdgeTransition(
                    node_embed_size=ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )

        self.torsion_pred = TorsionAngles(ipa_conf.c_s, 1)

    def forward(self, init_node_embed, edge_embed, input_feats):
        node_mask = input_feats['res_mask'].type(torch.float32)
        diffuse_mask = (1 - input_feats['fixed_mask'].type(torch.float32)) * node_mask
        edge_mask = node_mask[..., None] * node_mask[..., None, :]
        init_frames = input_feats['rigids_t'].type(torch.float32)

        curr_rigids = Rigid.from_tensor_7(torch.clone(init_frames))
        init_rigids = Rigid.from_tensor_7(init_frames)
        init_rots = init_rigids.get_rots()

        # Main trunk
        curr_rigids = self.scale_rigids(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_in = torch.cat([
                node_embed, self.trunk[f'skip_embed_{b}'](init_node_embed)
            ], dim=-1)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                seq_tfmr_in, src_key_padding_mask=1 - node_mask)
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * diffuse_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, diffuse_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]
        rot_score = self.diffuser.calc_rot_score(
            init_rigids.get_rots(),
            curr_rigids.get_rots(),
            input_feats['t']
        )
        rot_score = rot_score * node_mask[..., None]

        curr_rigids = self.unscale_rigids(curr_rigids)
        trans_score = self.diffuser.calc_trans_score(
            init_rigids.get_trans(),
            curr_rigids.get_trans(),
            input_feats['t'][:, None, None],
            use_torch=True,
        )
        trans_score = trans_score * node_mask[..., None]
        _, psi_pred = self.torsion_pred(node_embed)
        model_out = {
            'psi': psi_pred,
            'rot_score': rot_score,
            'trans_score': trans_score,
            'final_rigids': curr_rigids,
        }
        return model_out
