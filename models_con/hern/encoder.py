import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import pdb
from models_con.hern.utils import *
from models_con.hern.nnutils import *
from models_con.hern.data import ALPHABET, ATOM_TYPES
from models_con.hern.protein_features import ProteinFeatures
from models_con.torsion import full_atom_reconstruction
from pepflow.modules.protein.constants import AA, restype_to_heavyatom_names


class EGNNEncoder(nn.Module):

    def __init__(self,
                 hidden_size,
                 k_neighbors,
                 num_rbf,
                 dropout,
                 depth,
                 node_hdim=0,
                 features_type='backbone',
                 update_X=True):
        super(EGNNEncoder, self).__init__()
        self.update_X = update_X
        self.features_type = features_type
        self.features = ProteinFeatures(
            top_k=k_neighbors, num_rbf=num_rbf,
            features_type=features_type,
            direction='bidirectional'
        )
        self.node_in, self.edge_in = self.features.feature_dimensions[features_type]
        self.node_in += node_hdim

        self.W_v = nn.Linear(self.node_in, hidden_size)
        self.W_e = nn.Linear(self.edge_in, hidden_size)
        self.layers = nn.ModuleList([
            MPNNLayer(hidden_size, hidden_size * 3, dropout=dropout)
            for _ in range(depth)
        ])
        if self.update_X:
            self.W_x = nn.Linear(hidden_size, hidden_size)
            self.U_x = nn.Linear(hidden_size, hidden_size)
            self.T_x = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, 14))

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    # [backbone] X: [B,N,L,3], V/S: [B,N,H], A: [B,N,L]
    # [atom] X: [B,N*L,3], V/S: [B,N*L,H], A: [B,N*L]
    def forward(self, X, V, S, A):
        mask = A.clamp(max=1).float()
        vmask = mask[:, :, 1] if self.features_type == 'backbone' else mask
        _, E, E_idx = self.features(X, vmask)

        h = self.W_v(V)  # [B, N, H]
        h_e = self.W_e(E)  # [B, N, K, H] 
        nei_s = gather_nodes(S, E_idx)  # [B, N, K, H]
        emask = gather_nodes(vmask[..., None], E_idx).squeeze(-1)

        # message passing
        for layer in self.layers:
            nei_v = gather_nodes(h, E_idx)  # [B, N, K, H]
            nei_h = torch.cat([nei_v, nei_s, h_e], dim=-1)
            h = layer(h, nei_h, mask_attend=emask)  # [B, N, H]
            h = h * vmask.unsqueeze(-1)  # [B, N, H]

        if self.update_X and self.features_type == 'backbone':
            ca_mask = mask[:, :, 1]  # [B, N]
            mij = self.W_x(h).unsqueeze(2) + self.U_x(h).unsqueeze(1)  # [B,N,N,H]
            xij = X.unsqueeze(2) - X.unsqueeze(1)  # [B,N,N,L,3]
            xij = xij * self.T_x(mij).unsqueeze(-1)  # [B,N,N,L,3]
            f = torch.sum(xij * ca_mask[:, None, :, None, None], dim=2)  # [B,N,N,L,3] * [B,1,N,1,1]
            f = f / (1e-6 + ca_mask.sum(dim=1)[:, None, None, None])  # [B,N,L,3] / [B,1,1,1]
            X = X + f.clamp(min=-20.0, max=20.0)

        return h, X * mask[..., None]


class HierEGNNEncoder(nn.Module):
    def __init__(self, hidden_size=128,
                 k_neighbors=9,
                 depth=4,
                 clash_step=10,
                 num_rbf=16,
                 dropout=0.1,
                 update_X=False,
                 backbone_CA_only=False):
        super(HierEGNNEncoder, self).__init__()
        self.update_X = update_X
        self.backbone_CA_only = backbone_CA_only
        self.clash_step = clash_step
        atom_hidden_size = hidden_size // 4
        self.residue_mpn = EGNNEncoder(
            hidden_size=hidden_size,
            k_neighbors=k_neighbors,
            num_rbf=num_rbf,
            dropout=dropout,
            depth=depth,
            features_type='backbone',
            node_hdim=atom_hidden_size,
            update_X=update_X,
        )
        self.atom_mpn = EGNNEncoder(
            hidden_size=atom_hidden_size,
            k_neighbors=k_neighbors,
            num_rbf=num_rbf,
            dropout=dropout,
            depth=depth,
            features_type='atom',
            node_hdim=atom_hidden_size,
            update_X=update_X,
        )
        if self.update_X:
            # backbone coord update
            self.W_x = nn.Linear(hidden_size, hidden_size)
            self.U_x = nn.Linear(hidden_size, hidden_size)
            self.T_x = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, 4))
            # side chain coord update
            self.W_a = nn.Linear(atom_hidden_size, atom_hidden_size)
            self.U_a = nn.Linear(atom_hidden_size, atom_hidden_size)
            self.T_a = nn.Sequential(nn.ReLU(), nn.Linear(atom_hidden_size, 1))

        self.embedding = nn.Embedding(len(ATOM_TYPES), atom_hidden_size)
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    '''
    Input:
    >>> X: [B,N,L,3], all atom's coordinate
    >>> V: [B,N,6], each residue's chemical properties
    >>> S: [B,N,H], residue feature
    >>> A: [B,N,L], atom name
    '''

    # def forward(self, X, V, S, A):
    def forward(self, rot, trans, angle, sctrans,
                seq, res_mask, gen_mask, t_emb, node_init):
        # Prepare the inputs
        if angle.shape[-1] != 5:
            angle = F.pad(angle, (0, 4))
        pos_heavyatom, _, _ = full_atom_reconstruction(R_bb=rot, t_bb=trans, angles=angle,
                                                       aa=seq * res_mask)
        B, N, L = pos_heavyatom.size()[:3]
        # (B,L,14,3), instead of 15, ignore OXT masked
        # TODO: Reset the atom-wise emb
        sctrans_copy = copy.deepcopy(sctrans)
        # sctrans_copy[:, :, :, :] = 0
        # TODO: Reset the atom-wise emb
        pos_heavyatom[:, :, 5:] = torch.einsum('blij,blkj->blik', sctrans_copy,
                                               rot) + trans[:, :, None]  # abs coor.

        A = torch.zeros((B, N, L), dtype=torch.long, device=rot.device)

        for i in range(B):
            for j in range(N):
                try:
                    aa_type = AA(seq[i, j].item())
                    atom_names = restype_to_heavyatom_names.get(aa_type, [''] * 15)[:-1]
                except ValueError:
                    atom_names = [''] * 14  # for aa idx 21

                for k, atom_name in enumerate(atom_names):
                    # atoms_tensor[i, j, k] = atom_to_index.get(atom_name, 0)
                    A[i, j, k] = ATOM_TYPES.index(atom_name)
        A_pad = torch.tensor([1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0], device=A.device)
        A_pad[5:] = len(ATOM_TYPES) - 1
        A_pad = A_pad[None, None, :].expand_as(A)
        A = torch.where(gen_mask[..., None] == 1, A_pad, A)
        # Prepare the inputs


        X = pos_heavyatom
        V = t_emb  # t_emb
        S = node_init  # residue_emb
        # A = 0  # atom name
        # mask = 0

        # B, N, L = X.size()[:3]
        X_atom = X.view(B, N * L, 3)
        mask = A.clamp(max=1).float()  # Atom mask, set the pos of real atom as 1.

        # atom message passing
        h_atom = self.embedding(A).view(B, N * L, -1)
        # initialize atom feat. as atom type emb.
        h_atom, _ = self.atom_mpn(X_atom, h_atom, h_atom, A.view(B, -1))
        h_atom = h_atom.view(B, N, L, -1)
        h_atom = h_atom * mask[..., None]
        h_A = h_atom.sum(dim=-2) / (1e-6 + mask.sum(dim=-1)[..., None])

        # TODO: Reset the atom-wise emb
        # h_A = torch.zeros_like(h_A, device=A.device).detach()
        # TODO: Reset the atom-wise emb

        # residue message passing
        h_V = torch.cat([V, h_A], dim=-1)
        h_res, _ = self.residue_mpn(X, h_V, S, A)

        if self.update_X:
            # backbone update
            bb_mask = mask[:, :, :4]  # [B, N, 4]
            X_bb = X[:, :, :4]  # backbone atoms
            mij = self.W_x(h_res).unsqueeze(2) + self.U_x(h_res).unsqueeze(1)  # [B,N,N,H]
            xij = X_bb.unsqueeze(2) - X_bb.unsqueeze(1)  # [B,N,N,4,3]
            dij = xij.norm(dim=-1)  # [B,N,N,4]
            fij = torch.maximum(self.T_x(mij), 3.8 - dij)  # break term [B,N,N,4]
            xij = xij * fij.unsqueeze(-1)
            f_res = torch.sum(xij * bb_mask[:, None, :, :, None], dim=2)  # [B,N,N,4,3] * [B,1,N,4,1] -> [B,N,4,3]
            f_res = f_res / (1e-6 + bb_mask.sum(dim=1, keepdims=True)[..., None])  # [B,N,4,3]
            X_bb = X_bb + f_res.clamp(min=-20.0, max=20.0)

            # Clash correction
            for _ in range(self.clash_step):
                xij = X_bb.unsqueeze(2) - X_bb.unsqueeze(1)  # [B,N,N,4,3]
                dij = xij.norm(dim=-1)  # [B,N,N,4]
                fij = F.relu(3.8 - dij)  # repulsion term [B,N,N,4]
                xij = xij * fij.unsqueeze(-1)
                f_res = torch.sum(xij * bb_mask[:, None, :, :, None], dim=2)  # [B,N,N,4,3] * [B,1,N,4,1] -> [B,N,4,3]
                f_res = f_res / (1e-6 + bb_mask.sum(dim=1, keepdims=True)[..., None])  # [B,N,4,3]
                X_bb = X_bb + f_res.clamp(min=-20.0, max=20.0)

            # side chain update
            mij = self.W_a(h_atom).unsqueeze(3) + self.U_a(h_atom).unsqueeze(2)  # [B,N,L,1,H] + [B,N,1,L,H]
            xij = X.unsqueeze(3) - X.unsqueeze(2)  # [B,N,L,1,3] - [B,N,1,L,3]
            dij = xij.norm(dim=-1)  # [B,N,L,L]
            fij = torch.maximum(self.T_a(mij).squeeze(-1), 1.5 - dij)  # break term [B,N,L,L]
            xij = xij * fij.unsqueeze(-1)  # [B,N,L,L,3]
            f_atom = torch.sum(xij * mask[:, :, None, :, None], dim=3)  # [B,N,L,L,3] * [B,N,1,L,1] -> [B,N,L,3]
            X_sc = X + 0.1 * f_atom

            if self.backbone_CA_only:
                X = torch.cat((X_sc[:, :, :1], X_bb[:, :, 1:2], X_sc[:, :, 2:]), dim=2)
            else:
                X = torch.cat((X_bb[:, :, :4], X_sc[:, :, 4:]), dim=2)

        if not self.update_X:
            return h_res
        else:
            return h_res, X * mask[..., None]
