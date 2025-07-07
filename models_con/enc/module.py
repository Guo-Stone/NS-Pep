# Modified from PepGLAD
# !/usr/bin/python
# -*- coding:utf-8 -*-
import math
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from models_con.torsion import full_atom_reconstruction, get_heavyatom_mask
from pepflow.modules.common.geometry import reconstruct_backbone
from .dyMEAN.modules.am_egnn import AMEGNN  # adaptive-multichannel egnn
from .dyMEAN.nn_utils import SeparatedAminoAcidFeature, ProteinFeature, VOCAB


def create_encoder(
        name,
        atom_embed_size,
        embed_size,
        hidden_size,
        n_channel,
        n_layers,
        dropout,
        n_rbf,
        cutoff
):
    if name == 'dyMEAN':
        encoder = AMEGNN(
            embed_size, hidden_size, hidden_size, n_channel,
            channel_nf=atom_embed_size, radial_nf=hidden_size,
            in_edge_nf=0, n_layers=n_layers, residual=True,
            dropout=dropout, dense=False, n_rbf=n_rbf, cutoff=cutoff)
    else:
        raise NotImplementedError(f'Encoder {name} not implemented')

    return encoder


def variadic_arange(size):
    """
    from https://torchdrug.ai/docs/_modules/torchdrug/layers/functional/functional.html#variadic_arange

    Return a 1-D tensor that contains integer intervals of variadic sizes.
    This is a variadic variant of ``torch.arange(stop).expand(batch_size, -1)``.

    Suppose there are :math:`N` intervals.

    Parameters:
        size (LongTensor): size of intervals of shape :math:`(N,)`
    """
    starts = size.cumsum(0) - size

    range = torch.arange(size.sum(), device=size.device)
    range = range - starts.repeat_interleave(size)
    return range


def variadic_meshgrid(input1, size1, input2, size2):
    """
    from https://torchdrug.ai/docs/_modules/torchdrug/layers/functional/functional.html#variadic_meshgrid
    Compute the Cartesian product for two batches of sets with variadic sizes.

    Suppose there are :math:`N` sets in each input,
    and the sizes of all sets are summed to :math:`B_1` and :math:`B_2` respectively.

    Parameters:
        input1 (Tensor): input of shape :math:`(B_1, ...)`
        size1 (LongTensor): size of :attr:`input1` of shape :math:`(N,)`
        input2 (Tensor): input of shape :math:`(B_2, ...)`
        size2 (LongTensor): size of :attr:`input2` of shape :math:`(N,)`

    Returns
        (Tensor, Tensor): the first and the second elements in the Cartesian product
    """
    grid_size = size1 * size2
    local_index = variadic_arange(grid_size)
    local_inner_size = size2.repeat_interleave(grid_size)
    offset1 = (size1.cumsum(0) - size1).repeat_interleave(grid_size)
    offset2 = (size2.cumsum(0) - size2).repeat_interleave(grid_size)
    index1 = torch.div(local_index, local_inner_size, rounding_mode="floor") + offset1
    index2 = local_index % local_inner_size + offset2
    return input1[index1], input2[index2]


class Encoder(nn.Module):
    def __init__(
            self,
            embed_size=128,
            hidden_size=128,
            # latent_size=8,
            n_channel=14,
            # latent_n_channel=1,
            # mask_id=VOCAB.get_mask_idx(),
            # latent_id=VOCAB.symbol_to_idx(VOCAB.LAT),
            max_position=2048,
            relative_position=True,
            # CA_channel_idx=VOCAB.backbone_atoms.index('CA'),
            n_layers=3,
            dropout=0.1,
            # mask_ratio=0.0,
            # fix_alpha_carbon=False,
            # h_kl_weight=0.1,
            # z_kl_weight=0.5,
            # coord_loss_weights={
            #     'Xloss': 1.0,
            #     'ca_Xloss': 0.0,
            #     'bb_bond_lengths_loss': 1.0,
            #     'sc_bond_lengths_loss': 1.0,
            #     'bb_dihedral_angles_loss': 0.0,  # this significantly poison the training
            #     'sc_chi_angles_loss': 0.5
            # },
            # coord_loss_ratio=0.5,  # (1 - r)*seq + r * coord
            # coord_prior_var=1.0,  # sigma^2
            # anchor_at_ca=False,
            # share_decoder=False,
            n_rbf=0,
            cutoff=0,
            encoder='dyMEAN',
            mode='codesign'  # codesign, fixbb (inverse folding), fixseq (structure prediction)
    ) -> None:
        super().__init__()
        """
        self.mask_id = mask_id
        self.latent_id = latent_id
        self.ca_channel_idx = CA_channel_idx
        self.n_channel = n_channel
        self.mask_ratio = mask_ratio
        self.fix_alpha_carbon = fix_alpha_carbon
        self.h_kl_weight = h_kl_weight
        self.z_kl_weight = z_kl_weight
        # self.coord_loss_weights = coord_loss_weights
        # self.coord_loss_ratio = coord_loss_ratio
        self.mode = mode
        self.latent_size = 0 if self.mode == 'fixseq' else latent_size
        self.latent_n_channel = 0 if self.mode == 'fixbb' else latent_n_channel
        self.anchor_at_ca = anchor_at_ca
        self.coord_prior_var = coord_prior_var
        if self.fix_alpha_carbon: assert self.latent_n_channel == 1, f'Specifying fix alpha carbon (use Ca as the latent coordinate) but number of latent channels is not 1'
        if self.anchor_at_ca: assert self.latent_n_channel == 1, f'Specifying anchor_at_ca as True but number of latent channels is not 1'
        if self.mode == 'fixseq': assert self.coord_loss_ratio == 1.0, f'Specifying fixseq mode but coordination loss ratio is not 1.0: {self.coord_loss_ratio}'
        if self.mode == 'fixbb': assert self.coord_loss_ratio == 0.0, f'Specifying fixbb mode but coordination loss ratio is not 0.0: {self.coord_loss_ratio}'
        """
        self.embed_size = embed_size
        atom_embed_size = embed_size // 4
        self.aa_feature = SeparatedAminoAcidFeature(
            embed_size, atom_embed_size,
            max_position=max_position,
            relative_position=relative_position,
            fix_atom_weights=True
        )
        # self.protein_feature = ProteinFeature()
        self.encoder = create_encoder(
            name=encoder,
            atom_embed_size=atom_embed_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            n_channel=n_channel,
            n_layers=n_layers,
            dropout=dropout,
            n_rbf=n_rbf,
            cutoff=cutoff
        )

        self.merge_H0_t = nn.Sequential(
            nn.Linear(2 * self.embed_size, self.embed_size), nn.ReLU(),
            nn.Linear(self.embed_size, self.embed_size), nn.ReLU(),
            nn.Linear(self.embed_size, self.embed_size)  # 3D coordinates of 9 side chain atoms
        )
        """
        if self.mode != 'fixbb':
            self.sidechain_decoder = create_encoder(
                name=encoder,
                atom_embed_size=atom_embed_size,
                embed_size=embed_size,
                hidden_size=hidden_size,
                n_channel=n_channel,
                n_layers=n_layers,
                dropout=dropout,
                n_rbf=n_rbf,
                cutoff=cutoff
            )
            self.backbone_model = BackboneModel()
            self.sidechain_model = SideChainModel()
            self.W_Z_log_var = nn.Linear(hidden_size, latent_n_channel * 3)

        if self.mode != 'fixseq':
            self.W_mean = nn.Linear(hidden_size, latent_size)
            self.W_log_var = nn.Linear(hidden_size, latent_size)
            # self.hidden2latent = nn.Linear(hidden_size, latent_size)
            self.latent2hidden = nn.Linear(latent_size, hidden_size)
            self.merge_S_H = nn.Linear(hidden_size * 2, hidden_size)

            if share_decoder:
                self.seq_decoder = self.sidechain_decoder
            else:
                self.seq_decoder = create_encoder(
                    name=encoder,
                    atom_embed_size=atom_embed_size,
                    embed_size=embed_size,
                    hidden_size=hidden_size,
                    n_channel=n_channel,
                    n_layers=n_layers,
                    dropout=dropout,
                    n_rbf=n_rbf,
                    cutoff=cutoff
                )

        # residue type index mapping, from original index to 0~20, 0 is unk
        self.unk_idx = 0
        self.s_map = [0 for _ in range(len(VOCAB))]
        self.s_remap = [0 for _ in range(len(aas) + 1)]
        self.s_remap[0] = VOCAB.symbol_to_idx(VOCAB.UNK)
        for i, (a, _) in enumerate(aas):
            original_idx = VOCAB.symbol_to_idx(a)
            self.s_map[original_idx] = i + 1  # start from 1
            self.s_remap[i + 1] = original_idx
        self.s_map = nn.Parameter(torch.tensor(self.s_map, dtype=torch.long), requires_grad=False)
        self.s_remap = nn.Parameter(torch.tensor(self.s_remap, dtype=torch.long), requires_grad=False)

        if self.mode != 'fixseq':
            self.seq_linear = nn.Linear(hidden_size, len(self.s_remap))

    """

    @torch.no_grad()
    def prepare_inputs(self, X, S, mask, atom_mask, lengths):
        # batch ids
        batch_ids = self.get_batch_ids(S, lengths)

        # edges
        row, col = variadic_meshgrid(
            input1=torch.arange(batch_ids.shape[0], device=batch_ids.device),
            size1=lengths,
            input2=torch.arange(batch_ids.shape[0], device=batch_ids.device),
            size2=lengths,
        )  # (row, col)

        is_ctx = mask[row] == mask[col]
        is_inter = ~is_ctx
        ctx_edges = torch.stack([row[is_ctx], col[is_ctx]], dim=0)  # [2, Ec], intra-edges
        inter_edges = torch.stack([row[is_inter], col[is_inter]], dim=0)  # [2, Ei], inter-edges

        return ctx_edges, inter_edges, batch_ids

    @torch.no_grad()
    def get_batch_ids(self, S, lengths):
        batch_ids = torch.zeros_like(S)
        batch_ids[torch.cumsum(lengths, dim=0)[:-1]] = 1
        batch_ids.cumsum_(dim=0)
        return batch_ids

    """
    def rsample(self, H, Z, Z_centers, no_randomness=False):
        '''
            H: [N, latent_size]
            Z: [N, latent_channel, 3]
            Z_centers: [N, latent_channel, 3]
        '''

        if self.mode != 'fixseq':
            data_size = H.shape[0]
            H_mean = self.W_mean(H)
            H_log_var = -torch.abs(self.W_log_var(H))  # Following Mueller et al., z_log_var is log(\sigma^2)
            H_kl_loss = -0.5 * torch.sum(1.0 + H_log_var - H_mean * H_mean - torch.exp(H_log_var)) / data_size
            H_vecs = H_mean if no_randomness else H_mean + torch.exp(H_log_var / 2) * torch.randn_like(H_mean)
        else:
            H_vecs, H_kl_loss = None, 0

        if self.mode != 'fixbb':
            data_size = Z.shape[0]
            Z_mean_delta = Z - Z_centers
            Z_log_var = -torch.abs(self.W_Z_log_var(H)).view(-1, self.latent_n_channel, 3)
            Z_kl_loss = -0.5 * torch.sum(1.0 + Z_log_var - math.log(
                self.coord_prior_var) - Z_mean_delta * Z_mean_delta / self.coord_prior_var - torch.exp(
                Z_log_var) / self.coord_prior_var) / data_size
            Z_vecs = Z if no_randomness else Z + torch.exp(Z_log_var / 2) * torch.randn_like(Z)
        else:
            Z_vecs, Z_kl_loss = None, 0

        return H_vecs, Z_vecs, H_kl_loss, Z_kl_loss

    def _get_latent_channels(self, X, atom_mask):
        atom_weights = atom_mask.float()  # 1 for atom, 0 for padding/missing, [N, 14]
        if hasattr(self, 'fix_alpha_carbon') and self.fix_alpha_carbon:
            return X[:, self.ca_channel_idx].unsqueeze(1)  # use alpha carbon as latent channel
        elif self.latent_n_channel == 1:
            X = (X * atom_weights.unsqueeze(-1)).sum(1)  # [N, 3]
            X = X / atom_weights.sum(-1).unsqueeze(-1)  # [N, 3]
            return X.unsqueeze(1)
        elif self.latent_n_channel == 5:
            bb_X = X[:, :4]
            X = (X * atom_weights.unsqueeze(-1)).sum(1)  # [N, 3]
            X = X / atom_weights.sum(-1).unsqueeze(-1)  # [N, 3]
            X = torch.cat([bb_X, X.unsqueeze(1)], dim=1)  # [N, 5, 3]
            return X
        else:
            raise NotImplementedError(f'Latent number of channels: {self.latent_n_channel} not implemented')

    def _get_latent_channel_anchors(self, X, atom_mask):
        if self.anchor_at_ca:
            return X[:, self.ca_channel_idx].unsqueeze(1)
        else:
            return self._get_latent_channels(X, atom_mask)

    def _fill_latent_channels(self, latent_X):
        if self.latent_n_channel == 1:
            return latent_X.repeat(1, self.n_channel, 1)
        elif self.latent_n_channel == 5:
            bb_X = latent_X[:, :4]
            sc_X = latent_X[:, 4].unsqueeze(1).repeat(1, self.n_channel - 4, 1)
            return torch.cat([bb_X, sc_X], dim=1)
        else:
            raise NotImplementedError(f'Latent number of channels: {self.latent_n_channel} not implemented')

    def _remove_sidechain_atom_mask(self, atom_mask, mask_generate):
        atom_mask = atom_mask.clone()
        bb_mask = atom_mask[mask_generate]
        bb_mask[:, 4:] = 0  # only backbone atoms are visible
        atom_mask[mask_generate] = bb_mask
        return atom_mask

    @torch.no_grad()
    def _mask_pep(self, S, atom_mask, mask_generate):
        assert self.mask_ratio > 0
        S, atom_mask = S.clone(), atom_mask.clone()
        pep_S = S[mask_generate]
        do_mask = torch.rand_like(pep_S, dtype=torch.float) < self.mask_ratio
        pep_S[do_mask] = self.mask_id

        S[mask_generate] = pep_S
        atom_mask[mask_generate] = self._remove_sidechain_atom_mask(atom_mask[mask_generate], do_mask)

        return S, atom_mask

    """

    def encode(self, X, S, mask, position_ids, lengths,
               atom_mask, t_emb, node_init=None):
        """
        X: [n_all_res, 14, 3]  # n_all_res is the number of all residues in a batch
        S: [n_all_res]
        mask: [n_all_res]  # for residues to be generated?
        position_ids: [n_all_res]  # absolute index in PDB file
        lengths: [bsz]  # length of corresponding complex
        atom_mask: [n_all_res, 14]  # atom mask for real atoms
        """
        true_X = X.clone()

        ctx_edges, inter_edges, batch_ids = self.prepare_inputs(X, S, mask, atom_mask, lengths)
        H_0, (atom_embeddings, _) = self.aa_feature(S, position_ids)
        if node_init is not None:
            H_0 = node_init

        H_0 = self.merge_H0_t(torch.cat([H_0, t_emb], dim=-1))

        edges = torch.cat([ctx_edges, inter_edges], dim=1)
        atom_weights = atom_mask.float()  # 1 for atom, 0 for padding/missing, [N, 14]

        # H, pred_X = self.encoder(H_0, X, edges, channel_attr=atom_embeddings, channel_weights=atom_weights)
        H = self.encoder(H_0, X, edges, channel_attr=atom_embeddings, channel_weights=atom_weights)
        # H: [n_all_res, emb_dim], pred_X: [n_all_res, 14, 3]
        # H = H[mask]
        # return H, pred_X
        return H

    def forward(self, rot, trans, angle, sctrans, seq, res_mask,
                gen_mask, atom_mask, abs_idx, t_emb, node_init=None):
        num_batch, num_res, max_atom = atom_mask.shape

        if angle.shape[-1] != 5:
            angle = F.pad(angle, (0, 4))
        pos_heavyatom, _, _ = full_atom_reconstruction(R_bb=rot, t_bb=trans, angles=angle,
                                                       aa=seq * res_mask)
        # (B,L,14,3), instead of 15, ignore OXT masked
        pos_heavyatom[:, :, 5:] = torch.einsum('blij,blkj->blik', sctrans,
                                               rot) + trans[:, :, None]  # abs coor.

        real_res_indices = res_mask.nonzero(as_tuple=True)
        _pos_heavyatom = pos_heavyatom[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res, 14, 3]
        _seqs_t = seq[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res]
        _gen_mask = gen_mask[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res]
        _t_emb = t_emb[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res]
        _position_ids = abs_idx[real_res_indices[0], real_res_indices[1]]
        _lengths = torch.sum(res_mask, dim=-1)
        # EXCLUDING the virtual atoms in receptor
        _atom_mask = torch.where(gen_mask[..., None].expand_as(atom_mask) == 1,
                                 torch.ones_like(atom_mask), atom_mask)[real_res_indices[0], real_res_indices[1]]
        # INCLUDING the virtual atoms in receptor
        # _atom_mask = ~torch.logical_xor(atom_mask, get_heavyatom_mask(seq)[..., :-1])[
        #     real_res_indices[0], real_res_indices[1]]
        # _atom_mask = torch.where(_gen_mask[:, None].expand_as(_atom_mask) == 1,
        #                          torch.ones_like(_atom_mask), _atom_mask)
        if node_init is not None:
            _node_embed = node_init[real_res_indices[0], real_res_indices[1]]
        else:
            _node_embed = None

        H = self.encode(X=_pos_heavyatom, S=_seqs_t, mask=_gen_mask,
                        position_ids=_position_ids, lengths=_lengths,
                        atom_mask=_atom_mask, t_emb=_t_emb, node_init=_node_embed)
        # Reshape to the batch-wise
        node_emb = torch.zeros([num_batch, num_res, self.embed_size], device=H.device)
        # coor_pred = torch.zeros_like(pos_heavyatom, device=H.device)

        node_emb[real_res_indices[0], real_res_indices[1]] = H

        return node_emb, pos_heavyatom


class Encoder_rec(nn.Module):
    def __init__(
            self,
            embed_size=128,
            hidden_size=128,
            n_channel=14,
            max_position=2048,
            relative_position=True,
            n_layers=3,
            dropout=0.1,
            n_rbf=0,
            cutoff=0,
            encoder='dyMEAN',
            mode='codesign'  # codesign, fixbb (inverse folding), fixseq (structure prediction)
    ) -> None:
        super().__init__()

        self.embed_size = embed_size
        atom_embed_size = embed_size // 4
        self.aa_feature = SeparatedAminoAcidFeature(
            embed_size, atom_embed_size,
            max_position=max_position,
            relative_position=relative_position,
            fix_atom_weights=True
        )
        self.encoder = create_encoder(
            name=encoder,
            atom_embed_size=atom_embed_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            n_channel=n_channel,
            n_layers=n_layers,
            dropout=dropout,
            n_rbf=n_rbf,
            cutoff=cutoff
        )

    @torch.no_grad()
    def prepare_inputs(self, X, S, mask, atom_mask, lengths):
        # batch ids
        batch_ids = self.get_batch_ids(S, lengths)

        # edges
        row, col = variadic_meshgrid(
            input1=torch.arange(batch_ids.shape[0], device=batch_ids.device),
            size1=lengths,
            input2=torch.arange(batch_ids.shape[0], device=batch_ids.device),
            size2=lengths,
        )  # (row, col)

        is_ctx = mask[row] == mask[col]
        is_inter = ~is_ctx
        ctx_edges = torch.stack([row[is_ctx], col[is_ctx]], dim=0)  # [2, Ec], intra-edges
        inter_edges = torch.stack([row[is_inter], col[is_inter]], dim=0)  # [2, Ei], inter-edges

        return ctx_edges, inter_edges, batch_ids

    @torch.no_grad()
    def get_batch_ids(self, S, lengths):
        batch_ids = torch.zeros_like(S)
        batch_ids[torch.cumsum(lengths, dim=0)[:-1]] = 1
        batch_ids.cumsum_(dim=0)
        return batch_ids

    def encode(self, X, S, mask, position_ids, lengths,
               atom_mask, node_init=None):
        """
        X: [n_all_res, 14, 3]  # n_all_res is the number of all residues in a batch
        S: [n_all_res]
        mask: [n_all_res]  # for residues to be generated?
        position_ids: [n_all_res]  # absolute index in PDB file
        lengths: [bsz]  # length of corresponding complex
        atom_mask: [n_all_res, 14]  # atom mask for real atoms
        """
        true_X = X.clone()

        ctx_edges, inter_edges, batch_ids = self.prepare_inputs(X, S, mask, atom_mask, lengths)
        H_0, (atom_embeddings, _) = self.aa_feature(S, position_ids)
        if node_init is not None:
            H_0 = node_init

        edges = torch.cat([ctx_edges, inter_edges], dim=1)
        atom_weights = atom_mask.float()  # 1 for atom, 0 for padding/missing, [N, 14]
        H = self.encoder(H_0, X, edges, channel_attr=atom_embeddings, channel_weights=atom_weights)
        return H

    def forward(self, pos_atom, seq, res_mask, gen_mask, 
                atom_mask, abs_idx, node_init=None):
        num_batch, num_res, max_atom = atom_mask.shape
        """
        if angle.shape[-1] != 5:
            angle = F.pad(angle, (0, 4))
        pos_heavyatom, _, _ = full_atom_reconstruction(R_bb=rot, t_bb=trans, angles=angle,
                                                       aa=seq * res_mask)
        # (B,L,14,3), instead of 15, ignore OXT masked
        pos_heavyatom[:, :, 5:] = torch.einsum('blij,blkj->blik', sctrans,
                                               rot) + trans[:, :, None]  # abs coor.
        """
        rec_mask = res_mask * ~gen_mask
        real_res_indices = rec_mask.nonzero(as_tuple=True)
        _pos_heavyatom = pos_atom[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res, 14, 3]
        _seqs_t = seq[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res]
        _gen_mask = gen_mask[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res]
        _position_ids = abs_idx[real_res_indices[0], real_res_indices[1]]
        _lengths = torch.sum(rec_mask, dim=-1)
        # EXCLUDING the virtual atoms in receptor
        _atom_mask = (atom_mask * rec_mask[..., None])[real_res_indices[0], real_res_indices[1]]
        # _atom_mask = torch.where(gen_mask[..., None].expand_as(atom_mask) == 1,
        #                          torch.ones_like(atom_mask), atom_mask)[real_res_indices[0], real_res_indices[1]]
        # INCLUDING the virtual atoms in receptor
        # _atom_mask = ~torch.logical_xor(atom_mask, get_heavyatom_mask(seq)[..., :-1])[
        #     real_res_indices[0], real_res_indices[1]]
        # _atom_mask = torch.where(_gen_mask[:, None].expand_as(_atom_mask) == 1,
        #                          torch.ones_like(_atom_mask), _atom_mask)
        if node_init is not None:
            _node_embed = node_init[real_res_indices[0], real_res_indices[1]]
        else:
            _node_embed = None

        H = self.encode(X=_pos_heavyatom, S=_seqs_t, mask=_gen_mask,
                        position_ids=_position_ids, lengths=_lengths,
                        atom_mask=_atom_mask, node_init=_node_embed)
        # Reshape to the batch-wise
        node_emb = torch.zeros([num_batch, num_res, self.embed_size], device=H.device)
        node_emb[real_res_indices[0], real_res_indices[1]] = H

        return node_emb


class Encoder_comp(nn.Module):
    def __init__(
            self,
            embed_size=128,
            hidden_size=128,
            n_channel=14,
            max_position=2048,
            relative_position=True,
            n_layers=3,
            dropout=0.1,
            n_rbf=0,
            cutoff=0,
            encoder='dyMEAN',
            mode='codesign'  # codesign, fixbb (inverse folding), fixseq (structure prediction)
    ) -> None:
        super().__init__()

        self.embed_size = embed_size
        atom_embed_size = embed_size // 4
        self.aa_feature = SeparatedAminoAcidFeature(
            embed_size, atom_embed_size,
            max_position=max_position,
            relative_position=relative_position,
            fix_atom_weights=True
        )
        self.encoder = create_encoder(
            name=encoder,
            atom_embed_size=atom_embed_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            n_channel=n_channel,
            n_layers=n_layers,
            dropout=dropout,
            n_rbf=n_rbf,
            cutoff=cutoff
        )

        self.merge_H0_t = nn.Sequential(
            nn.Linear(2 * self.embed_size, self.embed_size), nn.ReLU(),
            nn.Linear(self.embed_size, self.embed_size), nn.ReLU(),
            nn.Linear(self.embed_size, self.embed_size))

    @torch.no_grad()
    def prepare_inputs(self, X, S, mask, atom_mask, lengths):
        # batch ids
        batch_ids = self.get_batch_ids(S, lengths)

        # edges
        row, col = variadic_meshgrid(
            input1=torch.arange(batch_ids.shape[0], device=batch_ids.device),
            size1=lengths,
            input2=torch.arange(batch_ids.shape[0], device=batch_ids.device),
            size2=lengths,
        )  # (row, col)

        is_ctx = mask[row] == mask[col]
        is_inter = ~is_ctx
        ctx_edges = torch.stack([row[is_ctx], col[is_ctx]], dim=0)  # [2, Ec], intra-edges
        inter_edges = torch.stack([row[is_inter], col[is_inter]], dim=0)  # [2, Ei], inter-edges

        return ctx_edges, inter_edges, batch_ids

    @torch.no_grad()
    def get_batch_ids(self, S, lengths):
        batch_ids = torch.zeros_like(S)
        batch_ids[torch.cumsum(lengths, dim=0)[:-1]] = 1
        batch_ids.cumsum_(dim=0)
        return batch_ids

    def encode(self, X, S, t_emb, mask, position_ids, lengths,
               atom_mask, node_init=None):
        """
        X: [n_all_res, 14, 3]  # n_all_res is the number of all residues in a batch
        S: [n_all_res]
        mask: [n_all_res]  # for residues to be generated?
        position_ids: [n_all_res]  # absolute index in PDB file
        lengths: [bsz]  # length of corresponding complex
        atom_mask: [n_all_res, 14]  # atom mask for real atoms
        """
        true_X = X.clone()

        ctx_edges, inter_edges, batch_ids = self.prepare_inputs(X, S, mask, atom_mask, lengths)
        if node_init is not None:
            H_0 = node_init
        else:
            H_0, (atom_embeddings, _) = self.aa_feature(S, position_ids)
        H_0 = self.merge_H0_t(torch.cat([H_0, t_emb], dim=-1))

        edges = torch.cat([ctx_edges, inter_edges], dim=1)
        atom_weights = atom_mask.float()  # 1 for atom, 0 for padding/missing, [N, 14]
        H = self.encoder(H_0, X, edges, channel_attr=atom_embeddings, channel_weights=atom_weights)
        return H

    def forward(self, pos_atom, seq, trans, rot, t_emb, chain_nb, res_nb, 
                res_mask, gen_mask, atom_mask, abs_idx, node_init=None):
        num_batch, num_res, max_atom = atom_mask.shape
        """
        if angle.shape[-1] != 5:
            angle = F.pad(angle, (0, 4))
        pos_heavyatom, _, _ = full_atom_reconstruction(R_bb=rot, t_bb=trans, angles=angle,
                                                       aa=seq * res_mask)
        # (B,L,14,3), instead of 15, ignore OXT masked
        pos_heavyatom[:, :, 5:] = torch.einsum('blij,blkj->blik', sctrans,
                                               rot) + trans[:, :, None]  # abs coor.
        """
        real_res_indices = res_mask.nonzero(as_tuple=True)

        # Reorganize the pos_atom and mask
        bb_pos = reconstruct_backbone(R=rot, t=trans, aa=seq, 
                                      chain_nb=chain_nb, res_nb=res_nb, mask=res_mask)  # (B, L, 4, 3)
        bb_pos = F.pad(bb_pos, (0, 0, 0, max_atom-bb_pos.shape[-2]), "constant", 0)  # (B, L, max_atom, 3)
        pos_atom = torch.where(gen_mask[..., None, None], bb_pos, pos_atom)

        template_mask = torch.zeros_like(atom_mask).to(seq.device)  # (B, L, max_atom)
        template_mask[:, :, :4] = 1
        atom_mask = torch.where(gen_mask[..., None], template_mask, atom_mask)  # Mask the side chains of ligands
        # Reorganize the pos_atom and mask

        _pos_heavyatom = pos_atom[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res, 14, 3]
        _seqs_t = seq[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res]
        _gen_mask = gen_mask[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res]
        _position_ids = abs_idx[real_res_indices[0], real_res_indices[1]]
        _t_emb = t_emb[real_res_indices[0], real_res_indices[1]]
        _lengths = torch.sum(res_mask, dim=-1)
        # EXCLUDING the virtual atoms in receptor
        _atom_mask = (atom_mask * res_mask[..., None])[real_res_indices[0], real_res_indices[1]]
        # _atom_mask = torch.where(gen_mask[..., None].expand_as(atom_mask) == 1,
        #                          torch.ones_like(atom_mask), atom_mask)[real_res_indices[0], real_res_indices[1]]
        # INCLUDING the virtual atoms in receptor
        # _atom_mask = ~torch.logical_xor(atom_mask, get_heavyatom_mask(seq)[..., :-1])[
        #     real_res_indices[0], real_res_indices[1]]
        # _atom_mask = torch.where(_gen_mask[:, None].expand_as(_atom_mask) == 1,
        #                          torch.ones_like(_atom_mask), _atom_mask)
        if node_init is not None:
            _node_embed = node_init[real_res_indices[0], real_res_indices[1]]
        else:
            _node_embed = None

        H = self.encode(X=_pos_heavyatom, S=_seqs_t, t_emb=_t_emb, mask=_gen_mask,
                        position_ids=_position_ids, lengths=_lengths,
                        atom_mask=_atom_mask, node_init=_node_embed)
        # Reshape to the batch-wise
        node_emb = torch.zeros([num_batch, num_res, self.embed_size], device=H.device)
        node_emb[real_res_indices[0], real_res_indices[1]] = H

        return node_emb

"""
        if self.mode != 'fixbb':
            if hasattr(self, 'fix_alpha_carbon') and self.fix_alpha_carbon:
                Z = self._get_latent_channels(true_X, atom_mask)
            else:
                Z = self._get_latent_channels(pred_X, atom_mask)
            Z_centers = self._get_latent_channel_anchors(true_X, atom_mask)
            Z, Z_centers = Z[mask], Z_centers[mask]
        else:
            Z, Z_centers = None, None

        # resample
        latent_H, latent_X, H_kl_loss, X_kl_loss = self.rsample(H, Z, Z_centers, no_randomness)
        return latent_H, latent_X, H_kl_loss, X_kl_loss

    def decode(self, X, S, H, Z, mask, position_ids, lengths, atom_mask, teacher_forcing):
        X, S, atom_mask = X.clone(), S.clone(), atom_mask.clone()
        true_S = S[mask].clone()
        if self.mode != 'fixbb':  # fill coordinates with latent points
            X[mask] = self._fill_latent_channels(Z)
        if self.mode != 'fixseq':  # fill sequences with mask token
            S[mask] = self.latent_id
            H_from_latent = self.latent2hidden(H)

        if self.mode == 'fixbb':  # only backbone atoms are visible
            atom_mask = self._remove_sidechain_atom_mask(atom_mask, mask)
        elif self.mode == 'codesign':  # all channels are visible when deciding the sequence (all dummy atoms)
            atom_mask[mask] = 1
        else:  # fixseq mode does not need to change atom mask
            pass

        ctx_edges, inter_edges, batch_ids = self.prepare_inputs(X, S, mask, atom_mask, lengths)
        edges = torch.cat([ctx_edges, inter_edges], dim=1)

        # decode sequence
        if self.mode != 'fixseq':
            H_0, (atom_embeddings, _) = self.aa_feature(S, position_ids)
            H_0 = H_0.clone()
            H_0[mask] = H_from_latent  # TODO: how about the position encoding
            H, _ = self.seq_decoder(H_0, X, edges, channel_attr=atom_embeddings, channel_weights=atom_mask.float())
            pred_S_logits = self.seq_linear(H[mask])  # [Ntgt, 21]
            S = S.clone()
            if teacher_forcing:  # teacher forcing
                S[mask] = true_S
            else:  # inference
                S[mask] = self.s_remap[torch.argmax(pred_S_logits, dim=-1)]
        else:
            pred_S_logits = None

        # decode sidechain
        if self.mode != 'fixbb':
            H_0, (atom_embeddings, atom_weights) = self.aa_feature(S, position_ids)
            H_0 = H_0.clone()
            if self.mode != 'fixseq':
                H_0[mask] = self.merge_S_H(torch.cat([H_from_latent, H_0[mask]], dim=-1))
                # H_0[mask] = H_from_latent
            atom_mask = atom_mask.clone()
            atom_mask[mask] = atom_weights.bool()[mask] & atom_mask[
                mask]  # reset atomic visibility of the reconstruction part with the decoded sequence
            _, pred_X = self.sidechain_decoder(H_0, X, edges, channel_attr=atom_embeddings,
                                               channel_weights=atom_mask.float())
            pred_X = pred_X[mask]
        else:
            pred_X = None

        return pred_S_logits, pred_X

    @oom_decorator
    def forward(self, X, S, mask, position_ids, lengths, atom_mask, teacher_forcing=True):
        true_X, true_S = X[mask].clone(), S[mask].clone()

        # encode: H (N*d), Z (N*3)
        if self.mask_ratio > 0:
            input_S, input_atom_mask = self._mask_pep(S, atom_mask, mask)
        else:
            input_S, input_atom_mask = S, atom_mask
        H, Z, H_kl_loss, Z_kl_loss = self.encode(X, input_S, mask, position_ids, lengths, input_atom_mask)

        if self.mode != 'fixbb':
            coord_reg_loss = F.mse_loss(Z, self._get_latent_channel_anchors(true_X, atom_mask[mask]))
        else:
            coord_reg_loss = 0

        # decode: S (N), Z (N * 14 * 3) with atom mask
        recon_S_logits, recon_X = self.decode(X, S, H, Z, mask, position_ids, lengths, atom_mask, teacher_forcing)

        # sequence reconstruction loss
        if self.mode != 'fixseq':
            seq_recon_loss = F.cross_entropy(recon_S_logits, self.s_map[true_S])
            # aar
            with torch.no_grad():
                aar = (torch.argmax(recon_S_logits, dim=-1) == self.s_map[true_S]).sum() / len(recon_S_logits)
        else:
            seq_recon_loss, aar = 0, 1.0

        # coordinates reconstruction loss
        if self.mode != 'fixbb':
            xloss_mask = atom_mask[mask]
            batch_ids = self.get_batch_ids(S, lengths)[mask]
            segment_ids = torch.ones_like(true_S, device=true_S.device, dtype=torch.long)
            if self.n_channel == 4:  # backbone only
                loss_profile = {}
            else:
                true_struct_profile = self.protein_feature.get_struct_profile(true_X, true_S, batch_ids,
                                                                              self.aa_feature, segment_ids, xloss_mask)
                recon_struct_profile = self.protein_feature.get_struct_profile(recon_X, true_S, batch_ids,
                                                                               self.aa_feature, segment_ids, xloss_mask)
                loss_profile = {key + '_loss': F.l1_loss(recon_struct_profile[key], true_struct_profile[key]) for key in
                                recon_struct_profile}

            # mse
            xloss = F.mse_loss(recon_X[xloss_mask], true_X[xloss_mask])
            loss_profile['Xloss'] = xloss

            # CA mse
            ca_xloss_mask = xloss_mask[:, self.ca_channel_idx]
            ca_xloss = F.mse_loss(recon_X[:, self.ca_channel_idx][ca_xloss_mask],
                                  true_X[:, self.ca_channel_idx][ca_xloss_mask])
            loss_profile['ca_Xloss'] = ca_xloss

            struct_recon_loss = 0
            for name in loss_profile:
                struct_recon_loss = struct_recon_loss + self.coord_loss_weights[name] * loss_profile[name]
        else:
            struct_recon_loss, loss_profile = 0, {}

        recon_loss = (1 - self.coord_loss_ratio) * (seq_recon_loss + self.h_kl_weight * H_kl_loss) + \
                     self.coord_loss_ratio * (struct_recon_loss + self.z_kl_weight * Z_kl_loss)

        return recon_loss, (seq_recon_loss, aar), (struct_recon_loss, loss_profile), (
        H_kl_loss, Z_kl_loss, coord_reg_loss)

    def _reconstruct(self, X, S, mask, position_ids, lengths, atom_mask, given_laten_H=None, given_latent_X=None,
                     allow_unk=False, optimize_sidechain=True, idealize=False, no_randomness=False):
        if given_laten_H is None and given_latent_X is None:
            # encode: H (N*d), Z (N*3)
            H, Z, _, _ = self.encode(X, S, mask, position_ids, lengths, atom_mask, no_randomness=no_randomness)

        else:
            H, Z = given_laten_H, given_latent_X

        # decode: S (N), Z (N * 14 * 3) with atom mask
        recon_S_logits, recon_X = self.decode(X, S, H, Z, mask, position_ids, lengths, atom_mask, teacher_forcing=False)
        batch_ids = self.get_batch_ids(S, lengths)[mask]

        if self.mode != 'fixseq':
            if not allow_unk:
                recon_S_logits[:, 0] = float('-inf')

            # map aa index back
            recon_S = self.s_remap[torch.argmax(recon_S_logits, dim=-1)]
            # ppls
            snll_all = F.cross_entropy(recon_S_logits, torch.argmax(recon_S_logits, dim=-1), reduction='none')
            batch_ppls = scatter_mean(snll_all, batch_ids, dim=0)
        else:
            recon_S = S[mask]
            batch_ppls = torch.zeros(batch_ids.max() + 1, device=recon_X.device).float()

        if self.mode == 'fixseq' or (self.mode != 'fixbb' and idealize):
            # rectify backbone
            recon_X = self.backbone_model(recon_X, batch_ids)
            # rectify sidechain
            recon_X = self.sidechain_model(recon_X, recon_S, batch_ids, optimize_sidechain)

        return recon_X, recon_S, batch_ppls, batch_ids

    @torch.no_grad()
    def test(self, X, S, mask, position_ids, lengths, atom_mask, given_laten_H=None, given_latent_X=None,
             return_tensor=False, allow_unk=False, optimize_sidechain=True, idealize=False, n_iter=1):

        no_randomness = given_laten_H is not None  # in reconstruction mode, with latent variable derived from diffusion model
        for i in range(n_iter):
            recon_X, recon_S, batch_ppls, batch_ids = self._reconstruct(X, S, mask, position_ids, lengths, atom_mask,
                                                                        given_laten_H, given_latent_X, allow_unk,
                                                                        optimize_sidechain, idealize, no_randomness)
            X, S = X.clone(), S.clone()
            if self.mode != 'fixbb':
                X[mask] = recon_X
            if self.mode != 'fixseq':
                S[mask] = recon_S
            given_laten_H, given_latent_X = None, None  # let the model encode and decode for later iterations

        if return_tensor:
            return recon_X, recon_S, batch_ppls

        batch_X, batch_S = [], []
        batch_ppls = batch_ppls.tolist()
        for i, l in enumerate(lengths):
            cur_mask = batch_ids == i
            if self.mode == 'fixbb':
                batch_X.append(None)
            else:
                batch_X.append(recon_X[cur_mask].tolist())
            if self.mode == 'fixseq':
                batch_S.append(None)
            else:
                batch_S.append(''.join([VOCAB.idx_to_symbol(s) for s in recon_S[cur_mask]]))

        return batch_X, batch_S, batch_ppls
"""
