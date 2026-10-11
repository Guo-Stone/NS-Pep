import copy
import pdb

import torch
from torch import nn
import torch.nn.functional as F
from models_con import ipa_pytorch as ipa_pytorch
from models_con.torsion import full_atom_reconstruction, get_heavyatom_mask
from models_con.enc.module import Encoder
from models_con.hern.encoder import HierEGNNEncoder
from data import utils as du

from models_con.utils import get_index_embedding, get_time_embedding

import pepflow.modules.protein.constants as constants
from pepflow.modules.common.layers import AngularEncoding, sample_from, clampped_one_hot

import math


class GAEncoder(nn.Module):
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf

        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12)  # 25*5=120, for competitive embedding size
        self.angle_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 5)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(22, self._ipa_conf.c_s)
        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 20)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # mixer
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=5),
                      self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t, seqs_t, node_embed, edge_embed, generate_mask, res_mask):
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        node_embed = self.res_feat_mixer(torch.cat(
            [node_embed, self.current_seq_embedder(seqs_t), self.embed_t(t, node_mask),
             self.angles_embedder(angles_t).reshape(num_batch, num_res, -1)], dim=-1))
        node_embed = node_embed * node_mask[..., None]
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_seqs1_prob = self.seq_net(node_embed)
        pred_angles1 = self.angle_net(node_embed)
        pred_angles1 = pred_angles1 % (2 * math.pi)  # inductive bias to bound between (0,2pi)

        return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob


class GAEncoder_NS18(nn.Module):
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf

        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12)  # 25*5=120, for competitive embedding size
        num_torsion_angle = len(constants.chi_angles_mask[constants.AA.ALA]) + 1  # Plus torsion of O
        self.angle_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, num_torsion_angle)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )
        
        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(constants.PAD_RESIDUE_INDEX + 1,
                                                 self._ipa_conf.c_s)
        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, constants.PAD_RESIDUE_INDEX - 1)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # mixer
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=num_torsion_angle),
                      self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t, seqs_t, node_embed, edge_embed, generate_mask, res_mask):
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]

        node_embed = self.res_feat_mixer(torch.cat(
            [node_embed, self.current_seq_embedder(seqs_t), self.embed_t(t, node_mask),
             self.angles_embedder(angles_t).reshape(num_batch, num_res, -1)], dim=-1))
        node_embed = node_embed * node_mask[..., None]
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_seqs1_prob = self.seq_net(node_embed)
        pred_angles1 = self.angle_net(node_embed)
        pred_angles1 = pred_angles1 % (2 * math.pi)  # inductive bias to bound between (0,2pi)

        return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob



class GAEncoder_NS3(nn.Module):
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf

        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12)  # 25*5=120, for competitive embedding size
        self.angle_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 5)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(constants.PAD_RESIDUE_INDEX + 1,
                                                 self._ipa_conf.c_s)
        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, constants.PAD_RESIDUE_INDEX - 1)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # mixer
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=5),
                      self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t, seqs_t, node_embed, edge_embed, generate_mask, res_mask):
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]

        node_embed = self.res_feat_mixer(torch.cat(
            [node_embed, self.current_seq_embedder(seqs_t), self.embed_t(t, node_mask),
             self.angles_embedder(angles_t).reshape(num_batch, num_res, -1)], dim=-1))
        node_embed = node_embed * node_mask[..., None]
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_seqs1_prob = self.seq_net(node_embed)
        pred_angles1 = self.angle_net(node_embed)
        pred_angles1 = pred_angles1 % (2 * math.pi)  # inductive bias to bound between (0,2pi)

        return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob


class GAEncoder_NS18_PredChi_delta(nn.Module):
    # Keep the model parameters similar to that of PepFlow
    # Predict Chi and delta of scaa
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf

        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12)
        num_angle_O = 1  # Only for the O atom
        self.angle_O_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, num_angle_O)
        )

        num_angle_chi = len(constants.chi_angles_mask[constants.AA.ALA])  # Exclude the O atom
        self.angle_chi_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, num_angle_chi)
        )        

        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(constants.PAD_RESIDUE_INDEX + 1,
                                                 self._ipa_conf.c_s)
        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, constants.PAD_RESIDUE_INDEX - 1)
        )

        self.sc_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, (constants.max_num_heavyatoms - 6) * 3)  # 3D coordinates of 11 side chain atoms
        )

        # mixer
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=num_angle_O),
                      self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )
        self.chi_trunk = 2

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t, 
                seqs_t, seqs_1, node_embed, edge_embed, 
                generate_mask, res_mask, Train=True):
        # Make sure that angles_t only contains the O atom angle
        angles_O_t = angles_t[:, :, 0:1]  # Only keep the first angle (O atom angle)
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]

        node_embed = self.res_feat_mixer(torch.cat(
            [node_embed, self.current_seq_embedder(seqs_t), self.embed_t(t, node_mask),
             self.angles_embedder(angles_O_t).reshape(num_batch, num_res, -1)], dim=-1))
        node_embed = node_embed * node_mask[..., None]
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_seqs1_prob = self.seq_net(node_embed)
        pred_angles_O_1 = self.angle_O_net(node_embed)
        pred_angles_O_1 = pred_angles_O_1 % (2 * math.pi)  # inductive bias to bound between (0,2pi)
        if Train:
            pred_seqs_1 = seqs_1
        else:
            pred_seqs_1 = sample_from(F.softmax(pred_seqs1_prob, dim=-1))


        # Predict the chi angles via teacher forcing
        node_embed_chi = self.res_feat_mixer(torch.cat(
            [node_embed, self.current_seq_embedder(pred_seqs_1), self.embed_t(t, node_mask),
             self.angles_embedder(angles_O_t).reshape(num_batch, num_res, -1)], dim=-1))
        node_embed_chi = node_embed_chi * node_mask[..., None]
        for b in range(self.chi_trunk):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed_chi,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed_chi = self.trunk[f'ipa_ln_{b}'](node_embed_chi + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed_chi, src_key_padding_mask=(1 - node_mask).bool())
            node_embed_chi = node_embed_chi + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed_chi = self.trunk[f'node_transition_{b}'](node_embed_chi)
            node_embed_chi = node_embed_chi * node_mask[..., None]

            if b < (self.chi_trunk - 1):
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed_chi, edge_embed)
                edge_embed *= edge_mask[..., None]

        pred_angles_chi_1 = self.angle_chi_net(node_embed_chi)
        pred_angles_chi_1 = pred_angles_chi_1 % (2 * math.pi)  # inductive bias to bound between (0,2pi)
        pred_sctrans_ep = self.sc_net(node_embed_chi)
        pred_sctrans_ep = ipa_pytorch.sctrans_organize(pred_sctrans_ep)

        pred_angles_1 = torch.cat([pred_angles_O_1, pred_angles_chi_1], dim=-1)  # Concatenate O and chi angles

        return pred_rotmats1, pred_trans1, pred_angles_1, pred_seqs1_prob, pred_seqs_1, pred_sctrans_ep


class GAEncoder_NS3_distEdge(nn.Module):
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf

        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12)  # 25*5=120, for competitive embedding size
        self.angle_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 5)
        )

        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(constants.PAD_RESIDUE_INDEX + 1, self._ipa_conf.c_s)
        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, constants.PAD_RESIDUE_INDEX - 1)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # mixer
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=5),
                      self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

        # Inject current distance matrix into Edge_emb
        self.distEdge = nn.Sequential(
            nn.Linear(self._ipa_conf.c_z + 1, self._ipa_conf.c_z), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_z, self._ipa_conf.c_z), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_z, self._ipa_conf.c_z)
        )

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t, seqs_t, node_embed, edge_embed, generate_mask, res_mask):
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]

        node_embed = self.res_feat_mixer(torch.cat(
            [node_embed, self.current_seq_embedder(seqs_t), self.embed_t(t, node_mask),
             self.angles_embedder(angles_t).reshape(num_batch, num_res, -1)], dim=-1))
        node_embed = node_embed * node_mask[..., None]
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                _, ca_dist = ipa_pytorch.ca_attn(curr_rigids, mask=edge_mask, offset_diag=True)
                bin_ca_dist = ipa_pytorch.bin_dist_mat(ca_dist, edge_mask)
                edge_embed = self.distEdge(torch.cat([edge_embed, bin_ca_dist[..., None]], dim=-1))
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_seqs1_prob = self.seq_net(node_embed)
        pred_angles1 = self.angle_net(node_embed)
        pred_angles1 = pred_angles1 % (2 * math.pi)  # inductive bias to bound between (0,2pi)

        return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob


class GAEncoder_NS18_predSC9(nn.Module):
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf

        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12)
        num_torsion_angle = len(constants.chi_angles_mask[constants.AA.ALA]) + 1
        self.angle_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, num_torsion_angle)
        )

        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(constants.PAD_RESIDUE_INDEX + 1,
                                                 self._ipa_conf.c_s)
        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, constants.PAD_RESIDUE_INDEX - 1)
        )

        # mixer
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=num_torsion_angle),
                      self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

        self.sc_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, (constants.max_num_heavyatoms - 6) * 3)  # 3D coordinates of 11 side chain atoms
        )

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t, seqs_t, node_embed, edge_embed, generate_mask, res_mask):
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]

        node_embed = self.res_feat_mixer(torch.cat(
            [node_embed, self.current_seq_embedder(seqs_t), self.embed_t(t, node_mask),
             self.angles_embedder(angles_t).reshape(num_batch, num_res, -1)], dim=-1))
        node_embed = node_embed * node_mask[..., None]
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_seqs1_prob = self.seq_net(node_embed)
        pred_angles1 = self.angle_net(node_embed)
        pred_angles1 = pred_angles1 % (2 * math.pi)  # inductive bias to bound between (0,2pi)

        pred_sctrans_ep = self.sc_net(node_embed)
        pred_sctrans_1 = ipa_pytorch.sctrans_organize(pred_sctrans_ep)

        return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob, pred_sctrans_1


class GAEncoder_NS18_PredChi(nn.Module):
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf

        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12)
        num_angle_O = 1  # Only for the O atom
        self.angle_O_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, num_angle_O)
        )

        num_angle_chi = len(constants.chi_angles_mask[constants.AA.ALA])  # Exclude the O atom
        self.angle_chi_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, num_angle_chi)
        )        

        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(constants.PAD_RESIDUE_INDEX + 1,
                                                 self._ipa_conf.c_s)
        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, constants.PAD_RESIDUE_INDEX - 1)
        )

        # mixer
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=num_angle_O),
                      self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

        # Attention trunk for Chi prediction
        self.num_blocks_chi = self._ipa_conf.num_blocks - 4
        self.trunk_chi = nn.ModuleDict()
        for b in range(self.num_blocks_chi):
            self.trunk_chi[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk_chi[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk_chi[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk_chi[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk_chi[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)

            if b < (self.num_blocks_chi - 1):
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk_chi[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t, 
                seqs_t, seqs_1, node_embed, edge_embed, 
                generate_mask, res_mask, Train=True):
        # Make sure that angles_t only contains the O atom angle
        angles_O_t = angles_t[:, :, 0:1]  # Only keep the first angle (O atom angle)
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]

        node_embed = self.res_feat_mixer(torch.cat(
            [node_embed, self.current_seq_embedder(seqs_t), self.embed_t(t, node_mask),
             self.angles_embedder(angles_O_t).reshape(num_batch, num_res, -1)], dim=-1))
        node_embed = node_embed * node_mask[..., None]
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_seqs1_prob = self.seq_net(node_embed)
        pred_angles_O_1 = self.angle_O_net(node_embed)
        pred_angles_O_1 = pred_angles_O_1 % (2 * math.pi)  # inductive bias to bound between (0,2pi)
        if Train:
            pred_seqs_1 = seqs_1
        else:
            pred_seqs_1 = sample_from(F.softmax(pred_seqs1_prob, dim=-1))


        # Predict the chi angles via teacher forcing
        node_embed_chi = self.res_feat_mixer(torch.cat(
            [node_embed, self.current_seq_embedder(pred_seqs_1), self.embed_t(t, node_mask),
             self.angles_embedder(angles_O_t).reshape(num_batch, num_res, -1)], dim=-1))
        node_embed_chi = node_embed_chi * node_mask[..., None]
        for b in range(self.num_blocks_chi):
            ipa_embed = self.trunk_chi[f'ipa_{b}'](
                node_embed_chi,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed_chi = self.trunk_chi[f'ipa_ln_{b}'](node_embed_chi + ipa_embed)
            seq_tfmr_out = self.trunk_chi[f'seq_tfmr_{b}'](
                node_embed_chi, src_key_padding_mask=(1 - node_mask).bool())
            node_embed_chi = node_embed_chi + self.trunk_chi[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed_chi = self.trunk_chi[f'node_transition_{b}'](node_embed_chi)
            node_embed_chi = node_embed_chi * node_mask[..., None]

            if b < (self.num_blocks_chi - 1):
                edge_embed = self.trunk_chi[f'edge_transition_{b}'](
                    node_embed_chi, edge_embed)
                edge_embed *= edge_mask[..., None]

        pred_angles_chi_1 = self.angle_chi_net(node_embed_chi)
        pred_angles_chi_1 = pred_angles_chi_1 % (2 * math.pi)  # inductive bias to bound between (0,2pi)

        pred_angles_1 = torch.cat([pred_angles_O_1, pred_angles_chi_1], dim=-1)  # Concatenate O and chi angles

        return pred_rotmats1, pred_trans1, pred_angles_1, pred_seqs1_prob, pred_seqs_1


class GAEncoder_NS18_PredChi_same_param(nn.Module):
    # Keep the model parameters similar to that of PepFlow
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf

        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12)
        num_angle_O = 1  # Only for the O atom
        self.angle_O_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, num_angle_O)
        )

        num_angle_chi = len(constants.chi_angles_mask[constants.AA.ALA])  # Exclude the O atom
        self.angle_chi_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, num_angle_chi)
        )        

        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(constants.PAD_RESIDUE_INDEX + 1,
                                                 self._ipa_conf.c_s)
        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, constants.PAD_RESIDUE_INDEX - 1)
        )

        # mixer
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=num_angle_O),
                      self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )
        self.chi_trunk = 2

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t, 
                seqs_t, seqs_1, node_embed, edge_embed, 
                generate_mask, res_mask, Train=True):
        # Make sure that angles_t only contains the O atom angle
        angles_O_t = angles_t[:, :, 0:1]  # Only keep the first angle (O atom angle)
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]

        node_embed = self.res_feat_mixer(torch.cat(
            [node_embed, self.current_seq_embedder(seqs_t), self.embed_t(t, node_mask),
             self.angles_embedder(angles_O_t).reshape(num_batch, num_res, -1)], dim=-1))
        node_embed = node_embed * node_mask[..., None]
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_seqs1_prob = self.seq_net(node_embed)
        pred_angles_O_1 = self.angle_O_net(node_embed)
        pred_angles_O_1 = pred_angles_O_1 % (2 * math.pi)  # inductive bias to bound between (0,2pi)
        if Train:
            pred_seqs_1 = seqs_1
        else:
            pred_seqs_1 = sample_from(F.softmax(pred_seqs1_prob, dim=-1))


        # Predict the chi angles via teacher forcing
        node_embed_chi = self.res_feat_mixer(torch.cat(
            [node_embed, self.current_seq_embedder(pred_seqs_1), self.embed_t(t, node_mask),
             self.angles_embedder(angles_O_t).reshape(num_batch, num_res, -1)], dim=-1))
        node_embed_chi = node_embed_chi * node_mask[..., None]
        for b in range(self.chi_trunk):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed_chi,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed_chi = self.trunk[f'ipa_ln_{b}'](node_embed_chi + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed_chi, src_key_padding_mask=(1 - node_mask).bool())
            node_embed_chi = node_embed_chi + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed_chi = self.trunk[f'node_transition_{b}'](node_embed_chi)
            node_embed_chi = node_embed_chi * node_mask[..., None]

            if b < (self.chi_trunk - 1):
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed_chi, edge_embed)
                edge_embed *= edge_mask[..., None]

        pred_angles_chi_1 = self.angle_chi_net(node_embed_chi)
        pred_angles_chi_1 = pred_angles_chi_1 % (2 * math.pi)  # inductive bias to bound between (0,2pi)

        pred_angles_1 = torch.cat([pred_angles_O_1, pred_angles_chi_1], dim=-1)  # Concatenate O and chi angles

        return pred_rotmats1, pred_trans1, pred_angles_1, pred_seqs1_prob, pred_seqs_1


class GAEncoder_NS18_PredChi_lighter(nn.Module):
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf

        # angles
        num_angle_chi = len(constants.chi_angles_mask[constants.AA.ALA]) + 1  # Including O atom
        self.angle_chi_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, num_angle_chi)
        )

        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(constants.PAD_RESIDUE_INDEX + 1,
                                                 self._ipa_conf.c_s)
        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, constants.PAD_RESIDUE_INDEX - 1)
        )

        # mixer
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(3 * self._ipa_conf.c_s,
                      self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

        # Attention trunk for Chi prediction
        self.num_blocks_chi = self._ipa_conf.num_blocks - 4
        self.trunk_chi = nn.ModuleDict()
        for b in range(self.num_blocks_chi):
            self.trunk_chi[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk_chi[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk_chi[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk_chi[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk_chi[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)

            if b < (self.num_blocks_chi - 1):
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk_chi[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t, 
                seqs_t, seqs_1, node_embed, edge_embed, 
                generate_mask, res_mask, Train=True):
        # Make sure that angles_t only contains the O atom angle
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]

        node_embed = self.res_feat_mixer(torch.cat(
            [node_embed, self.current_seq_embedder(seqs_t), self.embed_t(t, node_mask)], dim=-1))
        node_embed = node_embed * node_mask[..., None]
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_seqs1_prob = self.seq_net(node_embed)
        if Train:
            pred_seqs_1 = seqs_1
        else:
            pred_seqs_1 = sample_from(F.softmax(pred_seqs1_prob, dim=-1))


        # Predict the chi angles via teacher forcing
        node_embed_chi = self.res_feat_mixer(torch.cat(
            [node_embed, self.current_seq_embedder(pred_seqs_1), self.embed_t(t, node_mask)], dim=-1))
        node_embed_chi = node_embed_chi * node_mask[..., None]
        for b in range(self.num_blocks_chi):
            ipa_embed = self.trunk_chi[f'ipa_{b}'](
                node_embed_chi,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed_chi = self.trunk_chi[f'ipa_ln_{b}'](node_embed_chi + ipa_embed)
            seq_tfmr_out = self.trunk_chi[f'seq_tfmr_{b}'](
                node_embed_chi, src_key_padding_mask=(1 - node_mask).bool())
            node_embed_chi = node_embed_chi + self.trunk_chi[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed_chi = self.trunk_chi[f'node_transition_{b}'](node_embed_chi)
            node_embed_chi = node_embed_chi * node_mask[..., None]

            if b < (self.num_blocks_chi - 1):
                edge_embed = self.trunk_chi[f'edge_transition_{b}'](
                    node_embed_chi, edge_embed)
                edge_embed *= edge_mask[..., None]

        pred_angles_1 = self.angle_chi_net(node_embed_chi)
        pred_angles_1 = pred_angles_1 % (2 * math.pi)  # inductive bias to bound between (0,2pi)

        return pred_rotmats1, pred_trans1, pred_angles_1, pred_seqs1_prob, pred_seqs_1


class GAEncoder_NS18_predSC9_PredChi(nn.Module):
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf

        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12)
        num_angle_O = 1  # Only for the O atom
        self.angle_O_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, num_angle_O)
        )

        num_angle_chi = len(constants.chi_angles_mask[constants.AA.ALA])  # Exclude the O atom
        self.angle_chi_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, num_angle_chi)
        )        

        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(constants.PAD_RESIDUE_INDEX + 1,
                                                 self._ipa_conf.c_s)
        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, constants.PAD_RESIDUE_INDEX - 1)
        )

        # mixer
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=num_angle_O),
                      self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

        # Attention trunk for Chi prediction
        self.num_blocks_chi = self._ipa_conf.num_blocks - 4
        self.trunk_chi = nn.ModuleDict()
        for b in range(self.num_blocks_chi):
            self.trunk_chi[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk_chi[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk_chi[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk_chi[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk_chi[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)

            if b < (self.num_blocks_chi - 1):
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk_chi[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

        self.sc_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, (constants.max_num_heavyatoms - 6) * 3)  # 3D coordinates of 11 side chain atoms
        )

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t, 
                seqs_t, seqs_1, node_embed, edge_embed, 
                generate_mask, res_mask, Train=True):
        # Make sure that angles_t only contains the O atom angle
        angles_O_t = angles_t[:, :, 0:1]  # Only keep the first angle (O atom angle)
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]

        node_embed = self.res_feat_mixer(torch.cat(
            [node_embed, self.current_seq_embedder(seqs_t), self.embed_t(t, node_mask),
             self.angles_embedder(angles_O_t).reshape(num_batch, num_res, -1)], dim=-1))
        node_embed = node_embed * node_mask[..., None]
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_seqs1_prob = self.seq_net(node_embed)
        pred_angles_O_1 = self.angle_O_net(node_embed)
        pred_angles_O_1 = pred_angles_O_1 % (2 * math.pi)  # inductive bias to bound between (0,2pi)
        pred_sctrans_ep = self.sc_net(node_embed)
        pred_sctrans_1 = ipa_pytorch.sctrans_organize(pred_sctrans_ep)
        if Train:
            pred_seqs_1 = seqs_1
        else:
            pred_seqs_1 = sample_from(F.softmax(pred_seqs1_prob, dim=-1))


        # Predict the chi angles via teacher forcing
        node_embed_chi = self.res_feat_mixer(torch.cat(
            [node_embed, self.current_seq_embedder(pred_seqs_1), self.embed_t(t, node_mask),
             self.angles_embedder(angles_O_t).reshape(num_batch, num_res, -1)], dim=-1))
        node_embed_chi = node_embed_chi * node_mask[..., None]
        for b in range(self.num_blocks_chi):
            ipa_embed = self.trunk_chi[f'ipa_{b}'](
                node_embed_chi,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed_chi = self.trunk_chi[f'ipa_ln_{b}'](node_embed_chi + ipa_embed)
            seq_tfmr_out = self.trunk_chi[f'seq_tfmr_{b}'](
                node_embed_chi, src_key_padding_mask=(1 - node_mask).bool())
            node_embed_chi = node_embed_chi + self.trunk_chi[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed_chi = self.trunk_chi[f'node_transition_{b}'](node_embed_chi)
            node_embed_chi = node_embed_chi * node_mask[..., None]

            if b < (self.num_blocks_chi - 1):
                edge_embed = self.trunk_chi[f'edge_transition_{b}'](
                    node_embed_chi, edge_embed)
                edge_embed *= edge_mask[..., None]

        pred_angles_chi_1 = self.angle_chi_net(node_embed_chi)
        pred_angles_chi_1 = pred_angles_chi_1 % (2 * math.pi)  # inductive bias to bound between (0,2pi)

        pred_angles_1 = torch.cat([pred_angles_O_1, pred_angles_chi_1], dim=-1)  # Concatenate O and chi angles

        return pred_rotmats1, pred_trans1, pred_angles_1, pred_seqs1_prob, pred_sctrans_1, pred_seqs_1


class GAEncoder_NS18_predSC9_PredChi_SC(nn.Module):
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf

        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12)
        num_angle_O = 1  # Only for the O atom
        self.angle_O_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, num_angle_O)
        )

        num_angle_chi = len(constants.chi_angles_mask[constants.AA.ALA])  # Exclude the O atom
        self.angle_chi_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, num_angle_chi)
        )        

        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(constants.PAD_RESIDUE_INDEX + 1,
                                                 self._ipa_conf.c_s)
        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, constants.PAD_RESIDUE_INDEX - 1)
        )

        # mixer
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=num_angle_O),
                      self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

        # Attention trunk for Chi prediction
        self.num_blocks_chi = self._ipa_conf.num_blocks - 4
        self.trunk_chi = nn.ModuleDict()
        for b in range(self.num_blocks_chi):
            self.trunk_chi[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk_chi[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk_chi[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk_chi[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk_chi[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)

            if b < (self.num_blocks_chi - 1):
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk_chi[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

        self.sc_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, (constants.max_num_heavyatoms - 6) * 3)  # 3D coordinates of 11 side chain atoms
        )

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t, 
                seqs_t, seqs_1, node_embed, edge_embed, 
                generate_mask, res_mask, Train=True):
        # Make sure that angles_t only contains the O atom angle
        angles_O_t = angles_t[:, :, 0:1]  # Only keep the first angle (O atom angle)
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]

        node_embed = self.res_feat_mixer(torch.cat(
            [node_embed, self.current_seq_embedder(seqs_t), self.embed_t(t, node_mask),
             self.angles_embedder(angles_O_t).reshape(num_batch, num_res, -1)], dim=-1))
        node_embed = node_embed * node_mask[..., None]
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_seqs1_prob = self.seq_net(node_embed)
        pred_angles_O_1 = self.angle_O_net(node_embed)
        pred_angles_O_1 = pred_angles_O_1 % (2 * math.pi)  # inductive bias to bound between (0,2pi)

        if Train:
            pred_seqs_1 = seqs_1
        else:
            pred_seqs_1 = sample_from(F.softmax(pred_seqs1_prob, dim=-1))


        # Predict the chi angles via teacher forcing
        node_embed_chi = self.res_feat_mixer(torch.cat(
            [node_embed, self.current_seq_embedder(pred_seqs_1), self.embed_t(t, node_mask),
             self.angles_embedder(angles_O_t).reshape(num_batch, num_res, -1)], dim=-1))
        node_embed_chi = node_embed_chi * node_mask[..., None]
        for b in range(self.num_blocks_chi):
            ipa_embed = self.trunk_chi[f'ipa_{b}'](
                node_embed_chi,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed_chi = self.trunk_chi[f'ipa_ln_{b}'](node_embed_chi + ipa_embed)
            seq_tfmr_out = self.trunk_chi[f'seq_tfmr_{b}'](
                node_embed_chi, src_key_padding_mask=(1 - node_mask).bool())
            node_embed_chi = node_embed_chi + self.trunk_chi[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed_chi = self.trunk_chi[f'node_transition_{b}'](node_embed_chi)
            node_embed_chi = node_embed_chi * node_mask[..., None]

            if b < (self.num_blocks_chi - 1):
                edge_embed = self.trunk_chi[f'edge_transition_{b}'](
                    node_embed_chi, edge_embed)
                edge_embed *= edge_mask[..., None]

        pred_angles_chi_1 = self.angle_chi_net(node_embed_chi)
        pred_angles_chi_1 = pred_angles_chi_1 % (2 * math.pi)  # inductive bias to bound between (0,2pi)
        pred_angles_1 = torch.cat([pred_angles_O_1, pred_angles_chi_1], dim=-1)  # Concatenate O and chi angles
        pred_sctrans_ep = self.sc_net(node_embed_chi)
        pred_sctrans_1 = ipa_pytorch.sctrans_organize(pred_sctrans_ep)

        return pred_rotmats1, pred_trans1, pred_angles_1, pred_seqs1_prob, pred_sctrans_1, pred_seqs_1


class GAEncoder_NS3_predSC9(nn.Module):
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf

        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12)  # 25*5=120, for competitive embedding size
        self.angle_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 5)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(constants.PAD_RESIDUE_INDEX + 1,
                                                 self._ipa_conf.c_s)
        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, constants.PAD_RESIDUE_INDEX - 1)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # mixer
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=5),
                      self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

        self.sc_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, (constants.max_num_heavyatoms - 6) * 3)  # 3D coordinates of 11 side chain atoms
        )

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t, seqs_t, node_embed, edge_embed, generate_mask, res_mask):
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]

        node_embed = self.res_feat_mixer(torch.cat(
            [node_embed, self.current_seq_embedder(seqs_t), self.embed_t(t, node_mask),
             self.angles_embedder(angles_t).reshape(num_batch, num_res, -1)], dim=-1))
        node_embed = node_embed * node_mask[..., None]
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_seqs1_prob = self.seq_net(node_embed)
        pred_angles1 = self.angle_net(node_embed)
        pred_angles1 = pred_angles1 % (2 * math.pi)  # inductive bias to bound between (0,2pi)

        pred_sctrans_ep = self.sc_net(node_embed)
        pred_sctrans_1 = ipa_pytorch.sctrans_organize(pred_sctrans_ep)

        return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob, pred_sctrans_1


class GAEncoder_NS3_predSC9_noEnc(nn.Module):
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf

        # angles
        self.angle_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 5)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, constants.PAD_RESIDUE_INDEX - 1)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

        self.sc_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, (constants.max_num_heavyatoms - 6) * 3)  # 3D coordinates of 11 side chain atoms
        )

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t, seqs_t, node_embed, edge_embed, generate_mask, res_mask):
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        node_embed = node_embed * node_mask[..., None]
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_seqs1_prob = self.seq_net(node_embed)
        pred_angles1 = self.angle_net(node_embed)
        pred_angles1 = pred_angles1 % (2 * math.pi)  # inductive bias to bound between (0,2pi)

        pred_sctrans_ep = self.sc_net(node_embed)
        pred_sctrans_1 = ipa_pytorch.sctrans_organize(pred_sctrans_ep)

        return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob, pred_sctrans_1


class GAEncoder_NS3_predSC9_distEdge(nn.Module):
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf

        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12)  # 25*5=120, for competitive embedding size
        self.angle_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 5)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(constants.PAD_RESIDUE_INDEX + 1,
                                                 self._ipa_conf.c_s)
        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, constants.PAD_RESIDUE_INDEX - 1)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # mixer
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=5),
                      self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

        self.distEdge = nn.Sequential(
            nn.Linear(self._ipa_conf.c_z + 1, self._ipa_conf.c_z), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_z, self._ipa_conf.c_z), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_z, self._ipa_conf.c_z)
        )

        self.sc_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, (constants.max_num_heavyatoms - 6) * 3)  # 3D coordinates of 11 side chain atoms
        )

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t, seqs_t, node_embed, edge_embed, generate_mask, res_mask):
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]

        node_embed = self.res_feat_mixer(torch.cat(
            [node_embed, self.current_seq_embedder(seqs_t), self.embed_t(t, node_mask),
             self.angles_embedder(angles_t).reshape(num_batch, num_res, -1)], dim=-1))
        node_embed = node_embed * node_mask[..., None]
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                _, ca_dist = ipa_pytorch.ca_attn(curr_rigids, mask=edge_mask, offset_diag=True)
                bin_ca_dist = ipa_pytorch.bin_dist_mat(ca_dist, edge_mask)
                edge_embed = self.distEdge(torch.cat([edge_embed, bin_ca_dist[..., None]], dim=-1))

                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_seqs1_prob = self.seq_net(node_embed)
        pred_angles1 = self.angle_net(node_embed)
        pred_angles1 = pred_angles1 % (2 * math.pi)  # inductive bias to bound between (0,2pi)

        pred_sctrans_ep = self.sc_net(node_embed)
        pred_sctrans_1 = ipa_pytorch.sctrans_organize(pred_sctrans_ep)

        return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob, pred_sctrans_1


class GAEncoder_SEP(nn.Module):
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf

        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12)  # 25*5=120, for competitive embedding size
        self.angle_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 5)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(constants.PAD_RESIDUE_INDEX + 1,
                                                 self._ipa_conf.c_s)
        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, constants.PAD_RESIDUE_INDEX - 1)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # mixer
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=5),
                      self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t, seqs_t, node_embed, edge_embed, generate_mask, res_mask):
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]

        node_embed = self.res_feat_mixer(torch.cat(
            [node_embed, self.current_seq_embedder(seqs_t), self.embed_t(t, node_mask),
             self.angles_embedder(angles_t).reshape(num_batch, num_res, -1)], dim=-1))
        node_embed = node_embed * node_mask[..., None]
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_seqs1_prob = self.seq_net(node_embed)
        pred_angles1 = self.angle_net(node_embed)
        pred_angles1 = pred_angles1 % (2 * math.pi)  # inductive bias to bound between (0,2pi)

        return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob


class GAEncoder_tao_scIPA_future(nn.Module):
    """
    What's new:
    cut down parts of the angle head,
    add MLP(sctrans) into node_emb init.,
    update sctrans while frame changes,
    add a soft sctrans into scIPA
    """

    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf

        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12)  # 25*5=120, for competitive embedding size
        self.angle_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.num_angles)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(22, self._ipa_conf.c_s)

        # mixer
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=self._ipa_conf.num_angles) + 27,
                      self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.scIPA(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)
            self.trunk[f'sc_update_{b}'] = ipa_pytorch.SideChainUpdate(
                self._ipa_conf.c_s)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

        # for param in self.parameters():
        #     param.requires_grad = False

        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 20)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # self.extra_enc = Encoder(embed_size=self._ipa_conf.c_s)
        # 9 side chain atoms
        # self.sc_net = nn.Sequential(
        #     nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
        #     nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
        #     nn.Linear(self._ipa_conf.c_s, 27)  # 3D coordinates of 9 side chain atoms
        # )

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t,
                seqs_t, node_embed, edge_embed, sctrans_t,
                res_mask, gen_mask, atom_mask, resseq):
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]

        angles_t = angles_t[..., :self._ipa_conf.num_angles]

        sctrans_t_tile = ipa_pytorch.sctrans_tile(sctrans_t)

        node_embed = self.res_feat_mixer(torch.cat(
            [node_embed, self.current_seq_embedder(seqs_t), self.embed_t(t, node_mask),
             self.angles_embedder(angles_t).reshape(num_batch, num_res, -1),
             sctrans_t_tile], dim=-1))
        node_embed = node_embed * node_mask[..., None]
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        pred_sctrans_1 = copy.deepcopy(sctrans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask,
                pred_sctrans_1)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])
            sc_update = ipa_pytorch.sctrans_organize(self.trunk[f'sc_update_{b}']
                                                     (node_embed * node_mask[..., None]))
            pred_sctrans_1 = pred_sctrans_1 + sc_update * gen_mask[..., None, None]

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_angles1 = self.angle_net(node_embed)
        pred_angles1 = pred_angles1 % (2 * math.pi)  # inductive bias to bound between (0,2pi)
        # TODO: Debug
        """
        pos_heavyatom, _, _ = full_atom_reconstruction(R_bb=rotmats_t, t_bb=trans_t, angles=angles_t,
                                                       aa=seqs_t * res_mask)  # (B,L,14,3), instead of 15, ignore OXT masked
        pos_heavyatom[:, :, 5:] = torch.einsum('blij,blkj->blik', sctrans_t,
                                               rotmats_t) + trans_t[:, :, None]  # abs coor.

        real_res_indices = res_mask.nonzero(as_tuple=True)
        _pos_heavyatom = pos_heavyatom[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res, 14, 3]
        _seqs_t = seqs_t[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res]
        _gen_mask = gen_mask[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res]
        _position_ids = resseq[real_res_indices[0], real_res_indices[1]]
        _lengths = torch.sum(res_mask, dim=-1)
        # EXCLUDING the virtual atoms in receptor
        # _atom_mask = torch.where(gen_mask[..., None].expand_as(atom_mask) == 1,
        #                          torch.ones_like(atom_mask), atom_mask)[real_res_indices[0], real_res_indices[1]]
        # INCLUDING the virtual atoms in receptor
        _atom_mask = ~torch.logical_xor(atom_mask, get_heavyatom_mask(seqs_t)[..., :-1])[
            real_res_indices[0], real_res_indices[1]]
        _atom_mask = torch.where(_gen_mask[:, None].expand_as(_atom_mask) == 1,
                                 torch.ones_like(_atom_mask), _atom_mask)

        _node_embed = node_embed[real_res_indices[0], real_res_indices[1]]

        tmp = self.extra_enc.encode(X=_pos_heavyatom, S=_seqs_t, mask=_gen_mask,
                                    position_ids=_position_ids, lengths=_lengths,
                                    atom_mask=_atom_mask, node_init=_node_embed)
        """
        # TODO: Debug

        pred_seqs1_prob = self.seq_net(node_embed)

        return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob, pred_sctrans_1


class GAEncoder_tao_freeze(nn.Module):
    # What's new: Freeze the parts before and IPA
    # Turn the angle net back
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf

        self.feat_dim = self._ipa_conf.c_s

        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12)  # 25*5=120, for competitive embedding size
        self.angle_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.num_angles)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # mixer
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=self._ipa_conf.num_angles),
                      self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(22, self._ipa_conf.c_s)

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)
            # self.trunk[f'sc_update_{b}'] = ipa_pytorch.SideChainUpdate(
            #     self._ipa_conf.c_s)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

        for param in self.parameters():
            param.requires_grad = False

        # Post HERN
        # self.HERN = HierEGNNEncoder(hidden_size=self._ipa_conf.c_s, update_X=True)
        # Post HERN

        # Post scIPA
        self.sctrunk = nn.ModuleDict()
        self.sctrunk[f'ipa'] = ipa_pytorch.scIPA(self._ipa_conf)
        self.sctrunk[f'ipa_ln'] = nn.LayerNorm(self._ipa_conf.c_s)
        tfmr_in = self._ipa_conf.c_s
        tfmr_layer = torch.nn.TransformerEncoderLayer(
            d_model=tfmr_in,
            nhead=self._ipa_conf.seq_tfmr_num_heads,
            dim_feedforward=tfmr_in,
            batch_first=True,
            dropout=0.0,
            norm_first=False
        )
        self.sctrunk[f'seq_tfmr'] = torch.nn.TransformerEncoder(
            tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
        self.sctrunk[f'post_tfmr'] = ipa_pytorch.Linear(
            tfmr_in, self._ipa_conf.c_s, init="final")
        self.sctrunk[f'node_transition'] = ipa_pytorch.StructureModuleTransition(
            c=self._ipa_conf.c_s)
        # Post scIPA

        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s + 27, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 20)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # self.extra_enc = Encoder(embed_size=self._ipa_conf.c_s)
        # 9 side chain atoms
        self.sc_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s + 27, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 27)  # 3D coordinates of 9 side chain atoms
        )
        # self.sc_net = ipa_pytorch.SideChainUpdate(self._ipa_conf.c_s)

    def embed_t(self, timesteps, mask, dim=None):
        if dim is None:
            dim = self.feat_dim

        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t,
                seqs_t, node_embed, edge_embed, sctrans_t,
                res_mask, gen_mask, atom_mask, resseq):
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        angle_emb = self.angles_embedder(angles_t)
        node_embed = self.res_feat_mixer(torch.cat(
            [node_embed, self.current_seq_embedder(seqs_t), self.embed_t(t, node_mask),
             angle_emb.reshape(num_batch, num_res, -1)], dim=-1))

        node_embed = node_embed * node_mask[..., None]
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        # pred_sctrans_1 = copy.deepcopy(sctrans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask, )
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])
            # pred_sctrans_1 = self.trunk[f'sc_update_{b}'](
            #     pred_sctrans_1, node_embed * node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_angles1 = self.angle_net(node_embed)
        # pred_angles_ep = self.angle_net(torch.cat([node_embed, angle_emb], dim=-1))
        # pred_angles1 = (angles_t + pred_angles_ep) % (2 * math.pi)  # inductive bias to bound between (0,2pi)
        # TODO: Debug
        """
        pos_heavyatom, _, _ = full_atom_reconstruction(R_bb=rotmats_t, t_bb=trans_t, angles=angles_t,
                                                       aa=seqs_t * res_mask)  # (B,L,14,3), instead of 15, ignore OXT masked
        pos_heavyatom[:, :, 5:] = torch.einsum('blij,blkj->blik', sctrans_t,
                                               rotmats_t) + trans_t[:, :, None]  # abs coor.

        real_res_indices = res_mask.nonzero(as_tuple=True)
        _pos_heavyatom = pos_heavyatom[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res, 14, 3]
        _seqs_t = seqs_t[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res]
        _gen_mask = gen_mask[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res]
        _position_ids = resseq[real_res_indices[0], real_res_indices[1]]
        _lengths = torch.sum(res_mask, dim=-1)
        # EXCLUDING the virtual atoms in receptor
        # _atom_mask = torch.where(gen_mask[..., None].expand_as(atom_mask) == 1,
        #                          torch.ones_like(atom_mask), atom_mask)[real_res_indices[0], real_res_indices[1]]
        # INCLUDING the virtual atoms in receptor
        _atom_mask = ~torch.logical_xor(atom_mask, get_heavyatom_mask(seqs_t)[..., :-1])[
            real_res_indices[0], real_res_indices[1]]
        _atom_mask = torch.where(_gen_mask[:, None].expand_as(_atom_mask) == 1,
                                 torch.ones_like(_atom_mask), _atom_mask)

        _node_embed = node_embed[real_res_indices[0], real_res_indices[1]]

        tmp = self.extra_enc.encode(X=_pos_heavyatom, S=_seqs_t, mask=_gen_mask,
                                    position_ids=_position_ids, lengths=_lengths,
                                    atom_mask=_atom_mask, node_init=_node_embed)
        """
        # TODO: Debug
        # Post HERN
        """
        node_embed, pos = self.HERN(rot=pred_rotmats1, trans=pred_trans1, angle=pred_angles1,
                                    sctrans=sctrans_t, seq=seqs_t, res_mask=res_mask,
                                    gen_mask=gen_mask, t_emb=self.embed_t(t, node_mask, dim=6),
                                    node_init=node_embed)

        pred_sctrans_1 = (pos[:, :, 5:] - pred_trans1[:, :, None, :]) @ pred_rotmats1
        sctrans_pred_tile = ipa_pytorch.sctrans_tile(pred_sctrans_1)
        pred_seqs1_prob = self.seq_net(torch.cat((node_embed, sctrans_pred_tile), dim=-1))
        """
        # Post HERN

        # Post scIPA
        ipa_embed = self.sctrunk[f'ipa'](
            node_embed,
            edge_embed,
            curr_rigids,
            node_mask,
            sctrans_t)
        ipa_embed *= node_mask[..., None]
        node_embed = self.sctrunk[f'ipa_ln'](node_embed + ipa_embed)
        seq_tfmr_out = self.sctrunk[f'seq_tfmr'](
            node_embed, src_key_padding_mask=(1 - node_mask).bool())
        node_embed = node_embed + self.sctrunk[f'post_tfmr'](seq_tfmr_out)
        node_embed = self.sctrunk[f'node_transition'](node_embed)
        node_embed = node_embed * node_mask[..., None]
        # Post scIPA

        sctrans_t_tile = ipa_pytorch.sctrans_tile(sctrans_t)
        pred_sctrans_ep = self.sc_net(torch.cat((node_embed, sctrans_t_tile), dim=-1))
        pred_sctrans_1 = sctrans_t + ipa_pytorch.sctrans_organize(pred_sctrans_ep)
        pred_seqs1_prob = self.seq_net(torch.cat((node_embed, sctrans_t_tile), dim=-1))

        return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob, pred_sctrans_1


class GAEncoder_tao_freeze_v2(nn.Module):
    # What's new: Freeze the parts before and IPA
    # Add a two-layers scIPA with original EdgeUpdate
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf

        self.feat_dim = self._ipa_conf.c_s

        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12)  # 25*5=120, for competitive embedding size
        self.angle_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.num_angles)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # mixer
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=self._ipa_conf.num_angles),
                      self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(22, self._ipa_conf.c_s)

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)
            # self.trunk[f'sc_update_{b}'] = ipa_pytorch.SideChainUpdate(
            #     self._ipa_conf.c_s)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

        for param in self.parameters():
            param.requires_grad = False

        # Post HERN
        # self.HERN = HierEGNNEncoder(hidden_size=self._ipa_conf.c_s, update_X=True)
        # Post HERN

        # Post scIPA
        self.sctrunk = nn.ModuleDict()
        for b in range(2):
            self.sctrunk[f'ipa_{b}'] = ipa_pytorch.scIPA(self._ipa_conf)
            self.sctrunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.sctrunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.sctrunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.sctrunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            if b < 1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.sctrunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )
                # self.sctrunk[f'edge_transition_{b}'] =\
                #     ipa_pytorch.GraphTriangleBlock(**ipa_conf.graph_triangle_block)
            # Post scIPA

        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s + 27, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 20)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )


        # self.extra_enc = Encoder(embed_size=self._ipa_conf.c_s)
        # 9 side chain atoms
        self.sc_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s + 27, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 27)  # 3D coordinates of 9 side chain atoms
        )
        # self.sc_net = ipa_pytorch.SideChainUpdate(self._ipa_conf.c_s)

    def embed_t(self, timesteps, mask, dim=None):
        if dim is None:
            dim = self.feat_dim

        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t,
                seqs_t, node_embed, edge_embed, sctrans_t,
                res_mask, gen_mask, atom_mask, resseq):
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        angle_emb = self.angles_embedder(angles_t)
        node_embed = self.res_feat_mixer(torch.cat(
            [node_embed, self.current_seq_embedder(seqs_t), self.embed_t(t, node_mask),
             angle_emb.reshape(num_batch, num_res, -1)], dim=-1))

        node_embed = node_embed * node_mask[..., None]
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        # pred_sctrans_1 = copy.deepcopy(sctrans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask, )
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])
            # pred_sctrans_1 = self.trunk[f'sc_update_{b}'](
            #     pred_sctrans_1, node_embed * node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_angles1 = self.angle_net(node_embed)
        # pred_angles_ep = self.angle_net(torch.cat([node_embed, angle_emb], dim=-1))
        # pred_angles1 = (angles_t + pred_angles_ep) % (2 * math.pi)  # inductive bias to bound between (0,2pi)
        # TODO: Debug
        """
        pos_heavyatom, _, _ = full_atom_reconstruction(R_bb=rotmats_t, t_bb=trans_t, angles=angles_t,
                                                       aa=seqs_t * res_mask)  # (B,L,14,3), instead of 15, ignore OXT masked
        pos_heavyatom[:, :, 5:] = torch.einsum('blij,blkj->blik', sctrans_t,
                                               rotmats_t) + trans_t[:, :, None]  # abs coor.

        real_res_indices = res_mask.nonzero(as_tuple=True)
        _pos_heavyatom = pos_heavyatom[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res, 14, 3]
        _seqs_t = seqs_t[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res]
        _gen_mask = gen_mask[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res]
        _position_ids = resseq[real_res_indices[0], real_res_indices[1]]
        _lengths = torch.sum(res_mask, dim=-1)
        # EXCLUDING the virtual atoms in receptor
        # _atom_mask = torch.where(gen_mask[..., None].expand_as(atom_mask) == 1,
        #                          torch.ones_like(atom_mask), atom_mask)[real_res_indices[0], real_res_indices[1]]
        # INCLUDING the virtual atoms in receptor
        _atom_mask = ~torch.logical_xor(atom_mask, get_heavyatom_mask(seqs_t)[..., :-1])[
            real_res_indices[0], real_res_indices[1]]
        _atom_mask = torch.where(_gen_mask[:, None].expand_as(_atom_mask) == 1,
                                 torch.ones_like(_atom_mask), _atom_mask)

        _node_embed = node_embed[real_res_indices[0], real_res_indices[1]]

        tmp = self.extra_enc.encode(X=_pos_heavyatom, S=_seqs_t, mask=_gen_mask,
                                    position_ids=_position_ids, lengths=_lengths,
                                    atom_mask=_atom_mask, node_init=_node_embed)
        """
        # TODO: Debug
        # Post HERN
        """
        node_embed, pos = self.HERN(rot=pred_rotmats1, trans=pred_trans1, angle=pred_angles1,
                                    sctrans=sctrans_t, seq=seqs_t, res_mask=res_mask,
                                    gen_mask=gen_mask, t_emb=self.embed_t(t, node_mask, dim=6),
                                    node_init=node_embed)

        pred_sctrans_1 = (pos[:, :, 5:] - pred_trans1[:, :, None, :]) @ pred_rotmats1
        sctrans_pred_tile = ipa_pytorch.sctrans_tile(pred_sctrans_1)
        pred_seqs1_prob = self.seq_net(torch.cat((node_embed, sctrans_pred_tile), dim=-1))
        """
        # Post HERN

        # Post scIPA
        for b in range(2):
            ipa_embed = self.sctrunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask,
                sctrans_t)
            ipa_embed *= node_mask[..., None]
            node_embed = self.sctrunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.sctrunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.sctrunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.sctrunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            if b < 1:
                edge_embed = self.sctrunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                # edge_embed = self.sctrunk[f'edge_transition_{b}'](
                #     node_embed, edge_embed, curr_rigids, edge_mask)
                edge_embed *= edge_mask[..., None]
        # Post scIPA

        sctrans_t_tile = ipa_pytorch.sctrans_tile(sctrans_t)
        pred_sctrans_ep = self.sc_net(torch.cat((node_embed, sctrans_t_tile), dim=-1))
        pred_sctrans_1 = sctrans_t + ipa_pytorch.sctrans_organize(pred_sctrans_ep)
        pred_seqs1_prob = self.seq_net(torch.cat((node_embed, sctrans_t_tile), dim=-1))

        return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob, pred_sctrans_1


class GAEncoder_tao_dyMEAN_v2(nn.Module):
    # What's new: sctrans_ep v2; hard sc_att; discard redundant angles 2024.11.11
    # What's new: Add sctrans (pred_sctrans) into SeqNet 2024.11.12
    # What's new:
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf

        # angles
        # self.angles_embedder = AngularEncoding(num_funcs=12)  # 25*5=120, for competitive embedding size
        self.angle_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.num_angles)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # for condition on current seq
        # self.current_seq_embedder = nn.Embedding(22, self._ipa_conf.c_s)

        # mixer
        # self.res_feat_mixer = nn.Sequential(
        #     nn.Linear(3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=self._ipa_conf.num_angles),
        #               self._ipa_conf.c_s),
        #     nn.ReLU(),
        #     nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        # )

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.scIPA(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)
            # self.trunk[f'sc_update_{b}'] = ipa_pytorch.SideChainUpdate(
            #     self._ipa_conf.c_s)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

        # for param in self.parameters():
        #     param.requires_grad = False

        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s + 27, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 20)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # self.extra_enc = Encoder(embed_size=self._ipa_conf.c_s)
        # 9 side chain atoms
        self.sc_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s + 27, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 27)  # 3D coordinates of 9 side chain atoms
        )
        # self.sc_net = ipa_pytorch.SideChainUpdate(self._ipa_conf.c_s)

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t,
                seqs_t, node_embed, edge_embed, sctrans_t,
                res_mask, gen_mask, atom_mask, resseq):
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        # node_embed = self.res_feat_mixer(torch.cat(
        #     [node_embed, self.current_seq_embedder(seqs_t), self.embed_t(t, node_mask),
        #      self.angles_embedder(angles_t).reshape(num_batch, num_res, -1)], dim=-1))
        node_embed = node_embed * node_mask[..., None]
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        # pred_sctrans_1 = copy.deepcopy(sctrans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask,
                sctrans_t)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])
            # pred_sctrans_1 = self.trunk[f'sc_update_{b}'](
            #     pred_sctrans_1, node_embed * node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_angles1 = self.angle_net(node_embed)
        pred_angles1 = pred_angles1 % (2 * math.pi)  # inductive bias to bound between (0,2pi)
        # TODO: Debug
        """
        pos_heavyatom, _, _ = full_atom_reconstruction(R_bb=rotmats_t, t_bb=trans_t, angles=angles_t,
                                                       aa=seqs_t * res_mask)  # (B,L,14,3), instead of 15, ignore OXT masked
        pos_heavyatom[:, :, 5:] = torch.einsum('blij,blkj->blik', sctrans_t,
                                               rotmats_t) + trans_t[:, :, None]  # abs coor.

        real_res_indices = res_mask.nonzero(as_tuple=True)
        _pos_heavyatom = pos_heavyatom[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res, 14, 3]
        _seqs_t = seqs_t[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res]
        _gen_mask = gen_mask[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res]
        _position_ids = resseq[real_res_indices[0], real_res_indices[1]]
        _lengths = torch.sum(res_mask, dim=-1)
        # EXCLUDING the virtual atoms in receptor
        # _atom_mask = torch.where(gen_mask[..., None].expand_as(atom_mask) == 1,
        #                          torch.ones_like(atom_mask), atom_mask)[real_res_indices[0], real_res_indices[1]]
        # INCLUDING the virtual atoms in receptor
        _atom_mask = ~torch.logical_xor(atom_mask, get_heavyatom_mask(seqs_t)[..., :-1])[
            real_res_indices[0], real_res_indices[1]]
        _atom_mask = torch.where(_gen_mask[:, None].expand_as(_atom_mask) == 1,
                                 torch.ones_like(_atom_mask), _atom_mask)

        _node_embed = node_embed[real_res_indices[0], real_res_indices[1]]

        tmp = self.extra_enc.encode(X=_pos_heavyatom, S=_seqs_t, mask=_gen_mask,
                                    position_ids=_position_ids, lengths=_lengths,
                                    atom_mask=_atom_mask, node_init=_node_embed)
        """
        # TODO: Debug

        # pred_seqs1_prob = self.seq_net(torch.cat((node_embed, ipa_pytorch.sctrans_tile(sctrans_t)), dim=-1))
        # pred_sctrans_1 = self.sc_net(torch.cat((node_updated, sctrans_t.reshape(num_batch, num_res, -1)), dim=-1)) \
        #     .view(num_batch, num_res, -1, 3)

        sctrans_t_tile = ipa_pytorch.sctrans_tile(sctrans_t)
        pred_sctrans_ep = self.sc_net(torch.cat((node_embed, sctrans_t_tile), dim=-1))
        pred_sctrans_1 = sctrans_t + ipa_pytorch.sctrans_organize(pred_sctrans_ep)
        pred_seqs1_prob = self.seq_net(torch.cat((node_embed, sctrans_t_tile), dim=-1))

        return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob, pred_sctrans_1


class GAEncoder_tao_HERN(nn.Module):
    # What's new: sctrans_ep v2; hard sc_att; discard redundant angles 2024.11.11
    # What's new: Add sctrans (pred_sctrans) into SeqNet 2024.11.12
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf

        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12)  # 25*5=120, for competitive embedding size
        self.angle_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.num_angles)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(22, self._ipa_conf.c_s)

        # mixer
        # self.res_feat_mixer = nn.Sequential(
        #     nn.Linear(3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=self._ipa_conf.num_angles),
        #               self._ipa_conf.c_s),
        #     nn.ReLU(),
        #     nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        # )
        self.res_feat_mixer = HierEGNNEncoder(hidden_size=self._ipa_conf.c_s)

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.scIPA(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)
            # self.trunk[f'sc_update_{b}'] = ipa_pytorch.SideChainUpdate(
            #     self._ipa_conf.c_s)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

        # for param in self.parameters():
        #     param.requires_grad = False

        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s + 27, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 20)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # self.extra_enc = Encoder(embed_size=self._ipa_conf.c_s)
        # 9 side chain atoms
        self.sc_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s + 27, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 27)  # 3D coordinates of 9 side chain atoms
        )
        # self.sc_net = ipa_pytorch.SideChainUpdate(self._ipa_conf.c_s)

    def embed_t(self, timesteps, mask, dim=None):
        if dim is None:
            dim = self.feat_dim

        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t,
                seqs_t, node_embed, edge_embed, sctrans_t,
                res_mask, gen_mask, atom_mask, resseq):
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]

        # node_embed = self.res_feat_mixer(torch.cat(
        #     [node_embed, self.current_seq_embedder(seqs_t), self.embed_t(t, node_mask),
        #      self.angles_embedder(angles_t).reshape(num_batch, num_res, -1)], dim=-1))
        t_emb = self.embed_t(t, node_mask, dim=6)
        node_embed = self.res_feat_mixer(rot=rotmats_t, trans=trans_t, angle=angles_t,
                                         sctrans=sctrans_t, seq=seqs_t, res_mask=res_mask,
                                         gen_mask=gen_mask, t_emb=t_emb, node_init=node_embed)

        node_embed = node_embed * node_mask[..., None]
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        # pred_sctrans_1 = copy.deepcopy(sctrans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask,
                sctrans_t)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])
            # pred_sctrans_1 = self.trunk[f'sc_update_{b}'](
            #     pred_sctrans_1, node_embed * node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_angles1 = self.angle_net(node_embed)
        pred_angles1 = pred_angles1 % (2 * math.pi)  # inductive bias to bound between (0,2pi)
        # TODO: Debug
        """
        pos_heavyatom, _, _ = full_atom_reconstruction(R_bb=rotmats_t, t_bb=trans_t, angles=angles_t,
                                                       aa=seqs_t * res_mask)  # (B,L,14,3), instead of 15, ignore OXT masked
        pos_heavyatom[:, :, 5:] = torch.einsum('blij,blkj->blik', sctrans_t,
                                               rotmats_t) + trans_t[:, :, None]  # abs coor.

        real_res_indices = res_mask.nonzero(as_tuple=True)
        _pos_heavyatom = pos_heavyatom[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res, 14, 3]
        _seqs_t = seqs_t[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res]
        _gen_mask = gen_mask[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res]
        _position_ids = resseq[real_res_indices[0], real_res_indices[1]]
        _lengths = torch.sum(res_mask, dim=-1)
        # EXCLUDING the virtual atoms in receptor
        # _atom_mask = torch.where(gen_mask[..., None].expand_as(atom_mask) == 1,
        #                          torch.ones_like(atom_mask), atom_mask)[real_res_indices[0], real_res_indices[1]]
        # INCLUDING the virtual atoms in receptor
        _atom_mask = ~torch.logical_xor(atom_mask, get_heavyatom_mask(seqs_t)[..., :-1])[
            real_res_indices[0], real_res_indices[1]]
        _atom_mask = torch.where(_gen_mask[:, None].expand_as(_atom_mask) == 1,
                                 torch.ones_like(_atom_mask), _atom_mask)

        _node_embed = node_embed[real_res_indices[0], real_res_indices[1]]

        tmp = self.extra_enc.encode(X=_pos_heavyatom, S=_seqs_t, mask=_gen_mask,
                                    position_ids=_position_ids, lengths=_lengths,
                                    atom_mask=_atom_mask, node_init=_node_embed)
        """
        # TODO: Debug

        # pred_sctrans_1 = self.sc_net(torch.cat((node_updated, sctrans_t.reshape(num_batch, num_res, -1)), dim=-1)) \
        #     .view(num_batch, num_res, -1, 3)
        sctrans_t_tile = ipa_pytorch.sctrans_tile(sctrans_t)
        pred_sctrans_ep = self.sc_net(torch.cat((node_embed, sctrans_t_tile), dim=-1))
        pred_sctrans_1 = sctrans_t + ipa_pytorch.sctrans_organize(pred_sctrans_ep)
        # pred_sctrans_1 = self.sc_net(sctrans_t, node_embed)

        pred_seqs1_prob = self.seq_net(torch.cat((node_embed, sctrans_t_tile), dim=-1))
        return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob, pred_sctrans_1


class GAEncoder_tao_HERN_v2(nn.Module):
    # What's new: Use HERN after the main chunk for SC and Seq. sctrans in rel coor!
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf

        self.feat_dim = self._ipa_conf.c_s

        # mixer
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=self._ipa_conf.num_angles),
                      self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(22, self._ipa_conf.c_s)

        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12)  # 25*5=120, for competitive embedding size
        self.angle_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=self._ipa_conf.num_angles),
                      self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.num_angles)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)
            # self.trunk[f'sc_update_{b}'] = ipa_pytorch.SideChainUpdate(
            #     self._ipa_conf.c_s)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

        for param in self.parameters():
            param.requires_grad = False

        self.HERN = HierEGNNEncoder(hidden_size=self._ipa_conf.c_s, update_X=True)

        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s + 27, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 20)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # self.extra_enc = Encoder(embed_size=self._ipa_conf.c_s)
        # 9 side chain atoms
        self.sc_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s + 27, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 27)  # 3D coordinates of 9 side chain atoms
        )
        # self.sc_net = ipa_pytorch.SideChainUpdate(self._ipa_conf.c_s)

    def embed_t(self, timesteps, mask, dim=None):
        if dim is None:
            dim = self.feat_dim

        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t,
                seqs_t, node_embed, edge_embed, sctrans_t,
                res_mask, gen_mask, atom_mask, resseq):
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        angle_emb = self.angles_embedder(angles_t)
        node_embed = self.res_feat_mixer(torch.cat(
            [node_embed, self.current_seq_embedder(seqs_t), self.embed_t(t, node_mask),
             angle_emb.reshape(num_batch, num_res, -1)], dim=-1))

        node_embed = node_embed * node_mask[..., None]
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        # pred_sctrans_1 = copy.deepcopy(sctrans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask, )
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])
            # pred_sctrans_1 = self.trunk[f'sc_update_{b}'](
            #     pred_sctrans_1, node_embed * node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_angles_ep = self.angle_net(torch.cat([node_embed, angle_emb], dim=-1))
        pred_angles1 = (angles_t + pred_angles_ep) % (2 * math.pi)  # inductive bias to bound between (0,2pi)
        # TODO: Debug
        """
        pos_heavyatom, _, _ = full_atom_reconstruction(R_bb=rotmats_t, t_bb=trans_t, angles=angles_t,
                                                       aa=seqs_t * res_mask)  # (B,L,14,3), instead of 15, ignore OXT masked
        pos_heavyatom[:, :, 5:] = torch.einsum('blij,blkj->blik', sctrans_t,
                                               rotmats_t) + trans_t[:, :, None]  # abs coor.

        real_res_indices = res_mask.nonzero(as_tuple=True)
        _pos_heavyatom = pos_heavyatom[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res, 14, 3]
        _seqs_t = seqs_t[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res]
        _gen_mask = gen_mask[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res]
        _position_ids = resseq[real_res_indices[0], real_res_indices[1]]
        _lengths = torch.sum(res_mask, dim=-1)
        # EXCLUDING the virtual atoms in receptor
        # _atom_mask = torch.where(gen_mask[..., None].expand_as(atom_mask) == 1,
        #                          torch.ones_like(atom_mask), atom_mask)[real_res_indices[0], real_res_indices[1]]
        # INCLUDING the virtual atoms in receptor
        _atom_mask = ~torch.logical_xor(atom_mask, get_heavyatom_mask(seqs_t)[..., :-1])[
            real_res_indices[0], real_res_indices[1]]
        _atom_mask = torch.where(_gen_mask[:, None].expand_as(_atom_mask) == 1,
                                 torch.ones_like(_atom_mask), _atom_mask)

        _node_embed = node_embed[real_res_indices[0], real_res_indices[1]]

        tmp = self.extra_enc.encode(X=_pos_heavyatom, S=_seqs_t, mask=_gen_mask,
                                    position_ids=_position_ids, lengths=_lengths,
                                    atom_mask=_atom_mask, node_init=_node_embed)
        """
        # TODO: Debug
        node_embed, pos = self.HERN(rot=pred_rotmats1, trans=pred_trans1, angle=pred_angles1,
                                    sctrans=sctrans_t, seq=seqs_t, res_mask=res_mask,
                                    gen_mask=gen_mask, t_emb=self.embed_t(t, node_mask, dim=6),
                                    node_init=node_embed)

        pred_sctrans_1 = (pos[:, :, 5:] - pred_trans1[:, :, None, :]) @ pred_rotmats1
        sctrans_pred_tile = ipa_pytorch.sctrans_tile(pred_sctrans_1)
        # pred_sctrans_1 = self.sc_net(torch.cat((node_updated, sctrans_t.reshape(num_batch, num_res, -1)), dim=-1)) \
        #     .view(num_batch, num_res, -1, 3)
        # sctrans_t_tile = ipa_pytorch.sctrans_tile(sctrans_t)
        # pred_sctrans_ep = self.sc_net(torch.cat((node_embed, sctrans_t_tile), dim=-1))
        # pred_sctrans_1 = sctrans_t + ipa_pytorch.sctrans_organize(pred_sctrans_ep)
        # pred_sctrans_1 = self.sc_net(sctrans_t, node_embed)

        pred_seqs1_prob = self.seq_net(torch.cat((node_embed, sctrans_pred_tile), dim=-1))
        return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob, pred_sctrans_1


class GAEncoder_tao_scIPA(nn.Module):
    # What's new: sctrans_ep v2; hard sc_att; discard redundant angles 2024.11.11
    # What's new: Add sctrans (pred_sctrans) into SeqNet 2024.11.12
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf

        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12)  # 25*5=120, for competitive embedding size
        self.angle_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=self._ipa_conf.num_angles),
                      self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.num_angles)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(22, self._ipa_conf.c_s)

        # mixer
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=self._ipa_conf.num_angles),
                      self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)
            # self.trunk[f'sc_update_{b}'] = ipa_pytorch.SideChainUpdate(
            #     self._ipa_conf.c_s)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

        # Post scIPA
        self.sctrunk = nn.ModuleDict()
        self.sctrunk[f'ipa'] = ipa_pytorch.scIPA(self._ipa_conf)
        self.sctrunk[f'ipa_ln'] = nn.LayerNorm(self._ipa_conf.c_s)
        tfmr_in = self._ipa_conf.c_s
        tfmr_layer = torch.nn.TransformerEncoderLayer(
            d_model=tfmr_in,
            nhead=self._ipa_conf.seq_tfmr_num_heads,
            dim_feedforward=tfmr_in,
            batch_first=True,
            dropout=0.0,
            norm_first=False
        )
        self.sctrunk[f'seq_tfmr'] = torch.nn.TransformerEncoder(
            tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
        self.sctrunk[f'post_tfmr'] = ipa_pytorch.Linear(
            tfmr_in, self._ipa_conf.c_s, init="final")
        self.sctrunk[f'node_transition'] = ipa_pytorch.StructureModuleTransition(
            c=self._ipa_conf.c_s)
        # Post scIPA

        # for param in self.parameters():
        #     param.requires_grad = False

        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s + 27, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 20)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # self.extra_enc = Encoder(embed_size=self._ipa_conf.c_s)
        # 9 side chain atoms
        self.sc_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s + 27, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 27)  # 3D coordinates of 9 side chain atoms
        )
        # self.sc_net = ipa_pytorch.SideChainUpdate(self._ipa_conf.c_s)

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t,
                seqs_t, node_embed, edge_embed, sctrans_t,
                res_mask, gen_mask, atom_mask, resseq):
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        angle_emb = self.angles_embedder(angles_t)
        node_embed = self.res_feat_mixer(torch.cat(
            [node_embed, self.current_seq_embedder(seqs_t), self.embed_t(t, node_mask),
             angle_emb.reshape(num_batch, num_res, -1)], dim=-1))

        # t_emb = self.embed_t(t, node_mask) * gen_mask.bool()[..., None]
        #
        # node_embed, _ = self.res_feat_mixer(rot=rotmats_t, trans=trans_t, angle=angles_t,
        #                                  sctrans=sctrans_t, seq=seqs_t, res_mask=res_mask,
        #                                  gen_mask=gen_mask, atom_mask=atom_mask,
        #                                  abs_idx=resseq, t_emb=t_emb, node_init=node_embed)

        node_embed = node_embed * node_mask[..., None]
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        # pred_sctrans_1 = copy.deepcopy(sctrans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])
            # pred_sctrans_1 = self.trunk[f'sc_update_{b}'](
            #     pred_sctrans_1, node_embed * node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        # pred_angles1 = self.angle_net(node_embed)
        pred_angles_ep = self.angle_net(torch.cat([node_embed, angle_emb], dim=-1))
        pred_angles1 = (angles_t + pred_angles_ep) % (2 * math.pi)  # inductive bias to bound between (0,2pi)
        # TODO: Debug
        """
        pos_heavyatom, _, _ = full_atom_reconstruction(R_bb=rotmats_t, t_bb=trans_t, angles=angles_t,
                                                       aa=seqs_t * res_mask)  # (B,L,14,3), instead of 15, ignore OXT masked
        pos_heavyatom[:, :, 5:] = torch.einsum('blij,blkj->blik', sctrans_t,
                                               rotmats_t) + trans_t[:, :, None]  # abs coor.

        real_res_indices = res_mask.nonzero(as_tuple=True)
        _pos_heavyatom = pos_heavyatom[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res, 14, 3]
        _seqs_t = seqs_t[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res]
        _gen_mask = gen_mask[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res]
        _position_ids = resseq[real_res_indices[0], real_res_indices[1]]
        _lengths = torch.sum(res_mask, dim=-1)
        # EXCLUDING the virtual atoms in receptor
        # _atom_mask = torch.where(gen_mask[..., None].expand_as(atom_mask) == 1,
        #                          torch.ones_like(atom_mask), atom_mask)[real_res_indices[0], real_res_indices[1]]
        # INCLUDING the virtual atoms in receptor
        _atom_mask = ~torch.logical_xor(atom_mask, get_heavyatom_mask(seqs_t)[..., :-1])[
            real_res_indices[0], real_res_indices[1]]
        _atom_mask = torch.where(_gen_mask[:, None].expand_as(_atom_mask) == 1,
                                 torch.ones_like(_atom_mask), _atom_mask)

        _node_embed = node_embed[real_res_indices[0], real_res_indices[1]]

        tmp = self.extra_enc.encode(X=_pos_heavyatom, S=_seqs_t, mask=_gen_mask,
                                    position_ids=_position_ids, lengths=_lengths,
                                    atom_mask=_atom_mask, node_init=_node_embed)
        """
        # TODO: Debug

        # Post scIPA
        ipa_embed = self.sctrunk[f'ipa'](
            node_embed,
            edge_embed,
            curr_rigids,
            node_mask,
            sctrans_t)
        ipa_embed *= node_mask[..., None]
        node_embed = self.sctrunk[f'ipa_ln'](node_embed + ipa_embed)
        seq_tfmr_out = self.sctrunk[f'seq_tfmr'](
            node_embed, src_key_padding_mask=(1 - node_mask).bool())
        node_embed = node_embed + self.sctrunk[f'post_tfmr'](seq_tfmr_out)
        node_embed = self.sctrunk[f'node_transition'](node_embed)
        node_embed = node_embed * node_mask[..., None]
        # Post scIPA

        # pred_sctrans_1 = self.sc_net(torch.cat((node_updated, sctrans_t.reshape(num_batch, num_res, -1)), dim=-1)) \
        #     .view(num_batch, num_res, -1, 3)
        sctrans_t_tile = ipa_pytorch.sctrans_tile(sctrans_t)
        pred_sctrans_ep = self.sc_net(torch.cat((node_embed, sctrans_t_tile), dim=-1))
        pred_sctrans_1 = sctrans_t + ipa_pytorch.sctrans_organize(pred_sctrans_ep)
        # pred_sctrans_1 = self.sc_net(sctrans_t, node_embed)

        pred_seqs1_prob = self.seq_net(torch.cat((node_embed, sctrans_t_tile), dim=-1))
        return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob, pred_sctrans_1


class GAEncoder_tao_GTA(nn.Module):
    # What's new: sctrans_ep v2; hard sc_att; discard redundant angles 2024.11.11
    # What's new: Add sctrans (pred_sctrans) into SeqNet 2024.11.12
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf

        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12)  # 25*5=120, for competitive embedding size
        self.angle_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.num_angles)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(22, self._ipa_conf.c_s)

        # mixer
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(3 * self._ipa_conf.c_s + 27
                      + self.angles_embedder.get_out_dim(in_dim=self._ipa_conf.num_angles),
                      self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        self._ipa_conf.num_blocks = 4
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.scIPA(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)
            # self.trunk[f'sc_update_{b}'] = ipa_pytorch.SideChainUpdate(
            #     self._ipa_conf.c_s)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                # edge_in = self._ipa_conf.c_z
                # self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                #     node_embed_size=self._ipa_conf.c_s,
                #     edge_embed_in=edge_in,
                #     edge_embed_out=self._ipa_conf.c_z,
                # )
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.GraphTriangleBlock(**ipa_conf.graph_triangle_block)

        # for param in self.parameters():
        #     param.requires_grad = False

        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s + 27, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 20)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # self.extra_enc = Encoder(embed_size=self._ipa_conf.c_s)
        # 9 side chain atoms
        self.sc_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s + 27, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 27)  # 3D coordinates of 9 side chain atoms
        )
        # self.sc_net = ipa_pytorch.SideChainUpdate(self._ipa_conf.c_s)

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t,
                seqs_t, node_embed, edge_embed, sctrans_t,
                res_mask, gen_mask, atom_mask, resseq):
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]

        node_embed = self.res_feat_mixer(torch.cat(
            [node_embed, self.current_seq_embedder(seqs_t), self.embed_t(t, node_mask),
             self.angles_embedder(angles_t).reshape(num_batch, num_res, -1),
             ipa_pytorch.sctrans_tile(sctrans_t)], dim=-1))
        node_embed = node_embed * node_mask[..., None]
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        # pred_sctrans_1 = copy.deepcopy(sctrans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask,
                sctrans_t)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])
            # pred_sctrans_1 = self.trunk[f'sc_update_{b}'](
            #     pred_sctrans_1, node_embed * node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed, curr_rigids, edge_mask)
                edge_embed *= edge_mask[..., None]

        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_angles1 = self.angle_net(node_embed)
        pred_angles1 = pred_angles1 % (2 * math.pi)  # inductive bias to bound between (0,2pi)
        # TODO: Debug
        """
        pos_heavyatom, _, _ = full_atom_reconstruction(R_bb=rotmats_t, t_bb=trans_t, angles=angles_t,
                                                       aa=seqs_t * res_mask)  # (B,L,14,3), instead of 15, ignore OXT masked
        pos_heavyatom[:, :, 5:] = torch.einsum('blij,blkj->blik', sctrans_t,
                                               rotmats_t) + trans_t[:, :, None]  # abs coor.

        real_res_indices = res_mask.nonzero(as_tuple=True)
        _pos_heavyatom = pos_heavyatom[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res, 14, 3]
        _seqs_t = seqs_t[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res]
        _gen_mask = gen_mask[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res]
        _position_ids = resseq[real_res_indices[0], real_res_indices[1]]
        _lengths = torch.sum(res_mask, dim=-1)
        # EXCLUDING the virtual atoms in receptor
        # _atom_mask = torch.where(gen_mask[..., None].expand_as(atom_mask) == 1,
        #                          torch.ones_like(atom_mask), atom_mask)[real_res_indices[0], real_res_indices[1]]
        # INCLUDING the virtual atoms in receptor
        _atom_mask = ~torch.logical_xor(atom_mask, get_heavyatom_mask(seqs_t)[..., :-1])[
            real_res_indices[0], real_res_indices[1]]
        _atom_mask = torch.where(_gen_mask[:, None].expand_as(_atom_mask) == 1,
                                 torch.ones_like(_atom_mask), _atom_mask)

        _node_embed = node_embed[real_res_indices[0], real_res_indices[1]]

        tmp = self.extra_enc.encode(X=_pos_heavyatom, S=_seqs_t, mask=_gen_mask,
                                    position_ids=_position_ids, lengths=_lengths,
                                    atom_mask=_atom_mask, node_init=_node_embed)
        """
        # TODO: Debug

        # pred_sctrans_1 = self.sc_net(torch.cat((node_updated, sctrans_t.reshape(num_batch, num_res, -1)), dim=-1)) \
        #     .view(num_batch, num_res, -1, 3)
        sctrans_t_tile = ipa_pytorch.sctrans_tile(sctrans_t)
        pred_sctrans_ep = self.sc_net(torch.cat((node_embed, sctrans_t_tile), dim=-1))
        pred_sctrans_1 = sctrans_t + ipa_pytorch.sctrans_organize(pred_sctrans_ep)
        # pred_sctrans_1 = self.sc_net(sctrans_t, node_embed)

        pred_seqs1_prob = self.seq_net(torch.cat((node_embed, sctrans_t_tile), dim=-1))
        return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob, pred_sctrans_1


class GAEncoder_tao_scIPA_1111(nn.Module):
    # What's new: predict the epsilon of sctrans v2 (SCNet(node_emb, sctrans_t))
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf

        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12)  # 25*5=120, for competitive embedding size
        self.angle_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 5)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(22, self._ipa_conf.c_s)

        # mixer
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=5),
                      self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.scIPA(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

        # for param in self.parameters():
        #     param.requires_grad = False

        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 20)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # self.extra_enc = Encoder(embed_size=self._ipa_conf.c_s)
        # 9 side chain atoms
        self.sc_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s + 27, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 27)  # 3D coordinates of 9 side chain atoms
        )

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t,
                seqs_t, node_embed, edge_embed, sctrans_t,
                res_mask, gen_mask, atom_mask, resseq):
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]

        node_embed = self.res_feat_mixer(torch.cat(
            [node_embed, self.current_seq_embedder(seqs_t), self.embed_t(t, node_mask),
             self.angles_embedder(angles_t).reshape(num_batch, num_res, -1)], dim=-1))
        node_embed = node_embed * node_mask[..., None]
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask,
                sctrans_t)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_angles1 = self.angle_net(node_embed)
        pred_angles1 = pred_angles1 % (2 * math.pi)  # inductive bias to bound between (0,2pi)
        # TODO: Debug
        """
        pos_heavyatom, _, _ = full_atom_reconstruction(R_bb=rotmats_t, t_bb=trans_t, angles=angles_t,
                                                       aa=seqs_t * res_mask)  # (B,L,14,3), instead of 15, ignore OXT masked
        pos_heavyatom[:, :, 5:] = torch.einsum('blij,blkj->blik', sctrans_t,
                                               rotmats_t) + trans_t[:, :, None]  # abs coor.

        real_res_indices = res_mask.nonzero(as_tuple=True)
        _pos_heavyatom = pos_heavyatom[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res, 14, 3]
        _seqs_t = seqs_t[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res]
        _gen_mask = gen_mask[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res]
        _position_ids = resseq[real_res_indices[0], real_res_indices[1]]
        _lengths = torch.sum(res_mask, dim=-1)
        # EXCLUDING the virtual atoms in receptor
        # _atom_mask = torch.where(gen_mask[..., None].expand_as(atom_mask) == 1,
        #                          torch.ones_like(atom_mask), atom_mask)[real_res_indices[0], real_res_indices[1]]
        # INCLUDING the virtual atoms in receptor
        _atom_mask = ~torch.logical_xor(atom_mask, get_heavyatom_mask(seqs_t)[..., :-1])[
            real_res_indices[0], real_res_indices[1]]
        _atom_mask = torch.where(_gen_mask[:, None].expand_as(_atom_mask) == 1,
                                 torch.ones_like(_atom_mask), _atom_mask)

        _node_embed = node_embed[real_res_indices[0], real_res_indices[1]]

        tmp = self.extra_enc.encode(X=_pos_heavyatom, S=_seqs_t, mask=_gen_mask,
                                    position_ids=_position_ids, lengths=_lengths,
                                    atom_mask=_atom_mask, node_init=_node_embed)
        """
        # TODO: Debug

        pred_seqs1_prob = self.seq_net(node_embed)
        # pred_sctrans_1 = self.sc_net(torch.cat((node_updated, sctrans_t.reshape(num_batch, num_res, -1)), dim=-1)) \
        #     .view(num_batch, num_res, -1, 3)
        pred_sctrans_ep = self.sc_net(torch.cat((node_embed, ipa_pytorch.sctrans_tile(sctrans_t)), dim=-1))

        pred_sctrans_1 = sctrans_t + ipa_pytorch.sctrans_organize(pred_sctrans_ep)
        return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob, pred_sctrans_1


class GAEncoder_tao_scIPA_1110(nn.Module):
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf

        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12)  # 25*5=120, for competitive embedding size
        self.angle_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 5)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(22, self._ipa_conf.c_s)

        # mixer
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=5),
                      self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.scIPA(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

        # for param in self.parameters():
        #     param.requires_grad = False

        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 20)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # self.extra_enc = Encoder(embed_size=self._ipa_conf.c_s)
        # 9 side chain atoms
        self.sc_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 27)  # 3D coordinates of 9 side chain atoms
        )

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t,
                seqs_t, node_embed, edge_embed, sctrans_t,
                res_mask, gen_mask, atom_mask, resseq):
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]

        node_embed = self.res_feat_mixer(torch.cat(
            [node_embed, self.current_seq_embedder(seqs_t), self.embed_t(t, node_mask),
             self.angles_embedder(angles_t).reshape(num_batch, num_res, -1)], dim=-1))
        node_embed = node_embed * node_mask[..., None]
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask,
                sctrans_t)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_angles1 = self.angle_net(node_embed)
        pred_angles1 = pred_angles1 % (2 * math.pi)  # inductive bias to bound between (0,2pi)
        # TODO: Debug
        """
        pos_heavyatom, _, _ = full_atom_reconstruction(R_bb=rotmats_t, t_bb=trans_t, angles=angles_t,
                                                       aa=seqs_t * res_mask)  # (B,L,14,3), instead of 15, ignore OXT masked
        pos_heavyatom[:, :, 5:] = torch.einsum('blij,blkj->blik', sctrans_t,
                                               rotmats_t) + trans_t[:, :, None]  # abs coor.

        real_res_indices = res_mask.nonzero(as_tuple=True)
        _pos_heavyatom = pos_heavyatom[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res, 14, 3]
        _seqs_t = seqs_t[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res]
        _gen_mask = gen_mask[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res]
        _position_ids = resseq[real_res_indices[0], real_res_indices[1]]
        _lengths = torch.sum(res_mask, dim=-1)
        # EXCLUDING the virtual atoms in receptor
        # _atom_mask = torch.where(gen_mask[..., None].expand_as(atom_mask) == 1,
        #                          torch.ones_like(atom_mask), atom_mask)[real_res_indices[0], real_res_indices[1]]
        # INCLUDING the virtual atoms in receptor
        _atom_mask = ~torch.logical_xor(atom_mask, get_heavyatom_mask(seqs_t)[..., :-1])[
            real_res_indices[0], real_res_indices[1]]
        _atom_mask = torch.where(_gen_mask[:, None].expand_as(_atom_mask) == 1,
                                 torch.ones_like(_atom_mask), _atom_mask)

        _node_embed = node_embed[real_res_indices[0], real_res_indices[1]]

        tmp = self.extra_enc.encode(X=_pos_heavyatom, S=_seqs_t, mask=_gen_mask,
                                    position_ids=_position_ids, lengths=_lengths,
                                    atom_mask=_atom_mask, node_init=_node_embed)
        """
        # TODO: Debug

        pred_seqs1_prob = self.seq_net(node_embed)
        # pred_sctrans_1 = self.sc_net(torch.cat((node_updated, sctrans_t.reshape(num_batch, num_res, -1)), dim=-1)) \
        #     .view(num_batch, num_res, -1, 3)
        pred_sctrans_1 = self.sc_net(node_embed)
        pred_sctrans_1 = torch.split(pred_sctrans_1, pred_sctrans_1.shape[-1] // 3, dim=-1)
        pred_sctrans_1 = torch.stack(pred_sctrans_1, dim=-1)

        return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob, pred_sctrans_1


class GAEncoder_tao_dyMEAN(nn.Module):
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf

        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12)  # 25*5=120, for competitive embedding size
        self.angle_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 5)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(22, self._ipa_conf.c_s)

        # mixer
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=5),
                      self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

        for param in self.parameters():
            param.requires_grad = False

        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s + 27, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 20)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        self.extra_enc = Encoder(embed_size=self._ipa_conf.c_s)
        # 9 side chain atoms
        self.sc_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s + 27, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 27)  # 3D coordinates of 9 side chain atoms
        )

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t,
                seqs_t, node_embed, edge_embed, sctrans_t,
                res_mask, gen_mask, atom_mask, resseq):
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]

        node_embed = self.res_feat_mixer(torch.cat(
            [node_embed, self.current_seq_embedder(seqs_t), self.embed_t(t, node_mask),
             self.angles_embedder(angles_t).reshape(num_batch, num_res, -1)], dim=-1))
        node_embed = node_embed * node_mask[..., None]
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_angles1 = self.angle_net(node_embed)
        pred_angles1 = pred_angles1 % (2 * math.pi)  # inductive bias to bound between (0,2pi)
        # TODO: Debug
        """
        pos_heavyatom, _, _ = full_atom_reconstruction(R_bb=rotmats_t, t_bb=trans_t, angles=angles_t,
                                                       aa=seqs_t * res_mask)  # (B,L,14,3), instead of 15, ignore OXT masked
        pos_heavyatom[:, :, 5:] = torch.einsum('blij,blkj->blik', sctrans_t,
                                               rotmats_t) + trans_t[:, :, None]  # abs coor.

        real_res_indices = res_mask.nonzero(as_tuple=True)
        _pos_heavyatom = pos_heavyatom[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res, 14, 3]
        _seqs_t = seqs_t[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res]
        _gen_mask = gen_mask[real_res_indices[0], real_res_indices[1]]  # [num_all_real_res]
        _position_ids = resseq[real_res_indices[0], real_res_indices[1]]
        _lengths = torch.sum(res_mask, dim=-1)
        # EXCLUDING the virtual atoms in receptor
        # _atom_mask = torch.where(gen_mask[..., None].expand_as(atom_mask) == 1,
        #                          torch.ones_like(atom_mask), atom_mask)[real_res_indices[0], real_res_indices[1]]
        # INCLUDING the virtual atoms in receptor
        _atom_mask = ~torch.logical_xor(atom_mask, get_heavyatom_mask(seqs_t)[..., :-1])[
            real_res_indices[0], real_res_indices[1]]
        _atom_mask = torch.where(_gen_mask[:, None].expand_as(_atom_mask) == 1,
                                 torch.ones_like(_atom_mask), _atom_mask)

        _node_embed = node_embed[real_res_indices[0], real_res_indices[1]]

        tmp = self.extra_enc.encode(X=_pos_heavyatom, S=_seqs_t, mask=_gen_mask,
                                    position_ids=_position_ids, lengths=_lengths,
                                    atom_mask=_atom_mask, node_init=_node_embed)
        """
        # TODO: Debug

        node_updated, _ = self.extra_enc(rot=rotmats_t, trans=trans_t, angle=angles_t,
                                         sctrans=sctrans_t, seq=seqs_t, res_mask=res_mask,
                                         gen_mask=gen_mask, atom_mask=atom_mask,
                                         abs_idx=resseq, node_init=node_embed)

        pred_seqs1_prob = self.seq_net(torch.cat((node_updated, sctrans_t.reshape(num_batch, num_res, -1)), dim=-1))
        pred_sctrans_1 = self.sc_net(torch.cat((node_updated, sctrans_t.reshape(num_batch, num_res, -1)), dim=-1)) \
            .view(num_batch, num_res, -1, 3)

        return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob, pred_sctrans_1


class GAEncoder_tao_1106(nn.Module):
    # seq_head takes node_emb and sctrans as inputs.
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf

        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12)  # 25*5=120, for competitive embedding size
        self.angle_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 5)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(22, self._ipa_conf.c_s)

        # mixer
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=5),
                      self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

        for param in self.parameters():
            param.requires_grad = False

        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s + 27, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 20)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # 9 side chain atoms
        self.sc_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s + 27, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 27)  # 3D coordinates of 9 side chain atoms
        )

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t, seqs_t, node_embed, edge_embed, sctrans_t, res_mask):
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]

        node_embed = self.res_feat_mixer(torch.cat(
            [node_embed, self.current_seq_embedder(seqs_t), self.embed_t(t, node_mask),
             self.angles_embedder(angles_t).reshape(num_batch, num_res, -1)], dim=-1))
        node_embed = node_embed * node_mask[..., None]
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_angles1 = self.angle_net(node_embed)
        pred_angles1 = pred_angles1 % (2 * math.pi)  # inductive bias to bound between (0,2pi)
        pred_seqs1_prob = self.seq_net(torch.cat((node_embed, sctrans_t.reshape(num_batch, num_res, -1)), dim=-1))
        pred_sctrans_1 = self.sc_net(torch.cat((node_embed, sctrans_t.reshape(num_batch, num_res, -1)), dim=-1)) \
            .view(num_batch, num_res, -1, 3)

        return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob, pred_sctrans_1


class GAEncoder_tao_1105(nn.Module):
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf

        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12)  # 25*5=120, for competitive embedding size
        self.angle_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 5)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(22, self._ipa_conf.c_s)
        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 20)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # mixer
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=5),
                      self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

        for param in self.parameters():
            param.requires_grad = False

        # 9 side chain atoms
        self.sc_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s + 27, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 27)  # 3D coordinates of 9 side chain atoms
        )

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t, seqs_t, node_embed, edge_embed, sctrans_t, res_mask):
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]

        node_embed = self.res_feat_mixer(torch.cat(
            [node_embed, self.current_seq_embedder(seqs_t), self.embed_t(t, node_mask),
             self.angles_embedder(angles_t).reshape(num_batch, num_res, -1)], dim=-1))
        node_embed = node_embed * node_mask[..., None]
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_seqs1_prob = self.seq_net(node_embed)
        pred_angles1 = self.angle_net(node_embed)
        pred_angles1 = pred_angles1 % (2 * math.pi)  # inductive bias to bound between (0,2pi)
        pred_sctrans_1 = self.sc_net(torch.cat((node_embed, sctrans_t.reshape(num_batch, num_res, -1)), dim=-1)) \
            .view(num_batch, num_res, -1, 3)

        return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob, pred_sctrans_1


class GAEncoder_tao_1104(nn.Module):
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf

        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12)  # 25*5=120, for competitive embedding size
        self.angle_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 5)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(22, self._ipa_conf.c_s)
        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 20)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # mixer
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=5),
                      self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

        for param in self.parameters():
            param.requires_grad = False

        # 9 side chain atoms
        self.sc_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 27)  # 3D coordinates of 9 side chain atoms
        )

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t, seqs_t, node_embed, edge_embed, gen_mask, res_mask):
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]

        node_embed = self.res_feat_mixer(torch.cat(
            [node_embed, self.current_seq_embedder(seqs_t), self.embed_t(t, node_mask),
             self.angles_embedder(angles_t).reshape(num_batch, num_res, -1)], dim=-1))
        node_embed = node_embed * node_mask[..., None]
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_seqs1_prob = self.seq_net(node_embed)
        pred_angles1 = self.angle_net(node_embed)
        pred_angles1 = pred_angles1 % (2 * math.pi)  # inductive bias to bound between (0,2pi)
        pred_sctrans_1 = self.sc_net(node_embed).view(num_batch, num_res, -1, 3)

        return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob, pred_sctrans_1


class GAEncoder_tao_1103(nn.Module):
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf

        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12)  # 25*5=120, for competitive embedding size
        self.angle_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.num_angles)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(22, self._ipa_conf.c_s)
        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 20)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # 9 side chain atoms
        self.sc_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 27)  # 3D coordinates of 9 side chain atoms
        )

        # mixer
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=self._ipa_conf.num_angles) + 27,
                      self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t, seqs_t, node_embed, edge_embed, sctrans_t, res_mask):
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        node_embed = self.res_feat_mixer(torch.cat(
            [node_embed, self.current_seq_embedder(seqs_t), self.embed_t(t, node_mask),
             self.angles_embedder(angles_t).reshape(num_batch, num_res, -1),
             (sctrans_t + trans_t[:, :, None]).reshape(num_batch, num_res, -1)], dim=-1))
        # TODO: we could add a side chain encoder here
        node_embed = node_embed * node_mask[..., None]
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_seqs1_prob = self.seq_net(node_embed)
        pred_angles1 = self.angle_net(node_embed)
        pred_sctrans1 = self.sc_net(node_embed).view(num_batch, num_res, -1, 3)

        pred_angles1 = pred_angles1 % (2 * math.pi)  # inductive bias to bound between (0,2pi)

        return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob, pred_sctrans1
