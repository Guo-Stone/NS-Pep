import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pdb
import copy
import json
import math
from tqdm.auto import tqdm
import functools
import os
import argparse
import pandas as pd
from copy import deepcopy
from Bio.SeqUtils import seq1
from models_con.pep_dataloader import PepDataset

from pepflow.utils.train import recursive_to

from pepflow.modules.common.geometry import reconstruct_backbone, reconstruct_backbone_partially, align, batch_align
from pepflow.modules.protein.writers import save_pdb
import pepflow.modules.protein.constants as constants
from pepflow.utils.misc import load_config, seed_all
from pepflow.utils.data import PaddingCollate

from models_con.utils import process_dic

from models_con.flow_model import FlowModel, FlowModel_tao, FlowModel_SEP, FlowModel_NS3

from models_con.torsion import full_atom_reconstruction, get_heavyatom_mask

collate_fn = PaddingCollate(eight=False)

import argparse


def item_to_batch(item, nums=32):
    data_list = [deepcopy(item) for i in range(nums)]
    return collate_fn(data_list)


def sample_for_data_bb(data, model, device, save_root, num_steps=200, sample_structure=True, sample_sequence=True,
                       nums=8):
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    batch = recursive_to(item_to_batch(data, nums=nums), device=device)
    traj = model.sample(batch, num_steps=num_steps, sample_bb=sample_structure, sample_seq=sample_sequence)
    final = recursive_to(traj[-1], device=device)
    pos_bb = reconstruct_backbone(R=final['rotmats'], t=final['trans'], aa=final['seqs'], chain_nb=batch['chain_nb'],
                                  res_nb=batch['res_nb'], mask=batch['res_mask'])  # (32,L,4,3)
    pos_ha = F.pad(pos_bb, pad=(0, 0, 0, 15 - 4), value=0.)  # (32,L,A,3) pos14 A=14
    pos_new = torch.where(batch['generate_mask'][:, :, None, None], pos_ha, batch['pos_heavyatom'])
    mask_bb_atoms = torch.zeros_like(batch['mask_heavyatom'])
    mask_bb_atoms[:, :, :4] = True
    mask_new = torch.where(batch['generate_mask'][:, :, None], mask_bb_atoms, batch['mask_heavyatom'])
    aa_new = final['seqs']

    chain_nb = torch.LongTensor([0 if gen_mask else 1 for gen_mask in data['generate_mask']])
    chain_id = ['A' if gen_mask else 'B' for gen_mask in data['generate_mask']]
    icode = [' ' for _ in range(len(data['icode']))]
    for i in range(nums):
        ref_bb_pos = data['pos_heavyatom'][:, :4].cpu()
        pred_bb_pos = pos_new[i][:, :4].cpu()
        data_saved = {
            'chain_nb': data['chain_nb'], 'chain_id': data['chain_id'], 'resseq': data['resseq'],
            'icode': data['icode'],
            'aa': aa_new[i].cpu(), 'mask_heavyatom': mask_new[i].cpu(), 'pos_heavyatom': pos_new[i].cpu(),
        }

        save_pdb(data_saved, path=os.path.join(save_root, f'sample_{i}.pdb'))
    # save_pdb(data,path=os.path.join(save_root,'gt.pdb'))


def sample_traj(data, model, device, save_root, num_steps=200,
                sample_structure=True, sample_sequence=True,
                sample_ang=True, tao_repr=False, nums=16, idx=15):
    # idx refers to a specific sample in a batch
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    batch = recursive_to(item_to_batch(data, nums=nums), device=device)
    
    traj = model.sample(batch, num_steps=num_steps, sample_bb=sample_structure,
                        sample_seq=sample_sequence, sample_ang=sample_ang)
    traj_dynamics = []
    seq_dynamics = []
    for t in range(num_steps):
        final = recursive_to(traj[t], device=device)
        final = {key: value[idx:idx+1, ...] for key, value in final.items()}
        if final['angles'].shape[-1] == 1:
            final['angles'] = F.pad(final['angles'], (0, 4))

        pos_ha, _, _ = full_atom_reconstruction(R_bb=final['rotmats'], t_bb=final['trans'], angles=final['angles'],
                                                aa=final['seqs'])  # (32,L,14,3), instead of 15, ignore OXT masked
        # pos_ha[:, :, 5:] = final['sctrans']  # use side chain all-atom prediction
        
        pos_ha = F.pad(pos_ha, pad=(0, 0, 0, constants.max_num_heavyatoms - pos_ha.shape[2]), value=0.)  # (B,L,A,3) pos14 A=14
        pos_new = torch.where(batch['generate_mask'][idx:idx+1, :, None, None], pos_ha, batch['pos_heavyatom'][idx:idx+1])

        mask_new = get_heavyatom_mask(final['seqs'])
        aa_new = final['seqs']
        # aa_new = torch.where(batch['generate_mask'][idx:idx+1], 5, aa_new)
        if t < num_steps - 1:
            mask_new[..., 3:] = ~batch['generate_mask'][idx:idx+1, :, None] * mask_new[..., 3:]            
            # aa_new = torch.where(batch['generate_mask'][idx:idx+1], 5, aa_new)
        pep_seq = aa_new[batch['generate_mask'][idx:idx+1]].cpu().tolist()
        pep_seq = [str(constants.AA(seq_idx)) for seq_idx in pep_seq]
        pep_seq = [seq1(char) if seq1(char) != 'X' else f'({char})' for char in pep_seq]
        seq_dynamics.append(pep_seq)

        data_saved = {
            'chain_nb': data['chain_nb'], 'chain_id': data['chain_id'], 'resseq': data['resseq'],
            'icode': data['icode'],
            'aa': aa_new[0].cpu(), 'mask_heavyatom': mask_new[0].cpu(), 'pos_heavyatom': pos_new[0].cpu(),
        }
        
        save_pdb(data_saved, path=os.path.join(save_root, f'sample_{idx}_tmp_{t}.pdb'))

        with open(os.path.join(save_root, f'sample_{idx}_tmp_{t}.pdb'), 'r') as f:
            con = f.readlines()

        content = []
        atom_index = 1
        for line in con:
            if line.startswith('ATOM'):
                prefix = line[:6]
                suffix = line[11:]
                line_save = prefix + '{:>5}'.format(atom_index) + suffix
                content.append(line_save)
                atom_index += 1
            elif line.startswith('END'):
                # atom_index = 1
                content.append('ENDMDL' + '\n')
        traj_dynamics.append(content)
        if t < num_steps - 1:
            os.remove(os.path.join(save_root, f'sample_{idx}_tmp_{t}.pdb'))
    traj_dynamics.reverse()
    traj_dynamics = sum(traj_dynamics, [])
    with open(os.path.join(save_root, f'sample_traj_{idx}.pdb'), 'w') as f:
        f.writelines(traj_dynamics)  # Reverse
    with open(os.path.join(save_root, f'seqs_traj_{idx}.json'), 'w') as f:
        json.dump(seq_dynamics, f, indent=4)  # Natural order            
    # pdb.set_trace()  # TODO: Replace the pocket; Reverse the video; Show the dynamics of sequence


def sample_for_data_tao(data, model, device, save_root, num_steps=200,
                        sample_structure=True, sample_sequence=True,
                        sample_ang=True, tao_repr=False, nums=16):
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    batch = recursive_to(item_to_batch(data, nums=nums), device=device)
    """"""
    if False:
        # DEBUG--------
        from pepflow.modules.common.geometry import construct_3d_basis, compose_chain
        from models_con.torsion import _get_rigid_group, _make_psi_chi_rotation_matrices, _get_torsion
        # import pepflow.modules.protein.constants as constants
        from pepflow.modules.common.geometry import compose_rotation_and_translation
        def _make_rotation_matrix(ex, ey):
            # Gram-Schmidt orthogonalization
            ex_normalized = ex / torch.linalg.norm(ex)

            # make ey perpendicular to ex
            ey_normalized = ey - torch.dot(ey, ex_normalized) * ex_normalized
            ey_normalized /= torch.linalg.norm(ey_normalized)

            eznorm = torch.cross(ex_normalized, ey_normalized)
            m = torch.stack([ex_normalized, ey_normalized, eznorm]).transpose(0, 1)  # (3, 3_index)
            return m

        pos = copy.deepcopy(batch['pos_heavyatom'])
        rigid_rotation, rigid_translation, atom14_group, atom14_position = _get_rigid_group(batch['aa'])
        rot_psi, rot_chi1, rot_chi2, rot_chi3, rot_chi4 = \
            _make_psi_chi_rotation_matrices(batch['torsion_angle']).unbind(dim=2)
        MLE_data_PDB = [
            [51.861, 48.436, 17.324],  # N, N-C 1.47-1.49
            [52.977, 47.516, 17.110],  # CA
            [54.055, 48.096, 16.204],  # C
            [53.968, 49.218, 15.705],  # O
            [52.370, 46.213, 16.570],  # CB
            [51.273, 45.532, 17.410],  # CG
            [50.705, 44.360, 16.658],  # CD1
            [51.839, 45.133, 18.760],  # CD2
            [50.979, 48.816, 16.164],  # CN
            [0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000],
        ]
        MLE_data_2oju = [
            [55.376, 28.619, 96.838],  # N
            [56.061, 29.910, 96.999],  # CA
            [57.078, 29.860, 98.134],  # C
            [57.222, 28.838, 98.809],  # O
            [54.928, 30.880, 97.336],  # CB
            [53.879, 31.054, 96.235],  # CG
            [52.894, 32.128, 96.634],  # CD1
            [54.555, 31.423, 94.920],  # CD2
            [54.607, 28.020, 97.976],  # CN
            [0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000], ]
        # N, CA, C, O, CB, CG, CD1, CD2, CN
        TYS_data_PDB = [
            [16.013, 1.950, -4.617],  # N
            [16.552, 1.871, -3.258],  # CA
            [17.545, 0.752, -2.992],  # C
            [18.218, 0.693, -1.959],  # O
            [15.386, 1.742, -2.268],  # CB
            [14.473, 2.961, -2.241],  # CG
            [15.001, 4.212, -1.950],  # CD1
            [13.124, 2.796, -2.486],  # CD2
            [14.186, 5.316, -1.930],  # CE1
            [12.300, 3.899, -2.461],  # CE2
            [12.096, 6.125, -2.152],  # OH
            [12.840, 5.140, -2.179],  # CZ
            [11.372, 6.454, -0.939],  # S
            [12.284, 6.826, 0.109],  # O1
            [10.523, 5.390, -0.492],  # O2
            [10.554, 7.603, -1.197],  # O3
        ]
        TYS_data_3sv2 = [
            [16.072, 3.328, -4.981],  # N
            [16.591, 3.056, -3.645],  # CA
            [17.500, 1.868, -3.553],  # C
            [18.111, 1.626, -2.493],  # O
            [15.432, 2.857, -2.672],  # CB
            [14.488, 3.975, -2.629],  # CG
            [14.912, 5.201, -2.216],  # CD1
            [13.132, 3.759, -3.025],  # CD2
            [13.985, 6.285, -2.186],  # CE1
            [12.223, 4.819, -2.999],  # CE2
            [11.711, 7.134, -2.552],  # OH
            [12.646, 6.070, -2.582],  # CZ
            [10.963, 7.430, -1.290],  # S
            [11.940, 7.921, -0.230],  # O1
            [10.072, 8.459, -1.681],  # O2
            [10.218, 6.276, -0.783],  # O3
        ]
        PTR_data_PDB = [
            [46.366, 11.139, -0.665],  # N
            [44.969, 11.616, -0.749],  # CA
            [44.978, 13.010, -1.358],  # C
            [43.891, 13.514, -1.708],  # O
            [44.332, 11.618, 0.644],  # CB
            [44.885, 12.640, 1.620],  # CG
            [45.913, 12.302, 2.506],  # CD1
            [44.319, 13.921, 1.716],  # CD2
            [46.364, 13.214, 3.480],  # CE1
            [44.753, 14.849, 2.683],  # CE2
            [46.216, 15.385, 4.594],  # OH
            [45.772, 14.487, 3.562],  # CZ
            [45.382, 15.884, 5.757],  # P
            [44.096, 16.422, 5.355],  # O1P
            [46.274, 16.938, 6.218],  # O2P
            [45.279, 14.830, 6.778],  # O3P
            [0.000, 0.000, 0.000],  # OXT, padding
        ]
        PTR_data_3s8o = [
            [8.132, -7.119, -1.355],  # N
            [8.397, -8.000, -0.186],  # CA
            [8.015, -7.241, 1.077],  # C
            [8.306, -6.061, 1.203],  # O
            [9.927, -8.353, -0.162],  # CB
            [10.327, -9.067, -1.449],  # CG
            [10.807, -8.300, -2.494],  # CD1
            [10.160, -10.476, -1.562],  # CD2
            [11.087, -8.932, -3.706],  # CE1
            [10.428, -11.104, -2.802],  # CE2
            [11.241, -10.777, -5.103],  # OH
            [10.917, -10.327, -3.842],  # CZ
            [10.968, -12.322, -5.622],  # P
            [11.113, -12.180, -7.187],  # O1P
            [11.968, -13.266, -5.108],  # O2P
            [9.557, -12.640, -5.308],  # O3P
        ]
        SEP_data = [
            [12.751, 44.134, -4.949],  # N
            [12.373, 44.600, -6.265],  # CA
            [13.435, 45.364, -6.941],  # C
            [14.373, 45.871, -6.303],  # O
            [11.077, 45.353, -6.305],  # CB
            [10.895, 45.809, -7.608],  # OG
            [9.607, 45.328, -8.384],  # P
            [9.500, 46.086, -9.633],  # O1P
            [9.829, 43.907, -8.669],  # O2P
            [8.402, 45.541, -7.535],  # O3P
            [13.281, 45.410, -8.244],  # OXT
            [0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000],
        ]  # PDB
        SEP_data_5emb = [
            [10.685, 0.571, -14.136],  # N
            [9.233, 0.708, -14.143],  # CA
            [8.706, 0.002, -15.383],  # C
            [9.265, 0.170, -16.462],  # O
            [9.021, 2.201, -14.141],  # CB
            [7.628, 2.469, -13.946],  # OG
            [7.271, 4.030, -13.676],  # P
            [8.058, 4.659, -12.593],  # O1P
            [7.539, 4.744, -15.045],  # O2P
            [5.781, 3.933, -13.477],  # O3P
            [0.000, 0.000, 0.000],  # OXT
            [0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000],
        ]  # 5emb, resolution=0.85 A
        std_pos_PDB = torch.tensor([[-0.573, 1.328, -0.000], [0.000, 0.000, 0.000], [1.473, -0.000, -0.000],
                                    [0.646, 1.061, -0.000], [-0.526, -0.900, -1.078], [0.421, 1.327, 0.000],
                                    [1.222, -0.075, -0.996], [-0.254, 1.245, 0.729], [1.282, -0.579, 0.414],
                                    [-0.054, 0.232, -1.470], ]).to(pos.device)
        std_pos = torch.tensor([[-0.445, 1.389, -0.000], [0.000, 0.000, 0.000], [1.521, -0.000, -0.000],
                                [0.608, 1.066, -0.000], [-0.621, -0.568, -1.251], [0.462, 1.355, 0.000],
                                [0.677, 1.476, 0.000], [0.611, 1.347, 0.000], [0.400, -0.792, 1.292],
                                [0.267, -0.878, -1.195], ]).to(pos.device)  # SEP, 5emb
        # N, CA, C, O, CB, OG, P, O1P, O2P, O3P
        std_pos = torch.tensor([[-0.534, 1.370, 0.000], [0.000, 0.000, 0.000], [1.525, -0.000, -0.000],
                                [0.632, -1.059, -0.000], [-0.537, -0.597, -1.302], [0.645, 1.388, 0.000],
                                [0.510, 1.422, -0.000], [0.524, -0.747, 1.221], [0.760, -1.291, 0.000], ]).to(
            pos.device)  # MLE, 2oju
        std_pos = torch.tensor([[-0.648, 1.313, 0.000], [0.000, 0.000, 0.000], [1.520, -0.000, -0.000],
                                [0.659, -1.040, -0.000], [-0.494, -0.798, -1.215], [0.609, 1.396, -0.000],
                                [0.689, 1.206, -0.000], [0.672, -1.206, -0.022], [2.062, 1.214, -0.007],
                                [2.049, -1.204, -0.022], [3.958, -0.000, -0.019], [2.724, -0.003, -0.013],
                                [0.725, -0.015, 1.256], [0.506, 1.346, 0.004], [0.556, -0.727, -1.103],
                                [0.453, -0.612, 1.215]]).to(pos.device)  # TYS, PDB
        std_pos = torch.tensor([[-0.462, 1.404, 0.000], [0.000, 0.000, 0.000], [1.521, -0.000, -0.000],
                                [0.597, 1.097, -0.000], [-0.560, -0.735, -1.222], [0.659, 1.367, -0.000],
                                [0.717, 1.200, -0.000], [0.718, -1.203, -0.088], [2.122, 1.210, -0.099],
                                [2.123, -1.218, -0.189], [4.246, 0.026, -0.344], [2.816, -0.009, -0.197],
                                [0.873, 0.044, 1.239], [0.506, 1.346, 0.004], [0.556, -0.727, -1.103],
                                [0.453, -0.612, 1.215]]).to(pos.device)  # PTR, PDB
        NS_idx = 61
        pos[0, NS_idx] = torch.tensor(PTR_data_PDB, device=pos.device)
        # pdb.set_trace()
        t_bb = pos[:, :, 1]
        R_bb = construct_3d_basis(pos[:, :, 1],
                                  pos[:, :, 2],
                                  pos[:, :, 0])
        # torch.linalg.norm(pos[0, NS_idx, 0] - pos[0, NS_idx, 1])
        # pos_ideal = (pos - t_bb[:, :, None]) @ R_bb
        SEP_psi = _get_torsion(pos[0, NS_idx, 0], pos[0, NS_idx, 1],
                               pos[0, NS_idx, 2], pos[0, NS_idx, 3])
        rot_psi[0, NS_idx] = torch.tensor([[1, 0, 0],
                                           [0, torch.cos(SEP_psi), -torch.sin(SEP_psi)],
                                           [0, torch.sin(SEP_psi), torch.cos(SEP_psi)], ], device=pos.device)

        rigid_rotation[0, NS_idx, constants.PSI_FRAME] = _make_rotation_matrix(
            ex=std_pos[2] - std_pos[1],
            ey=std_pos[1] - std_pos[0],
        )
        rigid_translation[0, NS_idx, constants.PSI_FRAME] = std_pos[2]
        R_psi, t_psi = compose_chain(
            [
                (R_bb, t_bb),
                # Rigid transform based on T_bb, thirdly
                (rigid_rotation[:, :, constants.PSI_FRAME], rigid_translation[:, :, constants.PSI_FRAME]),
                # Rigid transform based on ideal coordinate, secondly
                (rot_psi, torch.zeros_like(t_bb)),  # Rigid transform based on torsion angle, firstly
            ]  # PSI rotate along the direction of CA->C, so rigid_trans is the trans of C
        )

        SEP_X1 = _get_torsion(pos[0, NS_idx, 0], pos[0, NS_idx, 1],
                              pos[0, NS_idx, 4], pos[0, NS_idx, 5])  # Common value: +60 deg, +180 deg
        rot_chi1[0, NS_idx] = torch.tensor([[1, 0, 0],
                                            [0, torch.cos(SEP_X1), -torch.sin(SEP_X1)],
                                            [0, torch.sin(SEP_X1), torch.cos(SEP_X1)], ], device=pos.device)
        X1_base = 4
        rigid_rotation[0, NS_idx, constants.CHI1_FRAME] = _make_rotation_matrix(
            ex=std_pos[4] - std_pos[1],  # CB, CA
            ey=std_pos[0] - std_pos[1],  # N, CA
        )
        rigid_translation[0, NS_idx, constants.CHI1_FRAME] = std_pos[X1_base]
        R_chi1, t_chi1 = compose_chain(
            [
                (R_bb, t_bb),
                # Rigid transform based on T_bb, thirdly
                (rigid_rotation[:, :, constants.CHI1_FRAME], rigid_translation[:, :, constants.CHI1_FRAME]),
                # Rigid transform based on ideal coordinate, secondly
                (rot_chi1, torch.zeros_like(t_bb)),  # Rigid transform based on torsion angle, firstly
            ]  # PSI rotate along the direction of CA->C, so rigid_trans is the trans of C
        )

        SEP_X2 = _get_torsion(pos[0, NS_idx, 1], pos[0, NS_idx, 4],
                              pos[0, NS_idx, 5], pos[0, NS_idx, 6])
        rot_chi2[0, NS_idx] = torch.tensor([[1, 0, 0],
                                            [0, torch.cos(SEP_X2), -torch.sin(SEP_X2)],
                                            [0, torch.sin(SEP_X2), torch.cos(SEP_X2)], ], device=pos.device)
        X2_base = 5
        rigid_rotation[0, NS_idx, constants.CHI2_FRAME] = _make_rotation_matrix(
            ex=std_pos[X2_base],
            ey=torch.FloatTensor([-1., 0., 0.]).to(pos.device),
        )

        rigid_translation[0, NS_idx, constants.CHI2_FRAME] = std_pos[X2_base]  # based on OG

        R_chi2, t_chi2 = compose_chain(
            [
                (R_chi1, t_chi1),
                (rigid_rotation[:, :, constants.CHI2_FRAME], rigid_translation[:, :, constants.CHI2_FRAME]),
                (rot_chi2, torch.zeros_like(t_bb)),
            ]  # If LYS, CHI2 rotate along the direction of CB->CG, so rigid_trans is the trans of CG
        )

        # SEP_X3 = _get_torsion(pos[0, NS_idx, 4], pos[0, NS_idx, 5],
        #                       pos[0, NS_idx, 6], pos[0, NS_idx, 7])
        # SEP_X3 = _get_torsion(pos[0, NS_idx, 2], pos[0, NS_idx, 1],
        #                       pos[0, NS_idx, 0], pos[0, NS_idx, 8])  # MLE
        SEP_X3 = _get_torsion(pos[0, NS_idx, 8], pos[0, NS_idx, 11],
                              pos[0, NS_idx, 10], pos[0, NS_idx, 12])  # TYS
        rot_chi3[0, NS_idx] = torch.tensor([[1, 0, 0],
                                            [0, torch.cos(SEP_X3), -torch.sin(SEP_X3)],
                                            [0, torch.sin(SEP_X3), torch.cos(SEP_X3)], ], device=pos.device)
        X3_base = 10
        rigid_rotation[0, NS_idx, constants.CHI3_FRAME] = _make_rotation_matrix(
            ex=std_pos[X3_base],  # std_pos[6]
            ey=torch.FloatTensor([-1., 0., 0.]).to(pos.device),
        )
        rigid_translation[0, NS_idx, constants.CHI3_FRAME] = std_pos[X3_base]  # std_pos[6]

        R_chi3, t_chi3 = compose_chain(
            [
                (R_chi2, t_chi2),  # R_chi2, t_chi2
                (rigid_rotation[:, :, constants.CHI3_FRAME], rigid_translation[:, :, constants.CHI3_FRAME]),
                (rot_chi3, torch.zeros_like(t_bb)),
            ]  # If LYS, CHI2 rotate along the direction of CB->CG, so rigid_trans is the trans of CG
        )

        SEP_X4 = _get_torsion(pos[0, NS_idx, 11], pos[0, NS_idx, 10],
                              pos[0, NS_idx, 12], pos[0, NS_idx, 13])  # TYS
        rot_chi4[0, NS_idx] = torch.tensor([[1, 0, 0],
                                            [0, torch.cos(SEP_X4), -torch.sin(SEP_X4)],
                                            [0, torch.sin(SEP_X4), torch.cos(SEP_X4)], ], device=pos.device)
        X4_base = 12
        rigid_rotation[0, NS_idx, constants.CHI4_FRAME] = _make_rotation_matrix(
            ex=std_pos[X4_base],  # std_pos[6]
            ey=torch.FloatTensor([-1., 0., 0.]).to(pos.device),
        )
        rigid_translation[0, NS_idx, constants.CHI4_FRAME] = std_pos[X4_base]  # std_pos[6]

        R_chi4, t_chi4 = compose_chain(
            [
                (R_chi3, t_chi3),  # R_chi2, t_chi2
                (rigid_rotation[:, :, constants.CHI4_FRAME], rigid_translation[:, :, constants.CHI4_FRAME]),
                (rot_chi4, torch.zeros_like(t_bb)),
            ]  # If LYS, CHI2 rotate along the direction of CB->CG, so rigid_trans is the trans of CG
        )

        # pdb.set_trace()
        # If want to check the ideal coor., don't need to save a pdb and visualize. Just reconstruct.
        # batch['torsion_angle'][0, NS_idx] = torch.tensor([SEP_psi, SEP_X1, SEP_X2, SEP_X3, SEP_X4]).to(pos.device)
        # batch['aa'][0, NS_idx] = 22  # PTR
        # pos_ha, _, _ = full_atom_reconstruction(R_bb=R_bb, t_bb=t_bb, angles=batch['torsion_angle'],
        #                                         aa=batch['aa'])

        pos_ideal = (pos - t_bb[:, :, None]) @ R_bb
        pos_ideal = (pos - t_psi[:, :, None]) @ R_psi
        pos_ideal = (pos - t_chi1[:, :, None]) @ R_chi1
        pos_ideal = (pos - t_chi2[:, :, None]) @ R_chi2
        pos_ideal = (pos - t_chi3[:, :, None]) @ R_chi3
        pos_ideal = (pos - t_chi4[:, :, None]) @ R_chi4
        # pos_ideal = torch.matmul(R_psi.transpose(-1, -2)[:, :, None].repeat(1, 1, 15, 1, 1),
        #                          (pos - t_psi[:, :, None]).unsqueeze(-1)).squeeze(-1)
        tmp = (torch.tensor(PTR_data_PDB).to(pos.device) - t_chi4[0, NS_idx]) @ R_chi4[0, NS_idx]
        # sctrans_1_c = (sctrans_1 - trans_1_c[:, :, None, :]) @ rotmats_1  # (B, L, 9, 3), rel coor.
        # DEBUG--------

    traj = model.sample(batch, num_steps=num_steps, sample_bb=sample_structure,
                        sample_seq=sample_sequence, sample_ang=sample_ang)

    final = recursive_to(traj[-1], device=device)
    if final['angles'].shape[-1] == 1:
        final['angles'] = F.pad(final['angles'], (0, 4))

    pos_ha, _, _ = full_atom_reconstruction(R_bb=final['rotmats'], t_bb=final['trans'], angles=final['angles'],
                                            aa=final['seqs'])  # (32,L,14,3), instead of 15, ignore OXT masked
    # pos_ha[:, :, 5:] = final['sctrans']  # use side chain all-atom prediction
    pos_ha = F.pad(pos_ha, pad=(0, 0, 0, constants.max_num_heavyatoms - pos_ha.shape[2]), value=0.)  # (B,L,A,3) pos14 A=14
    pos_new = torch.where(batch['generate_mask'][:, :, None, None], pos_ha, batch['pos_heavyatom'])
    # mask_bb_atoms = torch.zeros_like(batch['mask_heavyatom'])
    # mask_bb_atoms[:, :, :4] = True
    # mask_new = torch.where(batch['generate_mask'][:, :, None], mask_bb_atoms, batch['mask_heavyatom'])
    mask_new = get_heavyatom_mask(final['seqs'])
    aa_new = final['seqs']

    chain_nb = torch.LongTensor([0 if gen_mask else 1 for gen_mask in data['generate_mask']])
    chain_id = ['A' if gen_mask else 'B' for gen_mask in data['generate_mask']]
    icode = [' ' for _ in range(len(data['icode']))]
    for i in range(nums):
        ref_bb_pos = data['pos_heavyatom'][:, :4].cpu()
        pred_bb_pos = pos_new[i][:, :4].cpu()
        data_saved = {
            'chain_nb': data['chain_nb'], 'chain_id': data['chain_id'], 'resseq': data['resseq'],
            'icode': data['icode'],
            'aa': aa_new[i].cpu(), 'mask_heavyatom': mask_new[i].cpu(), 'pos_heavyatom': pos_new[i].cpu(),
        }

        save_pdb(data_saved, path=os.path.join(save_root, f'sample_{i}.pdb'))


def save_samples_bb(samples, save_dir):
    # meta data
    batch = recursive_to(samples['batch'], 'cpu')
    chain_id = [list(item) for item in zip(*batch['chain_id'])][0]  # fix chain id in collate func
    icode = [' ' for _ in range(len(chain_id))]  # batch icode have same problem
    nums = len(batch['id'])
    id = batch['id'][0]
    # batch convert
    # aa=batch['aa] if only bb level
    pos_bb = reconstruct_backbone(R=samples['rotmats'], t=samples['trans'], aa=samples['seqs'],
                                  chain_nb=batch['chain_nb'], res_nb=batch['res_nb'],
                                  mask=batch['res_mask'])  # (32,L,4,3)
    pos_ha = F.pad(pos_bb, pad=(0, 0, 0, 15 - 4), value=0.)  # (32,L,A,3) pos14 A=14
    pos_new = torch.where(batch['generate_mask'][:, :, None, None], pos_ha, batch['pos_heavyatom'])
    mask_bb_atoms = torch.zeros_like(batch['mask_heavyatom'])
    mask_bb_atoms[:, :, :4] = True
    mask_new = torch.where(batch['generate_mask'][:, :, None], mask_bb_atoms, batch['mask_heavyatom'])
    aa_new = samples['seqs']
    for i in range(nums):
        data_saved = {
            'chain_nb': batch['chain_nb'][0], 'chain_id': chain_id, 'resseq': batch['resseq'][0], 'icode': icode,
            'aa': aa_new[i], 'mask_heavyatom': mask_new[i], 'pos_heavyatom': pos_new[i],
        }
        save_pdb(data_saved, path=os.path.join(save_dir, f'sample_{i}.pdb'))
    data_saved = {
        'chain_nb': batch['chain_nb'][0], 'chain_id': chain_id, 'resseq': batch['resseq'][0], 'icode': icode,
        'aa': batch['aa'][0], 'mask_heavyatom': batch['mask_heavyatom'][0], 'pos_heavyatom': batch['pos_heavyatom'][0],
    }
    save_pdb(data_saved, path=os.path.join(save_dir, f'gt.pdb'))


def save_samples_sc(samples, save_dir):
    # meta data
    batch = recursive_to(samples['batch'], 'cpu')
    chain_id = [list(item) for item in zip(*batch['chain_id'])][0]  # fix chain id in collate func
    icode = [' ' for _ in range(len(chain_id))]  # batch icode have same problem
    nums = len(batch['id'])
    id = batch['id'][0]
    # batch convert
    # aa=batch['aa] if only bb level
    pos_ha, _, _ = full_atom_reconstruction(R_bb=samples['rotmats'], t_bb=samples['trans'], angles=samples['angles'],
                                            aa=samples['seqs'])  # (32,L,14,3), instead of 15, ignore OXT masked
    pos_ha = F.pad(pos_ha, pad=(0, 0, 0, 15 - 14), value=0.)  # (32,L,A,3) pos14 A=14
    pos_new = torch.where(batch['generate_mask'][:, :, None, None], pos_ha, batch['pos_heavyatom'])
    mask_new = get_heavyatom_mask(samples['seqs'])
    aa_new = samples['seqs']
    for i in range(nums):
        data_saved = {
            'chain_nb': batch['chain_nb'][0], 'chain_id': chain_id, 'resseq': batch['resseq'][0], 'icode': icode,
            'aa': aa_new[i], 'mask_heavyatom': mask_new[i], 'pos_heavyatom': pos_new[i],
        }
        save_pdb(data_saved, path=os.path.join(save_dir, f'sample_{i}.pdb'))
    data_saved = {
        'chain_nb': batch['chain_nb'][0], 'chain_id': chain_id, 'resseq': batch['resseq'][0], 'icode': icode,
        'aa': batch['aa'][0], 'mask_heavyatom': batch['mask_heavyatom'][0], 'pos_heavyatom': batch['pos_heavyatom'][0],
    }
    save_pdb(data_saved, path=os.path.join(save_dir, f'gt.pdb'))


def sample(data_path, output_dir, num=64):
    data_load = torch.load(data_path)
    # output_dir = "/home/guot0b/disk3/guot0b/PepFlowww-main/output/2aq9_X/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    device = "cuda:0"

    model_ckpt = "/home/guot0b/disk3/guot0b/PepFlowww-main/ckpt/model1.pt"
    config, config_name = load_config('./configs/learn_angle.yaml')

    seed_all(config.train.seed)

    model = FlowModel(config.model).to(device)
    ckpt = torch.load(model_ckpt, map_location=device)
    state_dict = ckpt['model']
    new_state_dict = {}

    for key in state_dict:
        new_key = key.replace('module.', '') if key.startswith('module.') else key
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)

    sample_for_data_bb(data_load, model, device, output_dir, nums=num)


def sample_tao(data_path, output_dir, ckpt_path, num=64, device='cuda:1', tao_repr=False):
    data_load = torch.load(data_path)
    config, config_name = load_config('./configs/learn_angle_tao.yaml')
    if tao_repr:
        model = FlowModel_tao(config.model).to(device)
    else:
        model = FlowModel_NS3(config.model).to(device)

    seed_all(config.train.seed)

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt['model']
    new_state_dict = {}

    for key in state_dict:
        new_key = key.replace('module.', '') if key.startswith('module.') else key
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)

    sample_for_data_tao(data_load, model, device, output_dir, tao_repr=tao_repr, nums=num)


if __name__ == '__main__':
    # sample = torch.load('./Codesign/outputs/1aze_B.pt')
    # save_samples_sc(sample,'./misc/test')
    # save_samples_bb(sample,'./misc/test')
    # for k,v in sample.items():
    #     if isinstance(v,torch.Tensor):
    #         print(f'{k},{v.shape}')

    # # subdir = 'bb_seq_angle' # bb,bb_seq,bb_seq_angle
    # names = [n.split('.')[0] for n in os.listdir(os.path.join(SAMPLE_DIR,subdir,'outputs'))]
    # for name in tqdm(names):
    #     sample = torch.load(os.path.join(SAMPLE_DIR,subdir,'outputs',f'{name}.pt'))
    #     os.makedirs(os.path.join(SAMPLE_DIR,subdir,'pdbs',name),exist_ok=True)
    #     save_samples_sc(sample,os.path.join(SAMPLE_DIR,subdir,'pdbs',name))
    """
    args = argparse.ArgumentParser()
    args.add_argument('--SAMPLEDIR', type=str)
    parser = args.parse_args()
    SAMPLE_DIR = parser.SAMPLEDIR
    names = [n.split('.')[0] for n in os.listdir(os.path.join(SAMPLE_DIR, 'outputs'))]
    for name in tqdm(names):
        sample = torch.load(os.path.join(SAMPLE_DIR, 'outputs', f'{name}.pt'))
        os.makedirs(os.path.join(SAMPLE_DIR, 'pdbs', name), exist_ok=True)
        save_samples_sc(sample, os.path.join(SAMPLE_DIR, 'pdbs', name))
    """
    sample_tao("/home/guot0b/disk3/guot0b/PepFlowww-main/data/PepMerge_SEP/5emb_B/5emb_re.pt",
               "/home/guot0b/disk3/guot0b/PepFlowww-main/output/5emb_1125_120k/",
               "/home/guot0b/disk3/guot0b/PepFlowww-main/logs/2024.11.24 PepFlow_SEP 160k scratch/checkpoints/120000.pt",
               num=8)
