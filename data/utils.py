import pdb
from typing import List, Dict, Any
from openfold.utils import rigid_utils as ru
from data import residue_constants
import numpy as np
import collections
import string
import pickle
import os
import random
import torch
from torch_scatter import scatter_add, scatter
from Bio.PDB.Chain import Chain
from Bio.PDB import PDBParser, PDBIO, MMCIFParser, Chain, Residue, Atom
from Bio.SeqUtils import seq1
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from data import protein
import dataclasses
from Bio import PDB
import io
import pandas as pd
import requests
import pepflow.modules.protein.constants as constants

"""From https://github.com/microsoft/protein-frame-flow"""

Rigid = ru.Rigid
Protein = protein.Protein

# Global map from chain characters to integers.
ALPHANUMERIC = string.ascii_letters + string.digits + ' '
CHAIN_TO_INT = {
    chain_char: i for i, chain_char in enumerate(ALPHANUMERIC)
}
INT_TO_CHAIN = {
    i: chain_char for i, chain_char in enumerate(ALPHANUMERIC)
}

NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE

CHAIN_FEATS = [
    'atom_positions', 'aatype', 'atom_mask', 'residue_index', 'b_factors'
]

to_numpy = lambda x: x.detach().cpu().numpy()
aatype_to_seq = lambda aatype: ''.join([
    residue_constants.restypes_with_x[x] for x in aatype])


def create_rigid(rots, trans):
    rots = ru.Rotation(rot_mats=rots)
    return Rigid(rots=rots, trans=trans)


def batch_align_structures(pos_1, pos_2, mask=None):
    if pos_1.shape != pos_2.shape:
        raise ValueError('pos_1 and pos_2 must have the same shape.')
    if pos_1.ndim != 3:
        raise ValueError(f'Expected inputs to have shape [B, N, 3]')
    num_batch = pos_1.shape[0]
    device = pos_1.device
    batch_indices = (
            torch.ones(*pos_1.shape[:2], device=device, dtype=torch.int64)
            * torch.arange(num_batch, device=device)[:, None]
    )
    flat_pos_1 = pos_1.reshape(-1, 3)
    flat_pos_2 = pos_2.reshape(-1, 3)
    flat_batch_indices = batch_indices.reshape(-1)
    if mask is None:
        aligned_pos_1, aligned_pos_2, align_rots = align_structures(
            flat_pos_1, flat_batch_indices, flat_pos_2)
        aligned_pos_1 = aligned_pos_1.reshape(num_batch, -1, 3)
        aligned_pos_2 = aligned_pos_2.reshape(num_batch, -1, 3)
        return aligned_pos_1, aligned_pos_2, align_rots

    flat_mask = mask.reshape(-1).bool()
    _, _, align_rots = align_structures(
        flat_pos_1[flat_mask],
        flat_batch_indices[flat_mask],
        flat_pos_2[flat_mask]
    )
    aligned_pos_1 = torch.bmm(
        pos_1,
        align_rots
    )
    return aligned_pos_1, pos_2, align_rots


_natural_amino_acids = {
    "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
    "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR",
    # "ZN", "NH2", "NA", "UNK",
    "SEP", "PTR", "TYS", "MLE", "M3L", "ALY", "TPO", "DAL", "HYP", "MVA",
    "DLY", "DLE", "SAR", "NLE", "BMT", "DGL", "DPR", "DTR",  # NS18
}


def extract_pdb(pdb_file_path: str, check_unaa=False):
    flag = False
    rand = random.randint(1, 1000000)
    if pdb_file_path.endswith(".cif"):
        # Convert to .pdb
        parser = MMCIFParser()
        structure = parser.get_structure("TMP", pdb_file_path)
        io = PDBIO()
        io.set_structure(structure)
        tmp_path = os.path.join(os.path.dirname(pdb_file_path), f"tmp_{rand}.pdb")
        io.save(tmp_path)
        pdb_file_path = tmp_path
        flag = True

    parser = PDBParser()
    structure = parser.get_structure('PDB_Name', pdb_file_path)
    _sequence = {}
    ca_coordinates = []
    for chain in sorted(structure[0], key=lambda x: x.id):
        tmp_seq = []
        for residue in chain:
            res_name = residue.get_resname()
            tmp_seq.append(res_name)
            # if res_name not in _natural_amino_acids and check_unaa:
            #     # print(f"There is unnatural a.a. {res_name} in {pdb_file_path}")
            #     return None
            # else:
            #     tmp_seq.append(seq1(res_name))
            #     # seq1 will turn unnatural aa into 'X'

            if 'CA' in residue:
                ca_atom = residue['CA']
            else:
                atom = next(iter(residue))
                ca_atom = residue[atom.name]
            ca_coordinates.append(ca_atom.coord)
        # _sequence[chain.id] = ''.join(tmp_seq)
        _sequence[chain.id] = tmp_seq

    if flag:
        os.remove(tmp_path)
    return np.array(ca_coordinates), _sequence


def rec_aa(pdb_file_path: str, chain_id=None):
    parser = PDBParser()
    structure = parser.get_structure('PDB_Name', pdb_file_path)
    _sequence = {}
    nsaa = []
    aa = []
    for chain in sorted(structure[0], key=lambda x: x.id):
        if chain_id is not None:
            if chain.id != chain_id:
                continue
        for residue in chain:
            res_name = residue.get_resname()
            if res_name not in _natural_amino_acids:
                nsaa.append(res_name)
            else:
                aa.append(res_name)
    return aa, nsaa


def count_aa(src_dir: str, output_path: str, T_ex=False):
    """

    Args:
        src_dir:
        output_path:
        T_ex: Transform nsaa excluding ...
        (T_ex=False, don't transform the nsaa to standard aa;
        T_ex=True, transform all the nsaa to standard aa;)

    Returns:

    """
    src_con = os.listdir(src_dir)
    aa_dict = {}
    nsaa_sample_rec = {}
    rec_csv = pd.DataFrame(columns=['aatype', 'Num', 'Ratio(%)', 'Weight', 'Sample'])

    for item in src_con:
        item_dir = os.path.join(src_dir, item)
        try:
            aa_tmp, nsaa_tmp = rec_aa(os.path.join(item_dir, 'peptide.pdb'))
            for aatype in aa_tmp:
                if aatype not in ["ZN", "NH2", "NA"]:
                    if aatype in aa_dict.keys():
                        aa_dict[aatype] += 1
                    else:
                        aa_dict[aatype] = 1
            for aatype in nsaa_tmp:
                if T_ex is False:
                    aa_type_ = aatype + '_ns'
                elif T_ex is True:
                    aa_type_ = constants.non_standard_residue_substitutions.get(aatype, 'UNK')

                if aa_type_ in aa_dict.keys():
                    aa_dict[aa_type_] += 1
                else:
                    aa_dict[aa_type_] = 1

                if aa_type_ in nsaa_sample_rec.keys():
                    if item not in nsaa_sample_rec[aa_type_]:
                        nsaa_sample_rec[aa_type_] += f"{item}, "
                else:
                    nsaa_sample_rec[aa_type_] = f"{item}, "
        except:
            pass
    sorted_aa_dict = dict(sorted(aa_dict.items(), key=lambda item: item[1], reverse=True))
    total_aa = sum([vvv for _, vvv in sorted_aa_dict.items()])
    weight_dict = {}
    for item in sorted_aa_dict.keys():
        ratio = (sorted_aa_dict[item] / total_aa) * 100
        weight = round(10 / ratio, 3)
        temp_df = pd.DataFrame({
            'aatype': [item],
            'Num': [sorted_aa_dict[item]],
            'Ratio(%)': [ratio],
            'Weight': [weight],
            'Sample': [nsaa_sample_rec.get(item, '')],
        })
        weight_dict[f'{item}'] = weight
        rec_csv = pd.concat([rec_csv, temp_df])
    rec_csv.to_csv(output_path, index=False)


def renumber_chain(src_dir):
    # To renumber the chain of data in PepFlow (Lig: 'A')
    parser = PDBParser()
    io = PDBIO()
    src_con = os.listdir(src_dir)

    for item in src_con:
        if os.path.isdir(os.path.join(src_dir, item)):
            struc_lig = parser.get_structure("lig", os.path.join(src_dir, item, "peptide.pdb"))
            struc_poc = parser.get_structure("rec", os.path.join(src_dir, item, "pocket.pdb"))

            ligand_renumber = Structure('ligand_re')
            model_lig = Model(0)
            ligand_renumber.add(model_lig)

            for chain in struc_lig[0]:
                chain.id = 'A'
                model_lig.add(chain.copy())

            io.set_structure(ligand_renumber)
            io.save(os.path.join(src_dir, item, "peptide_re.pdb"))

            pocket_renumber = Structure('pocket_re')
            model_poc = Model(0)
            pocket_renumber.add(model_poc)

            for chain in struc_poc[0]:
                # chain.id = 'B'
                # model_poc.add(chain.copy())
                existing_chain_ids = {chain.id for chain in struc_poc[0]}
                temp_chain_id_iter = iter("XYZWVUTSRQPO")  # Temporary chain IDs
                # Temporarily rename each chain
                # for chain in struc_poc[0]:
                while True:
                    temp_chain_id = next(temp_chain_id_iter)
                    if temp_chain_id not in existing_chain_ids:
                        chain.id = temp_chain_id
                        existing_chain_ids.add(temp_chain_id)
                        break
            pocket_chains = set()  # To store the final assigned chain IDs
            new_chain_ids = iter("BCDEFGHIJKLMNOPQRSTUVWXYZ")
            for chain in struc_poc[0]:
                while True:
                    new_chain_id = next(new_chain_ids)
                    if new_chain_id not in pocket_chains:
                        chain.id = new_chain_id
                        pocket_chains.add(new_chain_id)
                        model_poc.add(chain.copy())
                        break

            io.set_structure(pocket_renumber)
            io.save(os.path.join(src_dir, item, "pocket_re.pdb"))


def fetch_pdb_resolution(pdb_id):
    # Define the URL for the REST API call
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"

    # Send a GET request to the API
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()

        # Check if resolution information exists
        if 'rcsb_entry_info' in data and 'resolution_combined' in data['rcsb_entry_info']:
            resolution = data['rcsb_entry_info']['resolution_combined']
            return resolution
        else:
            return "Resolution information not available for this entry."
    else:
        return f"Failed to retrieve data for PDB ID {pdb_id}. Status code: {response.status_code}"

def handle_existing_folder(target_folder):
    if os.path.exists(target_folder):
        print(f"The folder '{target_folder}' already exists. Suggest you checking the current dir.")
        # user_input = input("Do you want to overwrite it? (y/n): ").strip().lower()
        user_input = 'no'  # Don't rewrite in the ibex
        if user_input in ['yes', 'y']:
            print("Overwriting the folder...")
            shutil.rmtree(target_folder)  # Delete the existing folder
            os.makedirs(target_folder)  # Create a new, empty folder
        elif user_input in ['no', 'n']:
            print("Keeping the existing folder. No changes were made.")
        else:
            print("Invalid input. Please respond with 'yes' or 'no'.")
            handle_existing_folder(target_folder)  # Retry
    else:
        os.makedirs(target_folder)
        print(f"Created the folder '{target_folder}'.")


def kabsch_R(fixed: torch.Tensor, moving: torch.Tensor):
    flag = False
    if type(fixed) == np.ndarray and type(moving) == np.ndarray:
        flag = True
        fixed = torch.tensor(fixed)
        moving = torch.tensor(moving)

    fixed_center = fixed.mean(0)
    moving_center = moving.mean(0)

    fixed_motif = fixed - fixed_center
    moved_motif = moving - moving_center

    A = fixed_motif
    B = moved_motif
    C = torch.matmul(A.T, B)

    # compute optimal rotation matrix using SVD
    U, S, Vt = torch.linalg.svd(C)

    # ensure right-handed coordinate system
    d = torch.eye(3, device=U.device)
    d[-1, -1] = torch.sign(torch.linalg.det(Vt.T @ U.T))

    # Calculate rotation matrix
    R = Vt.T @ d @ U.T
    # moved = (moving - moving.mean(0)) @ Rot + fixed.mean(0)
    if flag:
        R = R.numpy()
    return R


def find_LCS_and_masks(A, B):
    # Create a 2D array to store lengths of longest common subsequence.
    n, m = len(A), len(B)
    L = [[0] * (m + 1) for _ in range(n + 1)]

    # Build the LCS dynamic programming table
    for i in range(n):
        for j in range(m):
            if A[i] == B[j]:
                L[i + 1][j + 1] = L[i][j] + 1
            else:
                L[i + 1][j + 1] = max(L[i + 1][j], L[i][j + 1])

    # Recover the LCS from the table
    LCS = []
    i, j = n, m
    while i > 0 and j > 0:
        if A[i - 1] == B[j - 1]:
            LCS.append(A[i - 1])
            i -= 1
            j -= 1
        elif L[i - 1][j] >= L[i][j - 1]:
            i -= 1
        else:
            j -= 1
    LCS.reverse()

    # Convert LCS list to string
    LCS = ''.join(LCS)

    # Generate masks for A and B based on LCS
    mask_A = ['0'] * n
    mask_B = ['0'] * m

    idx_A = idx_B = 0
    for char in LCS:
        # Find this character in A starting from idx_A
        while A[idx_A] != char:
            idx_A += 1
        mask_A[idx_A] = '1'
        idx_A += 1

        # Find this character in B starting from idx_B
        while B[idx_B] != char:
            idx_B += 1
        mask_B[idx_B] = '1'
        idx_B += 1
    a_str = ''.join(mask_A)
    b_str = ''.join(mask_B)
    mask_A = np.array([char == '1' for char in a_str])
    mask_B = np.array([char == '1' for char in b_str])
    return LCS, mask_A, mask_B


