"""pep-rec dataset"""
import os
import logging
import pdb
import shutil
import json
import joblib
import pickle
import lmdb
import numpy as np
from Bio import PDB
from Bio.PDB import PDBExceptions
from Bio.PDB import PDBParser, PDBIO, Select, Selection, is_aa
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.SeqUtils import seq1
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from pepflow.modules.protein.parsers import parse_pdb, get_residue_heavyatom4side_chain
from pepflow.modules.common.geometry import *
from pepflow.modules.protein.constants import *
from pepflow.utils.data import mask_select_data, find_longest_true_segment, PaddingCollate
from torch.utils.data import DataLoader

from omegaconf import OmegaConf
from easydict import EasyDict

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler, dist

from pepflow.utils.misc import load_config
from pepflow.utils.train import recursive_to

from models_con.torsion import get_torsion_angle

import torch

from pepflow.modules.protein.writers import save_pdb

# bind_dic = torch.load("/datapool/data2/home/jiahan/ResProj/PepDiff/frame-flow/misc/affinity_dict.pt")

# Data_split
# data_split_file = "./data/data_split_NS3.json"
data_split_file = "/ibex/user/guot0b/PepFlowww-NS3/data/PepMerge_lmdb/data_split_NoscPep.json"
if os.path.exists(data_split_file):  # cluster-only file; not needed for inference
    with open(data_split_file, 'r') as f:
        data_split = json.load(f)

# testset
names = []


# with open("/home/guot0b/disk3/guot0b/PepFlowww-main/data/PepMerge_lmdb/Fixed Data/names.txt", 'r') as f:
#     for line in f:
#         names.append(line.strip())

def preprocess_structure(task, tao_repr=False):
    try:
        if task['id'] in names:
            raise ValueError(f'{task["id"]} belongs to test set!')
        pdb_path = task['pdb_path']
        # pep
        # process peptide and find center of mass
        pep = parse_pdb(os.path.join(pdb_path, 'peptide.pdb'))[0]
        center = torch.sum(pep['pos_heavyatom'][pep['mask_heavyatom'][:, BBHeavyAtom.CA], BBHeavyAtom.CA], dim=0) / (
                torch.sum(pep['mask_heavyatom'][:, BBHeavyAtom.CA]) + 1e-8)
        pep['pos_heavyatom'] = pep['pos_heavyatom'] - center[None, None, :]
        pep['torsion_angle'], pep['torsion_angle_mask'] = get_torsion_angle(pep['pos_heavyatom'],
                                                                            pep['aa'])  # calc angles after translation
        if len(pep['aa']) < 3 or len(pep['aa']) > 25:
            raise ValueError('peptide length not in [3,25]')
        # Assign weight to each pepide w.r.t. their aa content
        weight = []
        for idx in pep['aa']:
            weight.append(WEIGHT_DICT.get(str(AA(idx.item())), 0))
        # rec
        rec = parse_pdb(os.path.join(pdb_path, 'pocket.pdb'))[0]
        rec['pos_heavyatom'] = rec['pos_heavyatom'] - center[None, None, :]
        rec['torsion_angle'], rec['torsion_angle_mask'] = get_torsion_angle(rec['pos_heavyatom'],
                                                                            rec['aa'])  # calc angles after translation
        rec['chain_nb'] += 1

        if tao_repr:  # Set the pos of the non-existent side chain atoms to their Ca atoms
            pep['mask_heavyatom4sc'] = get_residue_heavyatom4side_chain(pep['aa'])
            pep_pos_sc2Ca = torch.where(pep['mask_heavyatom4sc'][..., None].repeat(1, 1, 3), pep['pos_heavyatom'],
                                        pep['pos_heavyatom'][:, BBHeavyAtom.CA][:, None, :].repeat(1,
                                                                                                   max_num_heavyatoms,
                                                                                                   1))
            pep['pos_heavyatom'] = pep_pos_sc2Ca

            rec['mask_heavyatom4sc'] = get_residue_heavyatom4side_chain(rec['aa'])
            rec_pos_sc2Ca = torch.where(rec['mask_heavyatom4sc'][..., None].repeat(1, 1, 3), rec['pos_heavyatom'],
                                        rec['pos_heavyatom'][:, BBHeavyAtom.CA][:, None, :].repeat(1,
                                                                                                   max_num_heavyatoms,
                                                                                                   1))
            rec['pos_heavyatom'] = rec_pos_sc2Ca

        # meta data
        data = {}
        data['id'] = task['id']
        data['weight'] = torch.mean(torch.tensor(weight)).item()
        data['generate_mask'] = torch.cat([torch.zeros_like(rec['aa']), torch.ones_like(pep['aa'])], dim=0).bool()
        for k in rec.keys():
            if isinstance(rec[k], torch.Tensor):
                data[k] = torch.cat([rec[k], pep[k]], dim=0)
            elif isinstance(rec[k], list):
                data[k] = rec[k] + pep[k]
            else:
                raise ValueError(f'Unknown type of {rec[k]}')
        if tao_repr:
            # Loss weighted based on related distance
            L, A, _ = data['pos_heavyatom'].shape
            structure_flat = data['pos_heavyatom'].view(L * A, 3)  # Shape: [L*A, 3]
            atom_distances = torch.cdist(structure_flat, structure_flat, p=2)  # Shape: [L*A, L*A]
            mask = data['mask_heavyatom'].view(L * A)
            mask = mask[:, None] * mask[None, :]  # Shape: [L*A, L*A]
            atom_distances = torch.where(mask, atom_distances, 1e3)

            atom_distances = atom_distances.view(L, A, L, A)
            atom_distances = atom_distances[len(rec['aa']):, :, :len(rec['aa']), :]  # Interface distance
            atom_distances = torch.min(atom_distances, dim=3).values
            atom_distances = torch.min(atom_distances, dim=1).values
            min_interface_dist = torch.min(atom_distances, dim=1).values  # [len(pep)]
            loss_weight = torch.zeros_like(data['generate_mask'], dtype=torch.float32)
            loss_weight[data['generate_mask']] = (4.5 / min_interface_dist).float()
            data['loss_weight'] = loss_weight
            # DEBUG
            # atom_distances = atom_distances[len(rec['aa']):, :, :len(rec['aa']), :]
            # atom_distances = torch.min(atom_distances, dim=3).values
            # atom_distances = torch.min(atom_distances, dim=1).values
            # # atom_distances += torch.eye(len(atom_distances)) * 6
            # weight_pre = 4.5 / atom_distances
            # weight = torch.min(weight_pre, dim=-1).values
            # pdb.set_trace()
            # import matplotlib.pyplot as plt
            # matrix_np = loss_weight.numpy()
            # # Plot the heatmap
            # plt.figure(figsize=(10, 8))
            # plt.imshow(matrix_np, cmap='viridis', interpolation='nearest')
            # plt.colorbar(label='Value')
            # plt.title('Heatmap of Matrix')
            # plt.xlabel('Column Index')
            # plt.ylabel('Row Index')
            # plt.savefig('distogram_1a81_H_local.png')

            # DEBUG
        return data

    except (
            PDBExceptions.PDBConstructionException,
            KeyError,
            ValueError,
            TypeError
    ) as e:
        logging.warning('[{}] {}: {}'.format(
            task['id'],
            e.__class__.__name__,
            str(e)
        ))
        return None

def preprocess_structure_NoPoc(task, tao_repr=False):
    # Discard the pocket information
    try:
        if task['id'] in names:
            raise ValueError(f'{task["id"]} belongs to test set!')
        pdb_path = task['pdb_path']
        # pep
        # process peptide and find center of mass
        pep = parse_pdb(os.path.join(pdb_path, 'peptide.pdb'))[0]
        center = torch.sum(pep['pos_heavyatom'][pep['mask_heavyatom'][:, BBHeavyAtom.CA], BBHeavyAtom.CA], dim=0) / (
                torch.sum(pep['mask_heavyatom'][:, BBHeavyAtom.CA]) + 1e-8)
        pep['pos_heavyatom'] = pep['pos_heavyatom'] - center[None, None, :]
        pep['torsion_angle'], pep['torsion_angle_mask'] = get_torsion_angle(pep['pos_heavyatom'],
                                                                            pep['aa'])  # calc angles after translation
        if len(pep['aa']) < 3 or len(pep['aa']) > 25:
            raise ValueError('peptide length not in [3,25]')
        # Assign weight to each pepide w.r.t. their aa content
        weight = []
        for idx in pep['aa']:
            weight.append(WEIGHT_DICT.get(str(AA(idx.item())), 0))

        # rec
        rec = parse_pdb(os.path.join(pdb_path, 'pocket.pdb'))[0]
        rec['pos_heavyatom'] = rec['pos_heavyatom'] - center[None, None, :]
        rec['torsion_angle'], rec['torsion_angle_mask'] = get_torsion_angle(rec['pos_heavyatom'],
                                                                            rec['aa'])  # calc angles after translation
        rec['chain_nb'] += 1

        if tao_repr:  # Set the pos of the non-existent side chain atoms to their Ca atoms
            pep['mask_heavyatom4sc'] = get_residue_heavyatom4side_chain(pep['aa'])
            pep_pos_sc2Ca = torch.where(pep['mask_heavyatom4sc'][..., None].repeat(1, 1, 3), pep['pos_heavyatom'],
                                        pep['pos_heavyatom'][:, BBHeavyAtom.CA][:, None, :].repeat(1,
                                                                                                   max_num_heavyatoms,
                                                                                                   1))
            pep['pos_heavyatom'] = pep_pos_sc2Ca

            rec['mask_heavyatom4sc'] = get_residue_heavyatom4side_chain(rec['aa'])
            rec_pos_sc2Ca = torch.where(rec['mask_heavyatom4sc'][..., None].repeat(1, 1, 3), rec['pos_heavyatom'],
                                        rec['pos_heavyatom'][:, BBHeavyAtom.CA][:, None, :].repeat(1,
                                                                                                   max_num_heavyatoms,
                                                                                                   1))
            rec['pos_heavyatom'] = rec_pos_sc2Ca

        # meta data
        data = {}
        data['id'] = task['id']
        data['weight'] = torch.mean(torch.tensor(weight)).item()
        data['generate_mask'] = torch.ones_like(pep['aa']).bool()
        for k in pep.keys():
            if isinstance(pep[k], torch.Tensor):
                data[k] = pep[k]
            elif isinstance(pep[k], list):
                data[k] = pep[k]
            else:
                raise ValueError(f'Unknown type of {pep[k]}')
        


        if tao_repr:
            # Loss weighted based on related distance
            data['pos_heavyatom_'] = torch.cat([rec['pos_heavyatom'], pep['pos_heavyatom']], dim=0)
            data['mask_heavyatom_'] = torch.cat([rec['mask_heavyatom'], pep['mask_heavyatom']], dim=0)
            L, A, _ = data['pos_heavyatom_'].shape
            structure_flat = data['pos_heavyatom_'].view(L * A, 3)  # Shape: [L*A, 3]
            atom_distances = torch.cdist(structure_flat, structure_flat, p=2)  # Shape: [L*A, L*A]
            mask = data['mask_heavyatom_'].view(L * A)
            mask = mask[:, None] * mask[None, :]  # Shape: [L*A, L*A]
            atom_distances = torch.where(mask, atom_distances, 1e3)

            atom_distances = atom_distances.view(L, A, L, A)
            atom_distances = atom_distances[len(rec['aa']):, :, :len(rec['aa']), :]  # Interface distance
            atom_distances = torch.min(atom_distances, dim=3).values
            atom_distances = torch.min(atom_distances, dim=1).values
            min_interface_dist = torch.min(atom_distances, dim=1).values  # [len(pep)]
            loss_weight = torch.zeros_like(data['generate_mask'], dtype=torch.float32)
            loss_weight[data['generate_mask']] = (4.5 / min_interface_dist).float()
            data['loss_weight'] = loss_weight
            del data['pos_heavyatom_']
            del data['mask_heavyatom_']
        return data

    except (
            PDBExceptions.PDBConstructionException,
            KeyError,
            ValueError,
            TypeError
    ) as e:
        logging.warning('[{}] {}: {}'.format(
            task['id'],
            e.__class__.__name__,
            str(e)
        ))
        return None


def preprocess_structure_v2(case_dir, output_path, renumber_chain=False, NoPoc=False):
    try:
        # pep
        # process peptide and find center of mass
        if not renumber_chain:
            pep = parse_pdb(os.path.join(case_dir, 'peptide.pdb'))[0]
        else:
            pep = parse_pdb(os.path.join(case_dir, 'peptide_re.pdb'))[0]
        center = torch.sum(pep['pos_heavyatom'][pep['mask_heavyatom'][:, BBHeavyAtom.CA], BBHeavyAtom.CA], dim=0) / (
                torch.sum(pep['mask_heavyatom'][:, BBHeavyAtom.CA]) + 1e-8)
        pep['pos_heavyatom'] = pep['pos_heavyatom'] - center[None, None, :]
        pep['torsion_angle'], pep['torsion_angle_mask'] = get_torsion_angle(pep['pos_heavyatom'],
                                                                            pep['aa'])  # calc angles after translation
        if len(pep['aa']) < 3 or len(pep['aa']) > 25:
            raise ValueError('peptide length not in [3,25]')
        # poc
        if not renumber_chain:
            poc = parse_pdb(os.path.join(case_dir, 'pocket.pdb'))[0]
        else:
            poc = parse_pdb(os.path.join(case_dir, 'pocket_re.pdb'))[0]
        poc['pos_heavyatom'] = poc['pos_heavyatom'] - center[None, None, :]
        poc['torsion_angle'], poc['torsion_angle_mask'] = get_torsion_angle(poc['pos_heavyatom'],
                                                                            poc['aa'])  # calc angles after translation
        poc['chain_nb'] += 1
        # meta data
        data = {}
        data['id'] = os.path.basename(case_dir)
        if NoPoc: 
            data['generate_mask'] = torch.ones_like(pep['aa']).bool()
        else:
            data['generate_mask'] = torch.cat([torch.zeros_like(poc['aa']), torch.ones_like(pep['aa'])], dim=0).bool()
        for k in poc.keys():
            if isinstance(poc[k], torch.Tensor):
                if NoPoc:
                    data[k] = pep[k]
                else:
                    data[k] = torch.cat([poc[k], pep[k]], dim=0)
            elif isinstance(poc[k], list):
                if NoPoc:
                    data[k] = pep[k]
                else:
                    data[k] = poc[k] + pep[k]
            else:
                raise ValueError(f'Unknown type of {poc[k]}')
            
        if NoPoc:
            return data
        else:
            torch.save(data, output_path)

    except (
            PDBExceptions.PDBConstructionException,
            KeyError,
            ValueError,
            TypeError
    ) as e:
        logging.warning('[{}] {}: {}'.format(
            case_dir,
            e.__class__.__name__,
            str(e)
        ))
        return None


def preprocess_structure_v3(pdb_path: str):
    # Used for evaluation
    ret = {}
    parse_data = parse_pdb(pdb_path)[0]
    pep_mask = parse_data['chain_nb'] == 0
    ret['pep_seq'] = parse_data['aa'][pep_mask]
    ret['pep_pos'] = parse_data['pos_heavyatom'][pep_mask]
    keys = ['chain_id', 'chain_nb', 'resseq', 'icode', 'res_nb', 'mask_heavyatom']
    for key in keys:
        value = parse_data[key]        
        if isinstance(value, torch.Tensor):
            # Safe: mask a tensor with a boolean tensor
            ret[f'pep_{key}'] = value[pep_mask]
        elif isinstance(value, list):
            # Apply the boolean mask using zip
            ret[f'pep_{key}'] = [v for v, m in zip(value, pep_mask.tolist()) if m]
    ret['rec_seq'] = parse_data['aa'][~pep_mask]
    ret['rec_pos'] = parse_data['pos_heavyatom'][~pep_mask]
    ret['complex_seq'] = parse_data['aa']
    ret['complex_pos'] = parse_data['pos_heavyatom']
    ret['complex_mask'] = parse_data['mask_heavyatom']
    ret['pep_mask'] = pep_mask
    return ret


def calculate_ligand_com(pdb_file_path: str):
    """
    Calculates the Center of Mass (CoM) of the part of a ligand interacting
    with a receptor.

    The script assumes:
    - Ligand is in Chain A.
    - Receptor is in Chain B.
    - Interaction is defined as any ligand atom within 10 Å of any receptor atom.

    CoM Calculation Rules:
    - If the ligand is a protein/peptide, CoM is based on C-alpha atoms.
    - If the ligand is a small molecule, CoM is based on all heavy (non-hydrogen) atoms.

    Args:
        pdb_file_path (str): The full path to the input .pdb file.

    Returns:
        np.ndarray | None: A NumPy array containing the (x, y, z) coordinates of the CoM,
                        or None if the calculation cannot be completed.
    """

    ATOMIC_MASS = {
    'H': 1.008, 'D': 2.014, 'C': 12.011, 'N': 14.007, 'O': 15.999,
    'P': 30.974, 'S': 32.06, 'F': 18.998, 'CL': 35.45, 'BR': 79.90,
    'I': 126.90, 'NA': 22.990, 'MG': 24.305, 'K': 39.098, 'CA': 40.078,
    'MN': 54.938, 'FE': 55.845, 'CO': 58.933, 'NI': 58.693, 'CU': 63.546,
    'ZN': 65.38
    }

    if not os.path.exists(pdb_file_path):
        print(f"Error: PDB file not found at '{pdb_file_path}'")
        return None

    parser = PDBParser()
    try:
        structure = parser.get_structure('complex', pdb_file_path)
    except Exception as e:
        print(f"Error parsing PDB file: {e}")
        return None

    model = structure[0]

    # 1. Separate Ligand (Chain A) and Receptor (Chain B) atoms
    if 'A' not in model or 'B' not in model:
        print("Error: Required chains 'A' (ligand) and/or 'B' (receptor) not found.")
        return None

    ligand_atoms = list(model['A'].get_atoms())
    receptor_atoms = list(model['B'].get_atoms())

    if not ligand_atoms or not receptor_atoms:
        print("Error: Chain 'A' or 'B' contains no atoms.")
        return None

    # 2. Determine if the ligand is residue-based (protein/peptide) or a small molecule
    ligand_residues = list(model['A'].get_residues())
    # Heuristic: If more than 80% of residues are standard amino acids, it's a protein/peptide.
    aa_count = sum(1 for res in ligand_residues if is_aa(res, standard=True))
    is_residue_based = (aa_count / len(ligand_residues)) > 0.8 if ligand_residues else False

    # 3. Find the part of the ligand within 10 angstroms of the receptor
    neighbor_search = NeighborSearch(receptor_atoms)
    interacting_ligand_atoms = [
        atom for atom in ligand_atoms
        if neighbor_search.search(atom.get_coord(), 10.0, 'A')
    ]

    if not interacting_ligand_atoms:
        print("No part of the ligand found within 10 Å of the receptor.")
        return None
    
    # print(f"Found {len(interacting_ligand_atoms)} ligand atoms within 10 Å of the receptor.")

    # 4. Select atoms for CoM calculation from the interacting part
    com_atoms = []
    if is_residue_based:
        # print("Ligand identified as protein/peptide. Using C-alpha atoms for CoM.")
        com_atoms = [atom for atom in interacting_ligand_atoms if atom.get_id() == 'CA']
    else:
        # print("Ligand identified as small molecule. Using heavy atoms for CoM.")
        com_atoms = [atom for atom in interacting_ligand_atoms if atom.element != 'H']

    if not com_atoms:
        print("No suitable atoms (C-alpha or heavy atoms) found in the interacting part for CoM calculation.")
        return None

    # print(f"Calculating CoM using {len(com_atoms)} atoms.")

    # 5. Calculate the Center of Mass
    total_mass = 0.0
    mass_weighted_coords = np.zeros(3)

    for atom in com_atoms:
        # Get mass, defaulting to Carbon's mass if element is unknown
        mass = ATOMIC_MASS.get(atom.element.strip().upper(), 12.011)
        total_mass += mass
        mass_weighted_coords += atom.get_coord() * mass

    if total_mass == 0:
        print("Error: Total mass of selected atoms is zero. Cannot calculate CoM.")
        return None

    center_of_mass = mass_weighted_coords / total_mass
    return center_of_mass
    

def preprocess_structure_v4(case_dir: int, pep_length: int, CoM='ligand'):
    # For length-specified generation. Assume that the pocket chains have been renumbered.
    try:
        poc = parse_pdb(os.path.join(case_dir, 'pocket.pdb'))[0]
        if CoM == 'ligand':
            center = calculate_ligand_com(os.path.join(case_dir, 'complex.pdb'))
            center = torch.tensor(center, dtype=poc['pos_heavyatom'].dtype)
        elif CoM == 'receptor':   
            center = torch.sum(poc['pos_heavyatom'][poc['mask_heavyatom'][:, BBHeavyAtom.CA], BBHeavyAtom.CA], dim=0) / (
                    torch.sum(poc['mask_heavyatom'][:, BBHeavyAtom.CA]) + 1e-8)        

        poc['pos_heavyatom'] = poc['pos_heavyatom'] - center[None, None, :]
        poc['torsion_angle'], poc['torsion_angle_mask'] = get_torsion_angle(poc['pos_heavyatom'], poc['aa'])  # calc angles after translation
        poc['chain_nb'] += 1

        pep = {
            'chain_id': ['A' for _ in range(pep_length)],
            'chain_nb': torch.tensor([0 for _ in range(pep_length)]),
            'resseq': torch.arange(pep_length)+1,
            'icode': [' ' for _ in range(pep_length)],
            'res_nb': torch.arange(pep_length)+1,
            'aa': torch.tensor([0 for _ in range(pep_length)]),
            'pos_heavyatom': torch.full((pep_length, max_num_heavyatoms, 3), fill_value=0.),
            'mask_heavyatom': torch.full((pep_length, max_num_heavyatoms), fill_value=True),
            'torsion_angle': torch.full((pep_length, len(chi_angles_mask[AA.ALA])+1), fill_value=0.),
            'torsion_angle_mask': torch.full((pep_length, len(chi_angles_mask[AA.ALA])+1), fill_value=True),
        }

        # meta data
        data = {}
        data['id'] = os.path.basename(case_dir)
        data['generate_mask'] = torch.cat([torch.zeros_like(poc['aa']), torch.ones_like(pep['aa'])], dim=0).bool()
        for k in poc.keys():
            if isinstance(poc[k], torch.Tensor):
                data[k] = torch.cat([poc[k], pep[k]], dim=0)
            elif isinstance(poc[k], list):
                data[k] = poc[k] + pep[k]
            else:
                raise ValueError(f'Unknown type of {poc[k]}')

        return data

    except (
            PDBExceptions.PDBConstructionException,
            KeyError,
            ValueError,
            TypeError
    ) as e:
        logging.warning('[{}] {}: {}'.format(
            case_dir,
            e.__class__.__name__,
            str(e)
        ))
        return None

class PepDataset(Dataset):
    MAP_SIZE = 32 * (1024 * 1024 * 1024)  # 32GB

    def __init__(self, structure_dir="./Data/PepMerge_new/", dataset_dir="./Data/",
                 name='pep', transform=None, reset=False, tao_repr=False, esm=False,
                 split='./data/data_split_NS3.json'):

        super().__init__()
        self.structure_dir = structure_dir
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.name = name
        self.tao = tao_repr
        self.esm = esm

        self.db_conn = None
        self.db_ids = None

        assert split.endswith(".json"), "The file of data split should be JSON!"
        with open(split, 'r') as file:
            self.data_split = json.load(file)
        self._load_structures(reset)

    @property
    def _cache_db_path(self):
        if not self.tao:
            return os.path.join(self.dataset_dir, f'{self.name}_structure_cache.lmdb')
        elif self.tao and not self.esm:
            return os.path.join(self.dataset_dir, f'tao_repr_{self.name}_structure_cache.lmdb')
        # elif self.esm:
        #     return os.path.join(self.dataset_dir, f'esm_{self.name}_structure_cache.lmdb')

    def _connect_db(self):
        self._close_db()
        self.db_conn = lmdb.open(
            self._cache_db_path,
            map_size=self.MAP_SIZE,
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db_conn.begin() as txn:
            keys = [k.decode() for k in txn.cursor().iternext(values=False)]
            self.db_ids = keys

    def _close_db(self):
        if self.db_conn is not None:
            self.db_conn.close()
        self.db_conn = None
        self.db_ids = None

    def _load_structures(self, reset):
        mode = self.name.split('_')[-1]  # TRAIN, VAL
        tmp_all_pdbs = self.data_split[mode]
        # tmp_all_pdbs = os.listdir(self.structure_dir)
        all_pdbs = [item for item in tmp_all_pdbs
                    if os.path.isdir(os.path.join(self.structure_dir, item))]

        if reset:
            if os.path.exists(self.dataset_dir):
                shutil.rmtree(self.dataset_dir)
            os.mkdir(self.dataset_dir)
            if os.path.exists(self._cache_db_path):
                os.remove(self._cache_db_path)
                lock_file = self._cache_db_path + "-lock"
                if os.path.exists(lock_file):
                    os.remove(lock_file)
            self._close_db()
            todo_pdbs = all_pdbs
        else:
            if not os.path.exists(self._cache_db_path):
                todo_pdbs = all_pdbs
            else:
                todo_pdbs = []
                # self._connect_db()
                # processed_pdbs = self.db_ids
                # self._close_db()
                # todo_pdbs = list(set(all_pdbs) - set(processed_pdbs))

        if len(todo_pdbs) > 0:
            self._preprocess_structures(todo_pdbs)

    def _preprocess_structures(self, pdb_list):
        tasks = []
        for pdb_fname in pdb_list:
            pdb_path = os.path.join(self.structure_dir, pdb_fname)
            tasks.append({
                'id': pdb_fname,
                'pdb_path': pdb_path,
            })
        n_jobs = max(joblib.cpu_count(), 1)  # max(joblib.cpu_count() // 2, 1), 1
        n_jobs = 1
        data_list = joblib.Parallel(
            n_jobs=n_jobs,
        )(
            joblib.delayed(preprocess_structure)(task, tao_repr=self.tao)
            for task in tqdm(tasks, dynamic_ncols=True, desc='Preprocess')
        )
        if not os.path.exists(self.dataset_dir):
            os.mkdir(self.dataset_dir)
        db_conn = lmdb.open(
            self._cache_db_path,
            map_size=self.MAP_SIZE,
            create=True,
            subdir=False,
            readonly=False,
        )
        ids = []
        with db_conn.begin(write=True, buffers=True) as txn:
            for data in tqdm(data_list, dynamic_ncols=True, desc='Write to LMDB'):
                if data is None:
                    continue
                ids.append(data['id'])
                txn.put(data['id'].encode('utf-8'), pickle.dumps(data))

    def __len__(self):
        self._connect_db()  # make sure db_ids is not None
        return len(self.db_ids)

    def __getitem__(self, index):
        self._connect_db()
        id = self.db_ids[index]
        with self.db_conn.begin() as txn:
            data = pickle.loads(txn.get(id.encode()))
        if self.transform is not None:
            data = self.transform(data)
        return data


if __name__ == '__main__':
    device = 'cuda:1'
    config, cfg_name = load_config("./configs/learn/learn_all.yaml")
    dataset = PepDataset(structure_dir="./Data/PepMerge_new/", dataset_dir="/Data/Fixed Data",
                         name='pep_pocket_test', transform=None, reset=True)
    print(len(dataset))
    print(dataset[0])

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=PaddingCollate(eight=False))

    batch = next(iter(dataloader))
    print(batch['torsion_angle'].shape)
    print(batch['torsion_angle_mask'].shape)