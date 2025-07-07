import os
import argparse
import pdb
import shutil
import numpy as np
import pandas as pd
import random
import json
import torch
from Bio import PDB
from Bio.PDB.Chain import Chain
from Bio.PDB import PDBParser, PDBIO, MMCIFParser
from Bio.SeqUtils import seq1
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from pepflow.utils.misc import load_config, seed_all
from models_con.pep_dataloader import preprocess_structure_v2, preprocess_structure_v3
from data.utils import extract_pdb, handle_existing_folder, kabsch_R, renumber_pdb_residues
from pepflow.modules.protein.writers import save_pdb
from eval.geometry import *
from eval.energy import *
from eval.attnpacker.project_sidechains import atom14_to_atom37, compute_clashes

def merge_pdb(pdb_file1, pdb_file2, output_file, set_chain_ID=False, specify_ID=None):
    # Initialize parser and IO objects
    parser = PDBParser(QUIET=True)
    io = PDBIO()

    # Parse the two PDB files
    structure1 = parser.get_structure("Structure1", pdb_file1)
    structure2 = parser.get_structure("Structure2", pdb_file2)

    # Create a new structure and model
    merged_structure = Structure("Merged")
    merged_model = Model(0)  # Model ID is set to 0

    # Add chains from the first structure to the new model
    for chain in structure1[0]:  # Assuming the first model in each structure
        if set_chain_ID:
            chain.id = 'A'
        if specify_ID is None:
            merged_model.add(chain)
        else:
            if chain.id == specify_ID:
                merged_model.add(chain)

    # Add chains from the second structure to the new model
    if set_chain_ID or specify_ID:
        for chain in structure2[0]:
            # chain.id = 'B'
            # model_poc.add(chain.copy())
            existing_chain_ids = {chain.id for chain in structure2[0]}
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
        for chain in structure2[0]:
            while True:
                new_chain_id = next(new_chain_ids)
                if new_chain_id not in pocket_chains:
                    chain.id = new_chain_id
                    pocket_chains.add(new_chain_id)
                    merged_model.add(chain.copy())
                    break
    else:
        for chain in structure2[0]:  # Assuming the first model in each structure
            merged_model.add(chain)

    # Add the model to the merged structure
    merged_structure.add(merged_model)

    # Save the merged structure to the output file
    io.set_structure(merged_structure)
    io.save(output_file)


def find_LCS_and_masks(A, B):
    n, m = len(A), len(B)
    L = [[0] * (m + 1) for _ in range(n + 1)]

    length = 0
    end_pos_A = 0

    # Build the table for longest common substring
    for i in range(n):
        for j in range(m):
            if A[i] == B[j]:
                L[i + 1][j + 1] = L[i][j] + 1
                if L[i + 1][j + 1] > length:
                    length = L[i + 1][j + 1]
                    end_pos_A = i + 1
            else:
                L[i + 1][j + 1] = 0

    # Extract the longest common substring
    LCS = A[end_pos_A - length:end_pos_A]

    # Find positions and create masks
    mask_A = [False] * n
    mask_B = [False] * m

    # Mark positions in A
    if length > 0:
        for i in range(n - length + 1):
            if A[i:i + length] == LCS:
                mask_A[i:i + length] = [True] * length
                break

        # Mark positions in B
        for j in range(m - length + 1):
            if B[j:j + length] == LCS:
                mask_B[j:j + length] = [True] * length
                break

    return LCS, np.array(mask_A), np.array(mask_B)


# For recovery and diversity evaluation
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default="/ibex/user/guot0b/DataPool/Pepflow/")
    parser.add_argument('--test_list', type=str,
                        default="/ibex/user/guot0b/PepFlowww-NS3/data/data_split_NS3_v3.json")
    parser.add_argument('--output_dir', type=str, default="/home/guot0b/disk3/guot0b/PepFlowww-1124/eval_output/")
    parser.add_argument('--output_name', type=str, default="eval_recovery.csv")
    parser.add_argument('--num', type=int, default=16, help="Number of samples to generate per case")
    parser.add_argument('--center', action='store_true', default=False,
                        help="Whether to move to center, as the gt does")
    parser.add_argument('--energy', action='store_true', default=False,
                    help="Whether to calculate the energy for SUCCESSFUL cases")

    args = parser.parse_args()

    handle_existing_folder(args.output_dir)

    # 1. Prepare the test data and begin to generate
    if args.test_list.endswith('.txt'):
        # Test set in pepflow
        test_item = []
        with open(args.test_list, 'r') as f:
            for line in f:
                test_item.append(line.strip())
    elif args.test_list.endswith('.json'):
        with open(args.test_list, 'r') as f:
            data_split = json.load(f)
        test_item = data_split['TEST']

    eval_metrics = ['Case', 'rec_pos', 'rec_seq', 'rec_seq(sub)', 'rec_ss', 'rec_bs', 'diversity', 'Clash',
                    'rec_seq(SEP)', 'rec_seq(TYS)', 'rec_seq(PTR)', 'rec_seq(NS3)', 'Success', 'AFF(suc)']
    rec_seq_list_per_case = []
    rec_seq_sub_list_per_case = []
    rec_pos_list_per_case = []
    rec_ss_list_per_case = []  # Secondary structure recovery
    rec_bs_list_per_case = []  # Binding site recovery
    div_list_per_case = []  # diversity
    num_clash_list_per_case = []

    NS3_idx = torch.tensor([20, 21, 22])
    SEP_idx = torch.tensor([20])
    TYS_idx = torch.tensor([21])
    PTR_idx = torch.tensor([22])

    data_csv = pd.DataFrame(columns=eval_metrics)
    result_path = os.path.join(args.output_dir, args.output_name)

    for item in test_item:
        # item = '3ade_B'
        print(f"Evaluating PDB_ID {item} now...")
        item_folder = os.path.join(args.data_dir, item)
        assert os.path.isdir(item_folder), f"The {item_folder} is not a folder!"
        folder_con = os.listdir(item_folder)

        # If there isn't .pt file, make it
        pt_file = os.path.join(item_folder, f"{item}_re_NS3.pt")
        # pt_file = os.path.join(item_folder, f"{item}_re.pt")
        pt_con = torch.load(pt_file)
        gt_seq = pt_con['aa'][torch.where(pt_con['generate_mask'])]
        gt_seq_rec = pt_con['aa'][torch.where(~pt_con['generate_mask'])]
        gt_pos = pt_con['pos_heavyatom'][torch.where(pt_con['generate_mask'])]  # including OXT
        gt_pos_rec = pt_con['pos_heavyatom'][torch.where(~pt_con['generate_mask'])]

        # A substituted version of ground truth
        gt_seq_sub = torch.where(gt_seq == 20, torch.tensor(15), gt_seq)
        gt_seq_sub = torch.where(gt_seq_sub == 21, torch.tensor(19), gt_seq_sub)
        gt_seq_sub = torch.where(gt_seq_sub == 22, torch.tensor(19), gt_seq_sub)
        
        if args.center:
            pep_center = torch.load(os.path.join(item_folder, f"{item}_re.pt"))['center']
            Rot = None

        # Calculate the seq rec only for NS3
        # gt_NS3_mask = torch.isin(gt_seq, NS3_idx)
        gt_SEP_mask = torch.isin(gt_seq, SEP_idx)
        gt_SEP_num = gt_SEP_mask.sum().item() * args.num
        gt_TYS_mask = torch.isin(gt_seq, TYS_idx)
        gt_TYS_num = gt_TYS_mask.sum().item() * args.num
        gt_PTR_mask = torch.isin(gt_seq, PTR_idx)
        gt_PTR_num = gt_PTR_mask.sum().item() * args.num

        pt_to_pdb = os.path.join(item_folder, "pocket_merge_from_pt_NS3.pdb")
        # pt_to_pdb = os.path.join(item_folder, "pocket_merge_from_pt.pdb")
        # This pdb are in the same setting as the generated one.
        # It makes sure that the lengths of gt and gen are identical.

        # Extract the traj for ss and binding site calculation
        # gt_pep_file = os.path.join(item_folder, f"peptide_re.pdb")
        # gt_complex_file = os.path.join(item_folder, f"pocket_merge.pdb")
        gt_traj = get_traj_chain(pt_to_pdb, "A")
        item_output_dir = os.path.join(args.output_dir, item)
        try:
            sample_list = os.listdir(item_output_dir)
        except FileNotFoundError:
            continue
        rec_seq_list_per_gen_item = []
        rec_seq_sub_list_per_gen_item = []
        rec_pos_list_per_gen_item = []
        rec_ss_list_per_gen_item = []
        rec_bs_list_per_gen_item = []
        num_clash_per_gen_item = []

        Ca_pos_list_per_case = []  # For diversity
        Success_num = 0
        AFF_num = 0  # Better Affinity than the gt when success
        SEP_rec = 0
        TYS_rec = 0
        PTR_rec = 0
        skip_flag = False
        for gen_item in sample_list:
            if not gen_item.endswith('.pdb'):
                continue
            # gen_ca, gen_seq = extract_pdb(os.path.join(item_output_dir, gen_item))
            item_path = os.path.join(item_output_dir, gen_item)
            gen_con = preprocess_structure_v3(item_path)
            gen_pos, gen_seq = gen_con['pep_pos'], gen_con['pep_seq']
            gen_pos_rec, gen_seq_rec = gen_con['rec_pos'], gen_con['rec_seq']
            Ca_pos_list_per_case.append(gen_pos[:, 1])
            # # 2.1 Seq recovery
            if len(gt_seq) != len(gen_seq):
                skip_flag = True
                break

            rec_seq = (gt_seq == gen_seq).sum() / len(gen_seq)
            rec_seq_list_per_gen_item.append(rec_seq.item())

            rec_seq_sub = (gt_seq_sub == gen_seq).sum() / len(gt_seq_sub)
            rec_seq_sub_list_per_gen_item.append(rec_seq_sub.item())            

            # # 2.1.1 Seq recovery for NS3
            # Calculate the seq rec only for NS3
            # gen_NS3_mask = torch.isin(gen_seq, NS3_idx)
            gen_SEP_mask = torch.isin(gen_seq, SEP_idx) & gt_SEP_mask
            gen_TYS_mask = torch.isin(gen_seq, TYS_idx) & gt_TYS_mask
            gen_PTR_mask = torch.isin(gen_seq, PTR_idx) & gt_PTR_mask

            SEP_rec += gen_SEP_mask.sum().item()
            TYS_rec += gen_TYS_mask.sum().item()
            PTR_rec += gen_PTR_mask.sum().item()

            # # 2.2 Pos recovery w.r.t. CA (Don't need to align)
            # Rot = kabsch_R(gt_pos_rec[:, 1], gen_pos_rec[:, 1])
            # gen_pos_aligned = (gen_pos - gen_pos_rec[:, 1].mean(0)) @ Rot + gt_pos_rec[:, 1].mean(0)

            if args.center:
                if 'PPFlow' in args.output_dir or 'DPB' in args.output_dir or 'PepGLAD' in args.output_dir or 'UniMoMo' in args.output_dir:
                    gen_pos -= pep_center[None, None, :]
                elif 'RFD' in args.output_dir or 'ProteinGenerator' in args.output_dir:
                    if Rot is None:
                        Rot = kabsch_R(gt_pos_rec[:, 1], gen_pos_rec[:, 1])
                    gen_pos = (gen_pos - gen_pos_rec[:, 1].mean(0)) @ Rot + gt_pos_rec[:, 1].mean(0)
                    # gen_pos = gen_pos - gen_pos_rec[:, 1].mean(0)[None, None, :] + gt_pos_rec[:, 1].mean(0)[None, None, :]
                    # gen_rec_aligned = (gen_pos_rec - gen_pos_rec[:, 1].mean(0)) @ Rot + gt_pos_rec[:, 1].mean(0)
                elif 'af3' in args.output_dir or 'chai' in args.output_dir or 'esm' in args.output_dir:                   
                    gen_seq_rec_list = [str(x) for x in gen_seq_rec.tolist()]
                    gt_seq_rec_list = [str(x) for x in gt_seq_rec.tolist()]
                    LCS, gen_rec_mask, gt_rec_mask = find_LCS_and_masks(gen_seq_rec_list, gt_seq_rec_list)
                    Rot = kabsch_R(gt_pos_rec[gt_rec_mask, 1], gen_pos_rec[gen_rec_mask, 1])
                    # gen_pos = (gen_pos - gen_pos_rec[:, 1].mean(0)) @ Rot + gt_pos_rec[:, 1].mean(0)
                    gen_pos = (gen_pos - gen_pos_rec[gen_rec_mask, 1].mean(0)) @ Rot + gt_pos_rec[gt_rec_mask, 1].mean(0)                    
                    
            binder_rmsd = torch.linalg.norm(gen_pos[:, 1] - gt_pos[:, 1], dim=-1).mean()
            if binder_rmsd.item() < 2.:  # 2.
                Success_num += 1
                if args.energy:
                    gt_energy_dict = torch.load(os.path.join(item_folder, "energy_pocket_merge_NS3.pt"), 
                                                weights_only=False)
                    gt_aff = round(gt_energy_dict['aff'], 5)
                    if 'af3' in args.output_dir or 'esm' in args.output_dir:
                        data_saved = {
                            'chain_nb': gen_con['pep_chain_nb'], 'chain_id': gen_con['pep_chain_id'],
                            'resseq': gen_con['pep_resseq'], 'icode': gen_con['pep_icode'],
                            'aa': gen_seq, 'mask_heavyatom': gen_con['pep_mask_heavyatom'],
                            'pos_heavyatom': gen_pos,
                        }
                        tmp_pep_path = f"tmp/tmp_pep_{random.randint(1, 1e8)}.pdb"
                        save_pdb(data_saved, path=tmp_pep_path)
                        tmp_com_path = f"tmp/tmp_com_{random.randint(1, 1e8)}.pdb"
                        pocket_file = os.path.join(item_folder, "pocket.pdb")
                        merge_pdb(tmp_pep_path, pocket_file, tmp_com_path, specify_ID='A')                    
                        gen_energy_dict = get_rosetta_score_base(tmp_com_path, chain_id='A')
                        os.remove(tmp_pep_path)
                        os.remove(tmp_com_path)
                    else:                        
                        gen_energy_dict = get_rosetta_score_base(item_path, chain_id='A')
                    gen_aff = round(gen_energy_dict['aff'], 5)
                    AFF_num += 1 if gen_aff < gt_aff else 0                                         
            # DEBUG
            # chain_gt = get_chain_from_pdb(pt_to_pdb)
            # chain_gen = get_chain_from_pdb(item_path)
            # rmsd_rec, rmsd_lig = get_rmsd(chain_gt, chain_gen)
            
            # rec_pos_list_per_gen_item.append(rmsd_lig)
            # DEBUG
            rec_pos_list_per_gen_item.append(binder_rmsd.item())

            # # 2.3 Secondary structure recovery
            gen_traj = get_traj_chain(item_path, "A")
            try:
                rec_ss = get_ss(gt_traj, gen_traj)
            except ValueError:
                rec_ss = -1
            if rec_ss != -1:
                rec_ss_list_per_gen_item.append(rec_ss.item())

            # # 2.4 Binding site recovery
            if 'DPB' in args.output_dir:
                pocket_file = os.path.join(item_folder, "pocket.pdb")
                item_path_tmp = "tmp.pdb"
                merge_pdb(item_path, pocket_file, item_path_tmp, specify_ID='A')
                item_path = item_path_tmp
            if any(name in args.output_dir for name in ['af3', 'chai', 'esm']):
                pt_to_pdb = os.path.join(item_folder, "receptor_merge_re_renum.pdb")
            if 'PepGLAD' in args.output_dir or 'UniMoMo' in args.output_dir:
                renumber_pdb_residues(item_path, item_path)
                pt_to_pdb = os.path.join(item_folder, "pocket_merge_re_renum.pdb")
            rec_bs = get_bind_ratio(item_path, pt_to_pdb, "A", "A")
            if 'DPB' in args.output_dir:
                os.remove(item_path_tmp)
            # use 'pocket_merge.pdb' instead of 'receptor_merge.pdb'. Because the pocket is in 10 A distance.
            rec_bs_list_per_gen_item.append(rec_bs)

            # # 2.6 Clash
            num_clash = float(100)
            num_clash_per_gen_item.append(num_clash)
        if skip_flag:
            continue
        rec_seq_per_case = round(torch.tensor(rec_seq_list_per_gen_item).mean().item(), 5)
        rec_seq_sub_per_case = round(torch.tensor(rec_seq_sub_list_per_gen_item).mean().item(), 5)
        rec_pos_per_case = round(torch.tensor(rec_pos_list_per_gen_item).mean().item(), 5)
        if len(rec_ss_list_per_gen_item) == args.num:
            rec_ss_per_case = round(torch.tensor(rec_ss_list_per_gen_item).mean().item(), 5)
        else:
            rec_ss_per_case = -1
        rec_bs_per_case = round(torch.tensor(rec_bs_list_per_gen_item).mean().item(), 5)
        num_clash_per_case = round(torch.tensor(num_clash_per_gen_item).mean().item(), 5)

        rec_seq_list_per_case.append(rec_seq_per_case)
        rec_seq_sub_list_per_case.append(rec_seq_sub_per_case)
        rec_pos_list_per_case.append(rec_pos_per_case)
        if rec_ss_per_case != -1:
            rec_ss_list_per_case.append(rec_ss_per_case)
        rec_bs_list_per_case.append(rec_bs_per_case)
        num_clash_list_per_case.append(num_clash_per_case)

        div_list_per_pair = []
        for idx in range(len(sample_list)-1):
            for idx_compared in range(idx+1, len(sample_list)):
                chain1 = get_chain_from_pdb(os.path.join(item_output_dir, sample_list[idx]))
                chain2 = get_chain_from_pdb(os.path.join(item_output_dir, sample_list[idx_compared]))
                tm_score_pair = get_tm(chain1, chain2)
                # tm_score_pair = cal_tm(Ca_pos_list_per_case[idx], Ca_pos_list_per_case[idx_compared])
                div_list_per_pair.append(1 - tm_score_pair)
        div_per_case = round(torch.tensor(div_list_per_pair).mean().item(), 5)
        div_list_per_case.append(div_per_case)


        NS3_rec = SEP_rec + TYS_rec + PTR_rec
        gt_NS3_rec = gt_SEP_num + gt_TYS_num + gt_PTR_num
        temp_df = pd.DataFrame({
            'Case': [item],
            'rec_pos': [rec_pos_per_case],
            'rec_seq': [rec_seq_per_case],
            'rec_seq(sub)': [rec_seq_sub_per_case],
            'rec_ss': [rec_ss_per_case],
            'rec_bs': [rec_bs_per_case],
            'diversity': [div_per_case],
            'Clash': [num_clash_per_case],
            'rec_seq(SEP)': [f"{SEP_rec}_{gt_SEP_num}"],
            'rec_seq(TYS)': [f"{TYS_rec}_{gt_TYS_num}"],
            'rec_seq(PTR)': [f"{PTR_rec}_{gt_PTR_num}"],
            'rec_seq(NS3)': [f"{NS3_rec}_{gt_NS3_rec}"],
            'Success': [Success_num],
            'AFF(suc)': [AFF_num],
        })
        data_csv = pd.concat([data_csv, temp_df])
        # data_csv.to_csv(result_path, index=False)

    rec_seq_all = round(torch.tensor(rec_seq_list_per_case).mean().item(), 5)
    rec_seq_sub_all = round(torch.tensor(rec_seq_sub_list_per_case).mean().item(), 5)
    rec_pos_all = round(torch.tensor(rec_pos_list_per_case).mean().item(), 5)
    rec_ss_all = round(torch.tensor(rec_ss_list_per_case).mean().item(), 5)
    rec_bs_all = round(torch.tensor(rec_bs_list_per_case).mean().item(), 5)
    div_all = round(torch.tensor(div_list_per_case).mean().item(), 5)
    num_clash_all = round(torch.tensor(num_clash_list_per_case).mean().item(), 5)

    gen_SEP_all = 0
    gt_SEP_all = 0
    for _item in data_csv['rec_seq(SEP)'].values:
        if _item != '0_0':
            gen_SEP_all += int(_item.split('_')[0])
            gt_SEP_all += int(_item.split('_')[1])

    gen_TYS_all = 0
    gt_TYS_all = 0
    for _item in data_csv['rec_seq(TYS)'].values:
        if _item != '0_0':
            gen_TYS_all += int(_item.split('_')[0])
            gt_TYS_all += int(_item.split('_')[1])

    gen_PTR_all = 0
    gt_PTR_all = 0
    for _item in data_csv['rec_seq(PTR)'].values:
        if _item != '0_0':
            gen_PTR_all += int(_item.split('_')[0])
            gt_PTR_all += int(_item.split('_')[1])

    gen_NS3_all = 0
    gt_NS3_all = 0
    for _item in data_csv['rec_seq(NS3)'].values:
        if _item != '0_0':
            gen_NS3_all += int(_item.split('_')[0])
            gt_NS3_all += int(_item.split('_')[1])

    temp_df = pd.DataFrame({
        'Case': ['All'],
        'rec_pos': [rec_pos_all],
        'rec_seq': [rec_seq_all],
        'rec_seq(sub)': [rec_seq_sub_all],
        'rec_ss': [rec_ss_all],
        'rec_bs': [rec_bs_all],
        'diversity': [div_all],
        'Clash': [num_clash_all],
        'rec_seq(SEP)': [f"{gen_SEP_all}_{gt_SEP_all}"],
        'rec_seq(TYS)': [f"{gen_TYS_all}_{gt_TYS_all}"],
        'rec_seq(PTR)': [f"{gen_PTR_all}_{gt_PTR_all}"],
        'rec_seq(NS3)': [f"{gen_NS3_all}_{gt_NS3_all}"],
        'Success': [f"{sum(data_csv['Success'].to_list())}_{16*(len(data_csv)-data_csv['rec_seq'].isna().sum())}"],
        'AFF(suc)': [sum(data_csv['AFF(suc)'].to_list())],
    })
    data_csv = pd.concat([data_csv, temp_df])
    data_csv.to_csv(result_path, index=False)
