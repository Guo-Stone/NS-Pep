import os
import argparse
import pdb
import shutil
import numpy as np
import pandas as pd
import random
import json
import torch
import pepflow.modules.protein.constants as constants
from pepflow.utils.misc import load_config, seed_all
from models_con.sample import sample, sample_for_NS18
from models_con.pep_dataloader import preprocess_structure_v2, preprocess_structure_v3
import models_con.flow_model as flow_model
from models_con.flow_model import FlowModel, FlowModel_tao
from data.utils import extract_pdb, handle_existing_folder
from pepflow.modules.protein.writers import save_pdb
from eval.geometry import *
from eval.attnpacker.project_sidechains import atom14_to_atom37, compute_clashes

# For recovery and diversity evaluation
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default="/ibex/user/guot0b/PepFlowww-NS18/logs/2025.05.08_v2_NS-Pep_NS18_1177_320k/checkpoints/320000.pt")
    parser.add_argument('--data_dir', type=str,
                        default="/ibex/user/guot0b/DataPool/Pepflow/")
    parser.add_argument('--test_list', type=str,
                        default="/ibex/user/guot0b/PepFlowww-NS18/data/data_split_NS18_v4_seed1177_curated.json")
    parser.add_argument('--output_dir', type=str, default="/ibex/user/guot0b/PepFlowww-NS18/logs/2025.05.08_v2_NS-Pep_NS18_1177_320k/test_320k_num16/")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num', type=int, default=16, help="Number of samples to generate per case")
    parser.add_argument('--config', type=str, default='./configs/learn_angle_NS18.yaml')
    parser.add_argument('--result_path', type=str, default='eval_recovery.csv')

    args = parser.parse_args()

    handle_existing_folder(args.output_dir)

    try:
        # Build the model to be evaluated
        config, config_name = load_config(args.config)
        config.dataset.train.tao_repr = True
        if config.dataset.train.tao_repr:
            # model = flow_model.NS_Pep_NS18(config.model).to(args.device)
            # model = flow_model.NS_Pep_NS18_v2(config.model).to(args.device)
            # model = flow_model.NS_Pep_NS18_v2_noSCAA(config.model).to(args.device)
            # model = flow_model.NS_Pep_NS18_chi(config.model).to(args.device)
            # model = flow_model.NS_Pep_NS18_chi_SC(config.model).to(args.device)
            model = flow_model.NS_Pep_v3(config.model).to(args.device)
            # model = flow_model.NS_Pep_v4(config.model).to(args.device)
            # model = flow_model.FlowModel_NS18_IAW(config.model).to(args.device)
            # model = flow_model.FlowModel_NS18_PSP(config.model).to(args.device)
            # model = flow_model.FlowModel_NS18_GSP(config.model).to(args.device)
            # model = flow_model.FlowModel_NS18_chi_IAW(config.model).to(args.device)
            # model = flow_model.FlowModel_NS18_GSP_t75(config.model).to(args.device)
            # model = flow_model.FlowModel_NS18_chi_scaa(config.model).to(args.device)
            # model = flow_model.FlowModel_NS18_chi_scaa_same_param(config.model).to(args.device)
        else:
            # model = flow_model.FlowModel_NS18(config.model).to(args.device)
            # model = flow_model.FlowModel_NS18_BLV(config.model).to(args.device)
            model = flow_model.FlowModel_NS18_RFGM(config.model).to(args.device)
            # model = flow_model.FlowModel_NS18_dyMEAN(config.model).to(args.device)
            # model = flow_model.FlowModel_NS18_chi(config.model).to(args.device)
            # model = flow_model.FlowModel_NS18_chi_same_param(config.model).to(args.device)


        seed_all(config.train.seed)

        ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
        state_dict = ckpt['model']
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('module.', '') if key.startswith('module.') else key
            new_state_dict[new_key] = state_dict[key]
        model.load_state_dict(new_state_dict)
    except:
        model = None  # We don't need to generate, because it has generated before

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

    with open("./data/NS18_list.json", 'r') as f:
        ns18 = json.load(f)

    eval_metrics = ['Case', 'rec_pos(C-RMSD)', 'rec_pos(L-RMSD)', 'rec_seq', 'rec_ss', 'rec_bs', 'diversity', 'rec_seq(NS18)']
    NS18_idx_map = {}
    NS18_idx = []
    for nsaa in ns18:
        eval_metrics.append(f"rec_seq({nsaa})")
        NS18_idx_map[nsaa] = torch.tensor(int(constants.AA(nsaa)))
        NS18_idx.append(int(constants.AA(nsaa)))
    NS18_idx = torch.tensor(NS18_idx)

    rec_seq_list_per_case = []
    rec_pos_list_per_case = []
    rec_pos_ligand_list_per_case = []
    rec_ss_list_per_case = []  # Secondary structure recovery
    rec_bs_list_per_case = []  # Binding site recovery
    div_list_per_case = []  # diversity
    num_clash_list_per_case = []

    data_csv = pd.DataFrame(columns=eval_metrics)
    result_path = os.path.join(args.output_dir, args.result_path)

    for item in test_item:
        print(f"Evaluating PDB_ID {item} now...")
        item_folder = os.path.join(args.data_dir, item)
        assert os.path.isdir(item_folder), f"The {item_folder} is not a folder!"
        folder_con = os.listdir(item_folder)
        # If there isn't .pt file, make it
        pt_file = os.path.join(item_folder, f"{item}_re_NS18.pt")
        if not os.path.exists(pt_file):
            preprocess_structure_v2(item_folder, pt_file, renumber_chain=True)

        # Generation
        item_output_dir = os.path.join(args.output_dir, item)
        if os.path.exists(item_output_dir) and len(os.listdir(item_output_dir)) == args.num:
            print(f"The directory exits.")
        else:
            try:
                data_load = torch.load(pt_file)
                sample_for_NS18(data_load, model, args.device, item_output_dir,
                                    tao_repr=False, nums=args.num,
                                    sample_structure=True, sample_sequence=True, sample_ang=True)
            except FileNotFoundError or RuntimeError:
                print(f"Skip the case {item}, because it can't be made as a .pt file!")
                continue


        # 2. Evaluation  (could be operated in parallel)
        # Extract the seq and struct for recovery calculation
        pt_con = torch.load(pt_file)
        gt_seq = pt_con['aa'][torch.where(pt_con['generate_mask'])]
        if len(gt_seq) < 3 or len(gt_seq) > 25:
            continue
        gt_pos = pt_con['pos_heavyatom'][torch.where(pt_con['generate_mask'])]  # including OXT
        gt_pos_rec = pt_con['pos_heavyatom'][torch.where(~pt_con['generate_mask'])]

        # Calculate the seq rec only for NS18
        gt_NS18_dict = {"NS18_num": 0}
        for nsaa in ns18:
            gt_NS18_dict[f"{nsaa}_mask"] = torch.isin(gt_seq, NS18_idx_map[nsaa])
            gt_NS18_dict[f"{nsaa}_num"] = gt_NS18_dict[f"{nsaa}_mask"].sum().item() * args.num
            gt_NS18_dict["NS18_num"] += gt_NS18_dict[f"{nsaa}_num"]

        pt_to_pdb = os.path.join(item_folder, "pocket_merge_from_pt_NS18.pdb")
        if not os.path.exists(pt_to_pdb):
            save_pdb(pt_con, path=pt_to_pdb)
        # This pdb are in the same setting as the generated one.
        # It makes sure that the lengths of gt and gen are identical.

        # Extract the traj for ss and binding site calculation
        # gt_pep_file = os.path.join(item_folder, f"peptide_re.pdb")
        # gt_complex_file = os.path.join(item_folder, f"pocket_merge.pdb")
        gt_traj = get_traj_chain(pt_to_pdb, "A")

        sample_list = os.listdir(item_output_dir)
        rec_seq_list_per_gen_item = []
        rec_pos_list_per_gen_item = []
        rec_pos_ligand_list_per_gen_item = []
        rec_ss_list_per_gen_item = []
        rec_bs_list_per_gen_item = []

        Ca_pos_list_per_case = []  # For diversity
        NS18_rec = {nsaa: 0 for nsaa in ns18}
        NS18_rec['NS18'] = 0
        for gen_item in sample_list:
            if not gen_item.endswith('.pdb'):
                continue
            # gen_ca, gen_seq = extract_pdb(os.path.join(item_output_dir, gen_item))
            item_path = os.path.join(item_output_dir, gen_item)
            gen_con = preprocess_structure_v3(item_path)
            gen_pos, gen_seq = gen_con['pep_pos'], gen_con['pep_seq']
            gen_pos_rec = gen_con['rec_pos']
            Ca_pos_list_per_case.append(gen_pos[:, 1])
            # # 2.1 Seq recovery
            assert len(gt_seq) == len(gen_seq), "len(gt_seq) != len(gen_seq)"
            rec_seq = (gt_seq == gen_seq).sum() / len(gen_seq)
            rec_seq_list_per_gen_item.append(rec_seq.item())

            # # 2.1.1 Seq recovery for NS18
            # Calculate the seq rec only for NS18            
            for nsaa in ns18:
                gen_mask = torch.isin(gen_seq, NS18_idx_map[nsaa]) & gt_NS18_dict[f"{nsaa}_mask"]
                NS18_rec[nsaa] += gen_mask.sum().item()

            # # 2.2 Pos recovery w.r.t. CA (Don't need to align)
            # Rot = kabsch_R(gt_pos_rec[:, 1], gen_pos_rec[:, 1])
            # gen_pos_aligned = (gen_pos - gen_pos_rec[:, 1].mean(0)) @ Rot + gt_pos_rec[:, 1].mean(0)
            # C-RMSD
            binder_rmsd = torch.linalg.norm(gen_pos[:, 1] - gt_pos[:, 1], dim=-1).mean()
            rec_pos_list_per_gen_item.append(binder_rmsd.item())
            # L-RMSD
            chain_gt = get_chain_from_pdb(pt_to_pdb)
            chain_gen = get_chain_from_pdb(item_path)
            rmsd_rec, rmsd_lig = get_rmsd(chain_gt, chain_gen)        
            rec_pos_ligand_list_per_gen_item.append(rmsd_lig)

            # # 2.3 Secondary structure recovery
            gen_traj = get_traj_chain(item_path, "A")
            try:
                rec_ss = get_ss(gt_traj, gen_traj)
            except ValueError:
                rec_ss = -1
            if rec_ss != -1:
                rec_ss_list_per_gen_item.append(rec_ss.item())

            # # 2.4 Binding site recovery
            rec_bs = get_bind_ratio(item_path, pt_to_pdb, "A", "A")
            # use 'pocket_merge.pdb' instead of 'receptor_merge.pdb'. Because the pocket is in 10 A distance.
            rec_bs_list_per_gen_item.append(rec_bs)

        rec_seq_per_case = round(torch.tensor(rec_seq_list_per_gen_item).mean().item(), 5)
        rec_pos_per_case = round(torch.tensor(rec_pos_list_per_gen_item).mean().item(), 5)
        rec_pos_ligand_per_case = round(torch.tensor(rec_pos_ligand_list_per_gen_item).mean().item(), 5)
        if len(rec_ss_list_per_gen_item) == args.num:
            rec_ss_per_case = round(torch.tensor(rec_ss_list_per_gen_item).mean().item(), 5)
        else:
            rec_ss_per_case = -1
        rec_bs_per_case = round(torch.tensor(rec_bs_list_per_gen_item).mean().item(), 5)

        rec_seq_list_per_case.append(rec_seq_per_case)
        rec_pos_list_per_case.append(rec_pos_per_case)
        rec_pos_ligand_list_per_case.append(rec_pos_ligand_per_case)
        if rec_ss_per_case != -1:
            rec_ss_list_per_case.append(rec_ss_per_case)
        rec_bs_list_per_case.append(rec_bs_per_case)

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

        for nsaa in ns18:
            NS18_rec['NS18'] += NS18_rec[nsaa]

        temp_df = pd.DataFrame({
            'Case': [item],
            'rec_pos(C-RMSD)': [rec_pos_per_case],
            'rec_pos(L-RMSD)': [rec_pos_ligand_per_case],
            'rec_seq': [rec_seq_per_case],
            'rec_ss': [rec_ss_per_case],
            'rec_bs': [rec_bs_per_case],
            'diversity': [div_per_case],
            'rec_seq(NS18)': [f"{NS18_rec['NS18']}_{gt_NS18_dict['NS18_num']}"],
        })

        for nsaa in ns18:
            temp_df[f'rec_seq({nsaa})'] = [f"{NS18_rec[nsaa]}_{gt_NS18_dict[f'{nsaa}_num']}"]
        data_csv = pd.concat([data_csv, temp_df])

    rec_seq_all = round(torch.tensor(rec_seq_list_per_case).mean().item(), 5)
    rec_pos_all = round(torch.tensor(rec_pos_list_per_case).mean().item(), 5)
    rec_pos_ligand_all = round(torch.tensor(rec_pos_ligand_list_per_case).mean().item(), 5)
    rec_ss_all = round(torch.tensor(rec_ss_list_per_case).mean().item(), 5)
    rec_bs_all = round(torch.tensor(rec_bs_list_per_case).mean().item(), 5)
    div_all = round(torch.tensor(div_list_per_case).mean().item(), 5)

    temp_df = pd.DataFrame({
        'Case': ['All'],
        'rec_pos(C-RMSD)': [rec_pos_all],
        'rec_pos(L-RMSD)': [rec_pos_ligand_all],
        'rec_seq': [rec_seq_all],
        'rec_ss': [rec_ss_all],
        'rec_bs': [rec_bs_all],
        'diversity': [div_all],
    })

    for nsaa in (['NS18'] + ns18):
        NS18_rec[f'gen_{nsaa}_all'] = 0
        NS18_rec[f'gt_{nsaa}_all'] = 0
        for _item in data_csv[f'rec_seq({nsaa})'].values:
            # pdb.set_trace()
            if _item != '0_0':
                NS18_rec[f'gen_{nsaa}_all'] += int(_item.split('_')[0])
                NS18_rec[f'gt_{nsaa}_all'] += int(_item.split('_')[1])
        temp_df[f'rec_seq({nsaa})'] = [f"{NS18_rec[f'gen_{nsaa}_all']}_{NS18_rec[f'gt_{nsaa}_all']}"]

    data_csv = pd.concat([data_csv, temp_df])
    data_csv.to_csv(result_path, index=False)
