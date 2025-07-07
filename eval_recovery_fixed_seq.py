import os
import argparse
import pdb
import shutil
import numpy as np
import pandas as pd
import random
import json
import torch
from pepflow.utils.misc import load_config, seed_all
from models_con.sample import sample, sample_for_data_tao
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
    parser.add_argument('--ckpt', type=str, default="/ibex/user/guot0b/PepFlowww-ibex/ckpt/model1.pt")
    parser.add_argument('--data_dir', type=str,
                        default="/ibex/user/guot0b/DataPool/Pepflow/")
    parser.add_argument('--test_list', type=str,
                        default="/ibex/user/guot0b/PepFlowww-NS3/data/data_split_NS3_v3.json")
    parser.add_argument('--output_dir', type=str, default="/home/guot0b/disk3/guot0b/PepFlowww-1124/eval_output/")
    parser.add_argument('--output_name', type=str, default="eval_recovery_fixed_seq.csv")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num', type=int, default=16, help="Number of samples to generate per case")
    parser.add_argument('--config', type=str, default='./configs/learn_angle_tao.yaml')

    args = parser.parse_args()

    handle_existing_folder(args.output_dir)

    try:
        # Build the model to be evaluated
        config, config_name = load_config(args.config)
        config.dataset.train.tao_repr = True
        if config.dataset.train.tao_repr:
            # model = FlowModel_tao(config.model).to(args.device)
            # model = flow_model.FlowModel_NS3_SCAux(config.model).to(args.device)
            # model = flow_model.FlowModel_NS3_weight_loss(config.model).to(args.device)
            model = flow_model.NS_Pep(config.model).to(args.device)
            # model = flow_model.NS_Pep_BLV(config.model).to(args.device)
            # model = flow_model.NS_Pep_seesaw(config.model).to(args.device)
            # model = flow_model.NS_Pep_wo_rBLV(config.model).to(args.device)
            # model = flow_model.NS_Pep_dyMEAN_v1(config.model).to(args.device)
            # model = flow_model.NS_Pep_wo_GSP(config.model).to(args.device)
            # model = flow_model.NS_Pep_wo_IAW(config.model).to(args.device)
            # model = flow_model.NS_Pep_RFGM_no_noise(config.model).to(args.device)
            # model = flow_model.NS_Pep_BLV_no_noise(config.model).to(args.device)
        else:
            # model = flow_model.FlowModel_NS3(config.model).to(args.device)
            # model = flow_model.FlowModel_NS3_BLV(config.model).to(args.device)
            # model = flow_model.FlowModel_NS3_BLV_acc(config.model).to(args.device)
            # model = flow_model.FlowModel_NS3_rBLV(config.model).to(args.device)
            # model = flow_model.FlowModel_NS3_rBLV_norm(config.model).to(args.device)
            # model = flow_model.FlowModel_NS3_seesaw(config.model).to(args.device)
            model = flow_model.FlowModel_NS3_distEdge(config.model).to(args.device)
            # model = flow_model.FlowModel_change_seq(config.model).to(args.device)

        seed_all(config.train.seed)
        ckpt = torch.load(args.ckpt, map_location=args.device)
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

    eval_metrics = ['Case', 'rec_pos', 'rec_seq', 'rec_ss', 'rec_bs', 'diversity', 'Clash',
                    'rec_seq(SEP)', 'rec_seq(TYS)', 'rec_seq(PTR)', 'rec_seq(NS3)']
    rec_seq_list_per_case = []
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
        print(f"Evaluating PDB_ID {item} now...")
        item_folder = os.path.join(args.data_dir, item)
        assert os.path.isdir(item_folder), f"The {item_folder} is not a folder!"
        folder_con = os.listdir(item_folder)

        # If there isn't .pt file, make it
        pt_file = os.path.join(item_folder, f"{item}_re_NS3.pt")
        if not os.path.exists(pt_file):
            preprocess_structure_v2(item_folder, pt_file, renumber_chain=True)

        # Generation
        item_output_dir = os.path.join(args.output_dir, item)
        if os.path.exists(item_output_dir) and len(os.listdir(item_output_dir)) == args.num:
            print(f"The directory exits.")
        else:
            try:
                data_load = torch.load(pt_file)
                sample_for_data_tao(data_load, model, args.device, item_output_dir,
                                    tao_repr=False, nums=args.num,
                                    sample_structure=True, sample_sequence=False, sample_ang=True)
            except FileNotFoundError or RuntimeError:
                print(f"Skip the case {item}, because it can't be made as a .pt file!")
                continue


        # 2. Evaluation  (could be operated in parallel)
        # Extract the seq and struct for recovery calculation
        pt_con = torch.load(pt_file)
        gt_seq = pt_con['aa'][torch.where(pt_con['generate_mask'])]
        gt_pos = pt_con['pos_heavyatom'][torch.where(pt_con['generate_mask'])]  # including OXT
        gt_pos_rec = pt_con['pos_heavyatom'][torch.where(~pt_con['generate_mask'])]

        # Calculate the seq rec only for NS3
        # gt_NS3_mask = torch.isin(gt_seq, NS3_idx)
        gt_SEP_mask = torch.isin(gt_seq, SEP_idx)
        gt_SEP_num = gt_SEP_mask.sum().item() * args.num
        gt_TYS_mask = torch.isin(gt_seq, TYS_idx)
        gt_TYS_num = gt_TYS_mask.sum().item() * args.num
        gt_PTR_mask = torch.isin(gt_seq, PTR_idx)
        gt_PTR_num = gt_PTR_mask.sum().item() * args.num

        pt_to_pdb = os.path.join(item_folder, "pocket_merge_from_pt_NS3.pdb")
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
        rec_ss_list_per_gen_item = []
        rec_bs_list_per_gen_item = []
        num_clash_per_gen_item = []

        Ca_pos_list_per_case = []  # For diversity
        SEP_rec = 0
        TYS_rec = 0
        PTR_rec = 0
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
            binder_rmsd = torch.linalg.norm(gen_pos[:, 1] - gt_pos[:, 1], dim=-1).mean()
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
            rec_bs = get_bind_ratio(item_path, pt_to_pdb, "A", "A")
            # use 'pocket_merge.pdb' instead of 'receptor_merge.pdb'. Because the pocket is in 10 A distance.
            rec_bs_list_per_gen_item.append(rec_bs)

            # # 2.6 Clash
            """
            gen_pos, gen_seq = gen_con['complex_pos'], gen_con['complex_seq']
            gen_atom_mask = gen_con['complex_mask']
            peptide_idx = torch.where(gen_con['pep_mask'])[0]
            atom37, atom37_mask = atom14_to_atom37(gen_pos,
                                                   gen_atom_mask,
                                                   gen_con['complex_seq'])
            ret = compute_clashes(atom37[None, ...], atom37_mask[None, ...], gen_seq[None, ...])
            clash_pairs = ret['steric']['clashing_pairs']
            num_clash = 0
            for pair_idx in range(len(clash_pairs)):
                if clash_pairs[pair_idx][0][0] in peptide_idx or clash_pairs[pair_idx][1][0] in peptide_idx:
                    num_clash += 1
            num_clash /= peptide_idx.sum()  # Average on peptide length
            """
            num_clash = float(100)
            num_clash_per_gen_item.append(num_clash)

        rec_seq_per_case = round(torch.tensor(rec_seq_list_per_gen_item).mean().item(), 5)
        rec_pos_per_case = round(torch.tensor(rec_pos_list_per_gen_item).mean().item(), 5)
        if len(rec_ss_list_per_gen_item) == args.num:
            rec_ss_per_case = round(torch.tensor(rec_ss_list_per_gen_item).mean().item(), 5)
        else:
            rec_ss_per_case = -1
        rec_bs_per_case = round(torch.tensor(rec_bs_list_per_gen_item).mean().item(), 5)
        num_clash_per_case = round(torch.tensor(num_clash_per_gen_item).mean().item(), 5)

        rec_seq_list_per_case.append(rec_seq_per_case)
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
            'rec_ss': [rec_ss_per_case],
            'rec_bs': [rec_bs_per_case],
            'diversity': [div_per_case],
            'Clash': [num_clash_per_case],
            'rec_seq(SEP)': [f"{SEP_rec}_{gt_SEP_num}"],
            'rec_seq(TYS)': [f"{TYS_rec}_{gt_TYS_num}"],
            'rec_seq(PTR)': [f"{PTR_rec}_{gt_PTR_num}"],
            'rec_seq(NS3)': [f"{NS3_rec}_{gt_NS3_rec}"],
        })
        data_csv = pd.concat([data_csv, temp_df])

    rec_seq_all = round(torch.tensor(rec_seq_list_per_case).mean().item(), 5)
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
        'rec_ss': [rec_ss_all],
        'rec_bs': [rec_bs_all],
        'diversity': [div_all],
        'Clash': [num_clash_all],
        'rec_seq(SEP)': [f"{gen_SEP_all}_{gt_SEP_all}"],
        'rec_seq(TYS)': [f"{gen_TYS_all}_{gt_TYS_all}"],
        'rec_seq(PTR)': [f"{gen_PTR_all}_{gt_PTR_all}"],
        'rec_seq(NS3)': [f"{gen_NS3_all}_{gt_NS3_all}"],
    })
    data_csv = pd.concat([data_csv, temp_df])
    data_csv.to_csv(result_path, index=False)
