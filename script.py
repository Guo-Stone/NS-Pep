import os
import pdb
import shutil
import pandas as pd
import random
import numpy as np
import pickle as pk
import torch
from data.utils import count_aa, rec_aa, extract_pdb, find_LCS_and_masks, kabsch_R, renumber_pdb_residues, merge_pdb
import random
import copy
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models_con.pep_dataloader import preprocess_structure_v2, preprocess_structure_v3
import seaborn as sns
import pepflow.modules.protein.constants as constants
from eval.energy import get_rosetta_score_base
from Bio.PDB import PDBParser, PDBIO, Select, Selection
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.SeqUtils import seq1

random.seed(666)


def get_fasta_from_pdb(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure("pdb", pdb_file)

    fasta_sequence = ""
    for chain in structure.get_chains():
        for residue in chain.get_residues():
            fasta_sequence += seq1(residue.get_resname())

    return fasta_sequence

structure_dir = "/ibex/user/guot0b/DataPool/Pepflow/"
# test_items = []
# with open("/ibex/user/guot0b/PepFlowww-NS3/data/PepMerge_lmdb/Fixed Data/names.txt", 'r') as f:
#     for line in f:
#         test_items.append(line.strip())

# test_list = "/ibex/user/guot0b/PepFlowww-NS3/data/PepMerge_lmdb/Fixed Data/names.txt"
test_list ="/ibex/user/guot0b/PepFlowww-ibex/data/data_split_NS3_v3.json"
if test_list.endswith('.txt'):
    # Test set in pepflow
    test_items = []
    with open(test_list, 'r') as f:
        for line in f:
            test_items.append(line.strip())
elif test_list.endswith('.json'):
    with open(test_list, 'r') as f:
        data_split = json.load(f)
    test_items = data_split['TEST']
    val_items = data_split['VAL']
    train_items = data_split['TRAIN']    

with open("/ibex/user/guot0b/PepFlowww-ibex/data/data_split_NS3_v4.json", 'r') as f:
    data_split = json.load(f)
NS3_test_items = data_split['TEST']

if False:
    from models_con.sample import sample_tao

    # sample debug
    sample_tao("/home/guot0b/disk3/guot0b/PepFlowww-main/data/5emb_ns3_test.pt",
               "/home/guot0b/disk3/guot0b/PepFlowww-main/output/test/",
               # "/home/guot0b/disk3/guot0b/PepFlowww-main/logs/2024.11.24 PepFlow_SEP 160k scratch/checkpoints/120000.pt",
               "/home/guot0b/disk3/guot0b/PepFlowww-main/logs/20241202_test/checkpoints/8.pt",
               num=1)
elif False:
    # Enlarge dataset
    src_dir = "/home/guot0b/disk3/guot0b/PepFlowww-main/data/PepMerge_release/"
    tgt_dir = "/home/guot0b/disk3/guot0b/PepFlowww-main/data/PepMerge_SEP_v2/"

    # src_con = os.listdir(src_dir)
    # tgt_con = os.listdir(tgt_dir)
    #
    # count = 0
    # threshold = 1000
    # for item in src_con:
    #     if item not in tgt_con:
    #         shutil.copytree(os.path.join(src_dir, item),
    #                         os.path.join(tgt_dir, item))
    #         count += 1
    #     if count > threshold:
    #         break

    count_aa(tgt_dir, os.path.join("/home/guot0b/disk3/guot0b/PepFlowww-main/data/", "aa_count_SEP_v2.csv"))
    print(f"Now we have {len(os.listdir(tgt_dir))} samples.")
elif False:
    # Get the resolution of structure
    from data.utils import fetch_pdb_resolution

    NAME = 'DAL_ns'
    # 'TYS_ns'(3sv2_I, 1.3), 'PTR_ns'(3s8o_B, 1.85),
    # 'MLE_ns'(2oju_D, 2.4), 'M3L_ns'(3mp1_P, 2.6), 'ALY_ns'(4fz3_B, 2.1)
    # 'TPO_ns'(6gy2_D, 3.11), 'DAL_ns'(2oju_D, 2.4)
    df = pd.read_csv("/home/guot0b/disk3/guot0b/PepFlowww-main/data/aa_count_detailed.csv")

    row_id = np.where((df['aatype'] == NAME).values)[0].item()
    con = df.iloc[row_id, -1][:-2].split(', ')

    min_res = 10.
    for item in con:
        res = fetch_pdb_resolution(item.split('_')[0])[0]
        if type(res) is float:
            if res < 10.:
                min_res = res
                min_item = item
    print(f"{min_item} {min_res}")
elif False:
    # Calculate aa of generated peptide
    tgt_dir = "/ibex/user/guot0b/PepFlowww-NS3/logs/2025.01.06_v1_pepflow_NS3_BLV_10_acc_weight_320k/test_320k_num16/"
    con = os.listdir(tgt_dir)
    record = {}

    for case in test_items:
        print(f"Calculating {case} now ...")
        case_dir = os.path.join(tgt_dir, case)
        sample_list = os.listdir(case_dir)
        for sample in sample_list:
            sample_path = os.path.join(case_dir, sample)
            rec = rec_aa(sample_path, chain_id='A')
            aatypes = rec[0]

            for item in aatypes:
                if item in record.keys():
                    record[item] += 1
                else:
                    record[item] = 1
    sorted_aa_dict = dict(sorted(record.items(), key=lambda item: item[1], reverse=True))
    print(sorted_aa_dict)
elif False:
    # Weight the data with amino acid content
    src_dir = "/home/guot0b/disk3/guot0b/PepFlowww-main/data/PepMerge_SEP_v2/"
    tgt_file = os.path.join(src_dir, "aa_count.csv")
    count_aa(src_dir, tgt_file, T_ex=True)
elif False:
    # Turn the dataset to numpy.array
    data_path = "/home/guot0b/disk3/guot0b/PepFlowww-main/pep_dataset_ns3.pt"
    data_ori = torch.load(data_path)
    data_new = []
    for idx in range(len(data_ori)):
        item_dict = {}
        for key in data_ori[idx].keys():
            if type(data_ori[idx][key]) is torch.Tensor:
                item_dict[key] = data_ori[idx][key].numpy()
            else:
                item_dict[key] = data_ori[idx][key]
        data_new.append(item_dict)
    with open("/home/guot0b/disk3/guot0b/PepFlowww-main/data/peptide_dataset.pkl", 'wb') as f:
        pk.dump(data_new, f)

    with open("/home/guot0b/disk3/guot0b/PepFlowww-main/data/peptide_dataset.pkl", 'rb') as f:
        loaded_dataset = pk.load(f)
elif False:
    # Split the train/val/tet
    src_folder = "/home/guot0b/disk3/guot0b/PepFlowww-main/data/PepMerge_release"
    all_items = os.listdir(src_folder)
    total_num = len(all_items)  # 10384

    ori_test_items = []
    with open("/home/guot0b/disk3/guot0b/PepFlowww-main/data/PepMerge_lmdb/Fixed Data/names.txt", 'r') as f:
        for line in f:
            ori_test_items.append(line.strip())
    # test_num = len(ori_test_items)  # 187
    # test_ratio = test_num / total_num
    test_ratio = 0.1

    ori_train_items = set(all_items) - set(ori_test_items)
    ori_train_val_pool = [item for item in ori_train_items]

    SEP_case = "4dat_B, 4iaf_S, 1jlu_S, 5ecg_F, 3ob2_A, 5ewz_D, 3p9y_E, 5em9_B, 5fc2_A, 1lf8_E, 6fby_D, 3wp1_A, " \
               "3q1i_E, 4aze_E, 5d3f_C, 2vxc_C, 3k0k_B, 6fi5_B, 3smo_P, 6fau_D, 5vkl_B, 5nwi_P, 7tbh_1_III_JA, " \
               "6faw_B, 5v4b_C, 7tbh_1_III_TA, 5lu2_C, 3nkx_P, 7tbh_1_III_AA, 3sqd_C, 3d9p_Z, 3g2v_C, 5nnd_D, 4yj4_B, " \
               "7tbh_1_III_DA, 6da1_C, 4cse_D, 6jxv_B, 5ieh_C, 5jej_C, 2c1j_C, 5jeo_B, 6bb4_P, 5lx2_B, 6fav_B, " \
               "6sad_C, 7tbh_1_III_VA, 6gkf_I, 6b8j_C, 4ib3_S, 3fqx_C, 6dat_F, 6m92_C, 2n8t_B, 5lvp_F, 7tbh_1_III_KA, " \
               "6gkg_J, 7tbh_1_III_Z, 4y2g_B, 3p1o_P, 1t2v_F, 7tbh_1_III_NA, 6mf5_C, 4iea_P, 6fby_B, 2ght_C, 4jiz_B, " \
               "7tbh_1_III_HA, 6g8p_P, 6m91_C, 5m36_C, 4psi_D, 1pua_B, 3p1s_P, 7tbh_1_III_MA, 4jlu_B, 6g6x_P, 3iqu_P, " \
               "5m37_C, 4y5i_F, 4ckt_D, 3k16_B, 4jfz_P, 3iqv_P, 3wp0_B, 6qsz_B, 3ual_P, 2lue_B, 2rt5_B, 6e4z_P, " \
               "6a5s_E, 3pxe_E, 6g8l_P, 6fbw_D, 3cu8_P, 6uyy_B, 5w94_E, 6k5t_B, 3fqu_C, 7tbh_1_III_BA, 2n1o_B, " \
               "1qja_Q, 4iay_S, 1y98_B, 3p1r_P, 5ypo_C, 6qzs_P, 4y32_C, 6m94_C, 6ghp_P, 4yoo_X, 5nwk_V, 3smk_P, " \
               "6s9k_B, 5n5w_P, 6e21_B, 4yos_E, 2w3o_C, 7tbh_1_III_Y, 6bd2_C, 2btp_P, 3d9k_Z, 1ujk_C, 4w5a_C, 6hep_E, " \
               "6fi4_B, 2b05_G, 5ods_E, 2azm_C, 7tbh_1_III_RA, 6gkg_I, 3k0h_B, 6g8k_P, 4wrq_C, 6qzs_C, 6k5r_B, " \
               "2c1n_C, 2wh0_Q, 3spr_P, 5n7g_C, 5n5t_P, 6f08_D, 7tbh_1_III_GA, 5sw9_B, 6uyv_B, 5zia_C, 5e2v_P, " \
               "6eih_P, 4ib1_S, 1qjb_Q, 5wum_B, 7tbh_1_III_IA, 6qtm_F, 5m35_C, 4ihl_P, 4fl5_P, 5yqg_F, 3bh8_C, " \
               "6rmm_P, 6fav_D, 5d3e_C, 3uat_B, 5nwj_P, 3p35_D, 5yqg_E, 3iqj_P, 6j9j_B, 1f8a_C, 6qk8_E, 4ifi_B, " \
               "3ubw_P, 4dau_B, 4hkc_B, 6jxx_B, 3d9m_Z, 5gmv_D, 4zq0_E, 5swf_B, 6npw_E, 6bd1_C, 3szm_I, 4j6s_E, " \
               "7tbh_1_III_FA, 4fj3_P, 1py1_E, 4igk_C, 6ef5_Q, 4y18_I, 3t7k_C, 3q4a_C, 3coj_H, 6uys_B, 5myc_P, " \
               "5d2d_C, 2br9_P, 4y3b_C, 6uyt_B, 1p16_C, 3sml_P, 5lu1_C, 5ema_B, 1ywt_C, 6uyu_B, 3bgm_C, 5e50_D, " \
               "5xw5_C, 4q94_D, 3k05_C, 7tbh_1_III_LA, 6ni2_V, 6h0e_I, 2lb0_B, 4iak_S, 3uzd_B, 2npm_X, 4iai_S, " \
               "6f09_A, 4bg6_Q, 1a37_P, 6a5q_E, 3l6f_C, 3v7d_E, 4iad_S, 6g8q_P, 5jel_B, 3ux0_P, 3fqr_C, 6fel_E, " \
               "6du2_C, 4iaz_S, 6m90_C, 4jxt_B, 6m93_C, 6fau_B, 3d9n_Y, 6f09_D, 6fbb_P, 6hm5_B, 3ueo_F, 5e2w_P, " \
               "5emb_B, 5n7f_C, 6qh7_Q, 4pz6_Q, 2ghq_C, 4fr3_P, 3huf_E, 4ib0_S, 4hpt_I, 6du3_C, 5hf3_B, 3sp5_P, " \
               "2l0i_B, 2laz_B, 4nm7_C, 5u6k_L, 5ulo_C, 6h06_I, 6bd1_D, 5m35_D, 5lxi_C, 2lay_B, 7tbh_1_III_UA, " \
               "5n5r_P, 2jkr_P, 4jqi_V, 3l41_B, 6faw_D, 5btv_P, 6b67_D, 1t15_B, 7tbh_1_III_CA, 7tbh_1_III_OA, 2lo6_B, " \
               "3axy_E, 6dc9_P, 3bhb_C, 3g2t_C, 6ejl_C, 4wjp_B, 6fcp_P, 3p1q_P, 3p1p_P, 6ttu_I, 3bh9_C, 4jg0_P, " \
               "1t29_B, 4no2_C, 2lax_B, 4wjn_B, 3d9l_Y, 4nnx_C, 6g8i_P, 2ruk_A, 4yk9_B, 5w4s_B, 5jek_C, " \
               "7tbh_1_III_PA, 7tbh_1_III_SA, 6fbw_B, 5dmg_P, 2n9e_A, 5m9d_B, 1p22_C, 6g8j_P, 3shv_C, 6f08_Q, 6eww_E, " \
               "2c63_P, 3u3z_B, 4u4a_D, 3p1n_P, 1ohe_B, 6iht_1_III_B, 3wdz_B, 5w7x_H, 3smm_P, 5k6s_B, 3k15_B, 4hpu_I, " \
               "6bnt_B, 2laj_B, 4x8n_B, 5my9_P, 6gkf_J, 7tbh_1_III_QA, 6mtu_D, 7tbh_1_III_EA, 3zkf_F, 2jkt_P, 4no3_C, " \
               "3o8i_B, 6ef5_P, 2c74_P, 2ovq_C, 5n75_P, 3smn_P, 5nas_C, 3mhr_P, 3t1n_C, 4o46_G"
    TYS_case = "1d3t_H, 1g1s_C, 4z64_P, 1d3d_H, 2zhf_I, 1d4p_H, 5ahg_I, 1no9_I, 1qj1_I, 1bcu_I, 3rm0_I, 1wbg_I, " \
               "2uuf_H, 2bvx_I, 1c1v_I, 1c1w_I, 2zhq_I, 1o5g_I, 1z71_B, 1xm1_H, 1ay6_I, 2zgb_I, 7kme_I, 2zi2_I, " \
               "4bak_D, 6i51_I, 3rly_I, 2zdv_I, 2zfr_I, 4uff_I, 1c4v_3, 1doj_B, 2zg0_I, 4bam_D, 1hgt_I, 3d49_I, " \
               "1ba8_C, 1a4w_I, 1awf_I, 3u98_I, 2pks_D, 1zgi_B, 1k21_I, 1ca8_C, 1twx_C, 1ypg_I, 1nrs_I, 1c4y_3, " \
               "3rmo_I, 3shc_I, 6e4z_P, 1tbz_I, 5mm6_I, 2c90_I, 1h8i_I, 1zrb_B, 4nze_I, 2mpm_B, 4ue7_I, 2ziq_I, " \
               "2c8y_I, 1aht_I, 4ufe_I, 3qx5_I, 1g30_C, 1c5l_I, 1nm6_B, 1ny2_3, 3utu_I, 2zhw_I, 3egk_I, 4e7r_I, " \
               "1mu6_C, 2uuj_H, 2zf0_I, 4loy_I, 4bah_D, 6eo8_I, 2zc9_I, 1o2g_I, 2c8z_I, 3u9a_I, 1ta2_B, 3rmn_I, " \
               "3p17_I, 1c1u_I, 2jh6_H, 3rm2_I, 2jh5_H, 1tmb_I, 3dux_I, 2zff_I, 3eq0_I, 1ghx_I, 2bvs_I, 6s6q_C, " \
               "1tom_I, 1ghy_I, 1g32_C, 1riw_D, 1zgv_B, 1mu8_C, 1c4u_3, 6fjt_I, 1gj5_I, 3qwc_I, 2znk_I, 1k22_I, " \
               "5lce_I, 1ypm_I, 1a2c_I, 1ad8_I, 2zgx_I, 1d6w_I, 1qj6_I, 1kts_C, 5jzy_I, 2uuk_H, 2k3u_B, 5jfd_I, " \
               "1vr1_I, 6rot_A, 4ufg_I, 1b5g_I, 1hdt_P, 1ae8_I, 1ypk_I, 2c8w_I, 1a46_I, 4ban_D, 1bb0_C, 1tmu_J, " \
               "1fpc_I, 1way_I, 1sl3_B, 1d3q_H, 3rml_I, 3hat_I, 5yy4_B, 3rlw_I, 1ta6_B, 4udw_I, 1hah_I, 1nt1_H, " \
               "1aix_I, 2zfp_I, 1t4v_I, 5a2m_I, 4lxb_I, 2bxu_I, 1ypj_I, 1ype_I, 1h8d_I, 5lpd_I, 2zo3_I, 1a61_I, " \
               "4ufd_I, 1gj4_I, 4z5w_Q, 3biv_I, 3rmm_I, 1a5g_I, 2c93_I, 1ai8_I, 3qtv_I, 3uwj_I, 1d3p_H, 1uma_I, " \
               "4z61_Q, 6hsx_I, 3qto_I, 8kme_3, 3t5f_I, 4bao_D, 2bdy_B, 1qbv_E, 4yes_H, 1c5o_I, 1ghw_I, 3sha_I, " \
               "2c8x_I, 1qj7_I, 4n3l_I, 5l6n_I, 3dhk_I, 2zhe_I, 1t4u_I, 2jh0_H, 3f68_I, 1ktt_C, 6eo9_I, 2bvr_I, " \
               "1hag_I, 5mjt_D, 1c5n_I, 4baq_D, 1ghv_I, 4z63_P, 6gbw_I, 2gde_D, 5af9_I, 3c1k_B, 3si4_I, 1d9i_I, " \
               "2zfq_I, 3si3_I, 1mue_C, 1sb1_I, 4ueh_I, 2zda_I, 3biu_I, 1afe_I, 5mls_D, 1ypl_I, 2bxt_I, 1qhr_I, " \
               "3po1_D, 3dt0_I, 3sv2_I"
    PTR_case = "1kc2_B, 3pfv_C, 1een_B, 3ob2_A, 1fyr_I, 3tu7_I, 3tl0_B, 1f1w_B, 2oq1_B, 1yrk_B, 1lkk_B, 1a81_H, " \
               "5njk_G, 1tze_I, 4a4b_B, 2q8y_B, 1ptt_B, 1ayb_P, 1ygr_C, 4k45_B, 4rh5_B, 1mw4_B, 1ka6_B, 2z8p_B, " \
               "2iuh_B, 5gji_B, 1uef_C, 4icz_F, 6r5g_B, 5cdw_f, 1j4k_P, 1fhr_P, 4roj_D, 1a1e_C, 2mrk_B, 4gvc_B, " \
               "1aou_P, 5hz1_A, 1fu5_B, 2g35_B, 1bmb_I, 6kc4_B, 3fxx_B, 3omh_E, 5hz0_A, 4a4c_B, 1m0v_B, 1g1h_B, " \
               "2lct_B, 1ju5_B, 3s8l_B, 1a1c_C, 3da9_D, 5u1m_B, 4rh9_B, 1r1p_E, 1shc_B, 6oi4_E, 3d44_B, 3op0_C, " \
               "1jyq_L, 6i5j_I, 2ci9_L, 6i5j_K, 2cia_L, 2pld_B, 6i5n_I, 3vrr_C, 1csz_B, 4qsy_B, 5cdw_I, 1aot_P, " \
               "1bm2_L, 2iui_C, 6e8k_B, 1ygu_C, 5hl0_B, 2eu0_B, 1ptu_B, 5czh_B, 3d42_B, 4ch8_P, 1sps_D, 1oo4_B, " \
               "3zni_B, 4gfu_F, 3ov1_B, 4rhg_B, 2lnw_B, 4p9v_B, 3vro_B, 4jmh_B, 4u1p_B, 5hkz_B, 1lcj_B, 4p9z_B, " \
               "2mfq_B, 5x94_L, 1d4w_C, 5c26_B, 1xxp_C, 5hky_B, 1a1b_C, 1j4x_D, 1hcs_A, 1r1s_B, 5gjh_B, 4jmg_B, " \
               "2rmx_B, 2hdx_G, 6pxc_U, 5o76_B, 4qum_B, 2cbl_B, 2vif_P, 3ove_B, 1a07_C, 3vrp_B, 2i6o_B, 1eeo_B, " \
               "1shb_B, 5df6_B, 7t1u_2_III_D, 1fpr_B, 1jyr_L, 2ple_B, 2hmh_B, 5aul_B, 1k2m_P, 1csy_B, 1i3z_B, 3mxc_L, " \
               "3bu6_B, 1g1f_B, 5hz3_A, 6roy_D, 1pic_B, 3ob1_A, 1h9o_B, 1ayc_P, 1aya_P, 5hyx_A, 2y1n_D, 6b67_D, " \
               "1hct_A, 1irs_B, 2etz_B, 6roz_B, 1p13_D, 4ch2_P, 3ml4_E, 2fci_B, 8dgo_1_III_C, 4zrt_B, 3n8m_B, 5tq1_B, " \
               "5njj_H, 3tkz_P, 6dm4_E, 2yu7_B, 4s0g_B, 2bbu_B, 1yvh_B, 3olr_E, 4xz1_B, 5czi_B, 3buo_A, 2ror_B, " \
               "1shd_B, 3n84_G, 3n7y_D, 2nmb_B, 1lkl_B, 5x7b_L, 1g1g_B, 1sha_B, 3plf_A, 3mxy_L, 1zfp_I, 5tqs_E, " \
               "4gfv_F, 3u3z_B, 1qg1_I, 6i5n_K, 1tce_B, 2h5k_C, 1x27_I, 1yvl_C, 1r1q_C, 3s8n_B, 4r3p_B, 4gl9_K, " \
               "2hwl_P, 4gpl_A, 3maz_B, 4nnd_F, 4r3r_B, 2l4k_B, 1fbv_B, 1lck_B, 6i4x_D, 6c5x_G, 3s8o_B, 2mc1_B"
    SEP_case = SEP_case.split(", ")
    SEP_case.remove('3ob2_A')
    SEP_case.remove('6b67_D')
    SEP_case.remove('3u3z_B')
    SEP_case.remove('6e4z_P')
    TYS_case = TYS_case.split(", ")
    PTR_case = PTR_case.split(", ")
    random.shuffle(SEP_case)
    random.shuffle(TYS_case)
    random.shuffle(PTR_case)
    # Split NS3 into train/val/test
    train_NS3 = []
    val_NS3 = []
    test_NS3 = []
    for tgt_list in [SEP_case, TYS_case, PTR_case]:
        for idx in range(len(tgt_list)):
            if idx <= int(test_ratio * len(tgt_list)):
                test_NS3.append(tgt_list[idx])
            elif int(test_ratio * len(tgt_list)) < idx <= int(2 * test_ratio * len(tgt_list)):
                val_NS3.append(tgt_list[idx])
            else:
                train_NS3.append(tgt_list[idx])

    assert len(test_NS3) == len(set(test_NS3)), "There are duplicates in test list!"
    # test_overlap = set(test_NS3) & set(ori_test_items)

    non_NS3_pool = set(all_items) - set(val_NS3) - set(train_NS3) - set(test_NS3)

    # Make test_set
    test_pool = (set(ori_test_items) - set(val_NS3) - set(train_NS3)) | set(test_NS3)
    new_test_items = [item for item in test_pool]
    for item in non_NS3_pool:
        if len(new_test_items) >= int(test_ratio * total_num):
            break
        if item not in new_test_items:
            new_test_items.append(item)

    # Make val_set
    non_NS3_pool = non_NS3_pool - set(new_test_items)
    new_val_items = [item for item in val_NS3]
    for item in non_NS3_pool:
        if len(new_val_items) >= int(test_ratio * total_num):
            break
        if item not in new_val_items:
            new_val_items.append(item)

    # Make train_set
    train_pool = (non_NS3_pool - set(new_val_items)) | set(train_NS3)
    new_train_items = [item for item in train_pool]
    """
    # Add test_NS3 into test_set
    new_test_items = copy.deepcopy(ori_test_items)
    new_train_val_pool = copy.deepcopy(ori_train_val_pool)
    for item in test_NS3:
        if item in ori_train_val_pool:
            new_train_val_pool.remove(item)
        if item not in new_test_items:
            new_test_items.append(item)

    for item in new_test_items:
        if item not in test_NS3:
            new_train_val_pool.append(item)
            new_test_items.remove(item)
        if len(new_test_items) == test_num:
            break

    # Split train and val set
    non_NS3_pool = (set(new_train_val_pool) - set(val_NS3)) - set(train_NS3)
    new_val_items = copy.deepcopy(val_NS3)
    new_train_items = copy.deepcopy(new_train_val_pool)
    for item in non_NS3_pool:
        new_val_items.append(item)
        new_train_items.remove(item)
        if len(new_val_items) == test_num:
            break

    """

    assert len(set(test_NS3) & set(new_test_items)) == len(set(test_NS3))
    assert len(set(val_NS3) & set(new_val_items)) == len(set(val_NS3))
    assert len(set(train_NS3) & set(new_train_items)) == len(set(train_NS3))
    assert len(set(new_test_items) & set(new_val_items)) == 0
    assert len(set(new_test_items) & set(new_train_items)) == 0
    assert len(set(new_val_items) & set(new_train_items)) == 0
    assert (len(set(new_test_items)) + len(set(new_train_items)) + len(set(new_val_items))) == total_num

    data_split = {'TRAIN': new_train_items,
                  'VAL': new_val_items,
                  'TEST': new_test_items}
    with open('./data/data_split_20241218.json', 'w') as f:
        json.dump(data_split, f, indent=4)
elif False:
    # Analyze the split data
    tgt_split = "/ibex/user/guot0b/PepFlowww-NS3/data/data_split_NS3_v3.json"
    aa_rec_path = "/ibex/user/guot0b/PepFlowww-NS3/data/Analysis_NS3_v3/aa_rec.json"
    fig_save_path = "/ibex/user/guot0b/PepFlowww-NS3/data/Analysis_NS3_v3/aa_distribution_v2_amplified.png"
    with open(tgt_split, 'r') as f:
        data_split = json.load(f)
    """
    aa_rec = {}
    tgt_case = []
    # for mode in data_split.keys():
    # pdb.set_trace()
    for mode in ['TRAIN', 'VAL', 'TEST']:
        aa_rec[mode] = {}
        for item in data_split[mode]:
            pdb_path = os.path.join(structure_dir, item, 'peptide.pdb')
            aa_tmp, _ = rec_aa(pdb_path)
            for aatype in aa_tmp:
                if aatype not in ["ZN", "NH2", "NA", "UNK"]:
                    if aatype in aa_rec[mode].keys():
                        aa_rec[mode][aatype] += 1
                    else:
                        aa_rec[mode][aatype] = 1
                    if aatype in ['SEP', 'PTR', 'TYS'] and mode == 'TEST':
                        tgt_case.append(item)

    aa_density_sorted = {}
    # for mode in data_split.keys():
    for mode in ['TRAIN', 'VAL', 'TEST']:
        aa_density_sorted[mode] = dict(sorted(aa_rec[mode].items(), key=lambda x: x[1], reverse=True))
    pdb.set_trace()
    with open(aa_rec_path, 'w') as f:
        json.dump(aa_density_sorted, f, indent=4)
    """

    with open(aa_rec_path, 'r') as f:
        data = json.load(f)

    total_counts = {split: sum(data[split].values()) for split in data}
    pdb.set_trace()
    # Calculate frequencies
    # frequencies = {split: {aa: count * 100 / total_counts[split] for aa, count in data[split].items()} for split in data}

    # Extracting unique amino acids sorted by appearance in 'TRAIN'
    amino_acids = sorted(data['TRAIN'], key=data['TRAIN'].get, reverse=True)
    # amino_acids = amino_acids[-3:]
    # Preparing the data for plotting
    # freq_values = {split: [frequencies[split].get(aa, 0) for aa in amino_acids] for split in data}
    """
    # Plotting each dataset in separate subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)  # 1 row, 3 columns, sharing y-axis
    # pdb.set_trace()
    for ax, split in zip(axes, data):
        counts = [data[split].get(aa, 0) for aa in amino_acids]
        bars = ax.bar(amino_acids, counts, color='navy', alpha=0.75)
        # ax.bar(amino_acids, freq_values[split], color='navy', alpha=0.75)
        ax.set_title(f'{split} Dataset')
        # ax.set_xticklabels(amino_acids, rotation=45, fontsize=14)
        ax.set_xticklabels(amino_acids, rotation=45)
        ax.set_xlabel('Amino Acids')
        if ax == axes[0]:  # Only add y-label to the first subplot to avoid redundancy
            # ax.set_ylabel('Percentage')
            ax.set_ylabel('Count')

        # Add the exact numbers above each bar
        # for bar in bars:
        #     height = bar.get_height()
        #     ax.text(bar.get_x() + bar.get_width() / 2, height,
        #             f'{round(height, 3)}', ha='center', va='bottom', fontsize=14)

    plt.tight_layout()
    """

    # Stacked layer
    # Preparing the data for plotting
    amino_acids = amino_acids[-3:]
    fig, ax = plt.subplots(figsize=(6, 6))

    # Colors for each dataset
    colors = {'TRAIN': 'navy', 'VAL': 'green', 'TEST': 'orange'}

    # Initialize bottom values for stacking
    bottom = [0] * len(amino_acids)

    # Plot stacked bars
    for split in ['TRAIN', 'VAL', 'TEST']:
        counts = [data[split].get(aa, 0) for aa in amino_acids]
        ax.bar(amino_acids, counts, bottom=bottom, label=split, color=colors[split], alpha=0.75)
        bottom = [b + c for b, c in zip(bottom, counts)]  # Update bottom for next stack

    # Customize the plot
    # ax.set_title('Amino Acid Counts Across Datasets')
    # ax.set_xlabel('Amino Acids')
    # ax.set_ylabel('Counts')
    # ax.set_xticks(range(len(amino_acids)))
    # ax.set_xticklabels(amino_acids, rotation=45)
    # ax.legend(title='Dataset')
    plt.tight_layout()

    plt.savefig(fig_save_path)

    # pdb.set_trace()
elif False:
    # Cluster the data w.r.t. sequential identity
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1
    import tqdm

    src_folder = "/ibex/user/guot0b/PepFlowww-NS3/data/Analysis_NS3_v3/"
    total_num = len(os.listdir(structure_dir))
    ratio = 0.1  # 8: 1: 1

    # all_pdbs = os.listdir(structure_dir)
    all_pdbs = test_items
    """
    if not os.path.exists(src_folder):
        os.mkdir(src_folder)
    seq_file = os.path.join(src_folder, 'seqs_NS3_v2.fasta')

    with open(seq_file, 'w') as f:
        for pdb in all_pdbs:
            fasta = get_fasta_from_pdb(os.path.join(structure_dir, pdb, 'receptor.pdb'))
            f.write(f'>{pdb}\n')
            f.write(fasta + '\n')

    # $ mmseqs easy-cluster seqs.fasta clusterRes tmp --min-seq-id 0.4 -c 0.8 --cov-mode 1
    """
    tab = pd.read_csv("/ibex/user/guot0b/PepFlowww-NS3/data/Analysis_NS3_v3/clusterRes_cluster.tsv",
                      sep='\t', header=None)
    tab.columns = ['center', 'id']
    tab['cnts'] = tab.groupby('center')['id'].transform('count')
    tab.sort_values('cnts', ascending=False, inplace=True)

    cnts = tab.drop_duplicates('center')
    pdb.set_trace()
    # tab.to_csv("/home/guot0b/disk3/guot0b/PepFlowww-main/data/PepMerge_release_analysis/item_cluster.csv", index=None)
    cnts.to_csv("/ibex/user/guot0b/PepFlowww-NS3/data/Analysis_NS3_v3/center.csv", index=None)
    # centers_candidate = set((cnts[(cnts['cnts'] >= 10) & (cnts['cnts'] <= 100)])['center'])
    centers_candidate = cnts[(cnts['cnts'] >= 10) & (cnts['cnts'] <= 100)]

    # target_samples = int(total_num * ratio)
    target_samples = 187

    while True:  # Split until there are NS3 in test set. => No need. We can use our NS3_split instead.
        # Shuffle the DataFrame to randomize selection order
        centers_candidate = centers_candidate.sample(frac=1).reset_index(drop=True)

        # Select clusters until the sum of 'cnts' is close to the target
        val_clusters = []
        current_sum_val = 0

        for _, row in centers_candidate.iterrows():
            if current_sum_val + row['cnts'] > target_samples:
                # Check if this new row will exceed the target significantly
                if abs(target_samples - (current_sum_val + row['cnts'])) > abs(target_samples - current_sum_val):
                    break
            val_clusters.append(row['center'])
            current_sum_val += row['cnts']
            if current_sum_val >= target_samples:
                break
        print(f"The amount of val_set is {current_sum_val}")

        test_clusters = []
        current_sum_test = 0

        for _, row in centers_candidate.iterrows():
            if current_sum_test + row['cnts'] > target_samples:
                # Check if this new row will exceed the target significantly
                if abs(target_samples - (current_sum_test + row['cnts'])) > abs(target_samples - current_sum_test):
                    break
            if row['center'] not in val_clusters:
                test_clusters.append(row['center'])
                current_sum_test += row['cnts']
            if current_sum_test >= target_samples:
                break

        print(f"The amount of test_set is {current_sum_test}")

        val_samples = []
        test_samples = []
        for _, row in tab[(tab['cnts'] >= 10) & (tab['cnts'] <= 100)].iterrows():
            if row['center'] in val_clusters:
                val_samples.append(row['id'])
            if row['center'] in test_clusters:
                test_samples.append(row['id'])

        assert len(val_samples) == current_sum_val
        assert len(test_samples) == current_sum_test

        # Record the distribution of aatype in test_set
        aa_rec = {}
        tgt_case = []
        for item in test_samples:
            pdb_path = os.path.join(structure_dir, item, 'peptide.pdb')
            aa_tmp, _ = rec_aa(pdb_path)
            for aatype in aa_tmp:
                if aatype not in ["ZN", "NH2", "NA", "UNK"]:
                    if aatype in aa_rec.keys():
                        aa_rec[aatype] += 1
                    else:
                        aa_rec[aatype] = 1
                    if aatype in ['SEP', 'PTR', 'TYS']:
                        tgt_case.append(item)

        if 'SEP' in aa_rec.keys() and 'PTR' in aa_rec.keys() and 'TYS' in aa_rec.keys():
            if aa_rec['SEP'] > 10 and aa_rec['PTR'] > 5 and aa_rec['TYS'] > 10:
                break

    pdb.set_trace()

    train_samples = set(all_pdbs) - set(val_samples) - set(test_samples)
    train_samples = [item for item in train_samples]

    data_split = {'TRAIN': train_samples,
                  'VAL': val_samples,
                  'TEST': test_samples}
    with open(os.path.join(src_folder, 'data_split_20250101.json'), 'w') as f:
        json.dump(data_split, f, indent=4)
elif False:
    # Delete the similar case in test set
    tab = pd.read_csv("/ibex/user/guot0b/PepFlowww-NS3/data/Analysis_NS3_v3/clusterRes_cluster.tsv",
                      sep='\t', header=None)
    tab.columns = ['center', 'id']
    tab['cnts'] = tab.groupby('center')['id'].transform('count')
    tab.sort_values('cnts', ascending=False, inplace=True)

    cnts = tab.drop_duplicates('center')

    new_test_cases = []
    clusters = cnts['center'].values
    for cluster in clusters:
        cluster_size = cnts['cnts'][cnts['center']==cluster].values.item()
        if cluster_size > 5:
            candidates = random.sample(tab['id'][tab['center']==cluster].values.tolist(), 5)
            new_test_cases += candidates
        else:
            new_test_cases += tab['id'][tab['center']==cluster].values.tolist()

    data_split['TEST'] = new_test_cases
    pdb.set_trace()
    with open("./data/data_split_NS3_v3.json", 'w') as f:
        json.dump(data_split, f, indent=4)
elif False:
    # Modify the current dataset slightly.
    ratio = 0.1
    SEP_case = "4dat_B, 4iaf_S, 1jlu_S, 5ecg_F, 3ob2_A, 5ewz_D, 3p9y_E, 5em9_B, 5fc2_A, 1lf8_E, 6fby_D, 3wp1_A, " \
               "3q1i_E, 4aze_E, 5d3f_C, 2vxc_C, 3k0k_B, 6fi5_B, 3smo_P, 6fau_D, 5vkl_B, 5nwi_P, 7tbh_1_III_JA, " \
               "6faw_B, 5v4b_C, 7tbh_1_III_TA, 5lu2_C, 3nkx_P, 7tbh_1_III_AA, 3sqd_C, 3d9p_Z, 3g2v_C, 5nnd_D, 4yj4_B, " \
               "7tbh_1_III_DA, 6da1_C, 4cse_D, 6jxv_B, 5ieh_C, 5jej_C, 2c1j_C, 5jeo_B, 6bb4_P, 5lx2_B, 6fav_B, " \
               "6sad_C, 7tbh_1_III_VA, 6gkf_I, 6b8j_C, 4ib3_S, 3fqx_C, 6dat_F, 6m92_C, 2n8t_B, 5lvp_F, 7tbh_1_III_KA, " \
               "6gkg_J, 7tbh_1_III_Z, 4y2g_B, 3p1o_P, 1t2v_F, 7tbh_1_III_NA, 6mf5_C, 4iea_P, 6fby_B, 2ght_C, 4jiz_B, " \
               "7tbh_1_III_HA, 6g8p_P, 6m91_C, 5m36_C, 4psi_D, 1pua_B, 3p1s_P, 7tbh_1_III_MA, 4jlu_B, 6g6x_P, 3iqu_P, " \
               "5m37_C, 4y5i_F, 4ckt_D, 3k16_B, 4jfz_P, 3iqv_P, 3wp0_B, 6qsz_B, 3ual_P, 2lue_B, 2rt5_B, 6e4z_P, " \
               "6a5s_E, 3pxe_E, 6g8l_P, 6fbw_D, 3cu8_P, 6uyy_B, 5w94_E, 6k5t_B, 3fqu_C, 7tbh_1_III_BA, 2n1o_B, " \
               "1qja_Q, 4iay_S, 1y98_B, 3p1r_P, 5ypo_C, 6qzs_P, 4y32_C, 6m94_C, 6ghp_P, 4yoo_X, 5nwk_V, 3smk_P, " \
               "6s9k_B, 5n5w_P, 6e21_B, 4yos_E, 2w3o_C, 7tbh_1_III_Y, 6bd2_C, 2btp_P, 3d9k_Z, 1ujk_C, 4w5a_C, 6hep_E, " \
               "6fi4_B, 2b05_G, 5ods_E, 2azm_C, 7tbh_1_III_RA, 6gkg_I, 3k0h_B, 6g8k_P, 4wrq_C, 6qzs_C, 6k5r_B, " \
               "2c1n_C, 2wh0_Q, 3spr_P, 5n7g_C, 5n5t_P, 6f08_D, 7tbh_1_III_GA, 5sw9_B, 6uyv_B, 5zia_C, 5e2v_P, " \
               "6eih_P, 4ib1_S, 1qjb_Q, 5wum_B, 7tbh_1_III_IA, 6qtm_F, 5m35_C, 4ihl_P, 4fl5_P, 5yqg_F, 3bh8_C, " \
               "6rmm_P, 6fav_D, 5d3e_C, 3uat_B, 5nwj_P, 3p35_D, 5yqg_E, 3iqj_P, 6j9j_B, 1f8a_C, 6qk8_E, 4ifi_B, " \
               "3ubw_P, 4dau_B, 4hkc_B, 6jxx_B, 3d9m_Z, 5gmv_D, 4zq0_E, 5swf_B, 6npw_E, 6bd1_C, 3szm_I, 4j6s_E, " \
               "7tbh_1_III_FA, 4fj3_P, 1py1_E, 4igk_C, 6ef5_Q, 4y18_I, 3t7k_C, 3q4a_C, 3coj_H, 6uys_B, 5myc_P, " \
               "5d2d_C, 2br9_P, 4y3b_C, 6uyt_B, 1p16_C, 3sml_P, 5lu1_C, 5ema_B, 1ywt_C, 6uyu_B, 3bgm_C, 5e50_D, " \
               "5xw5_C, 4q94_D, 3k05_C, 7tbh_1_III_LA, 6ni2_V, 6h0e_I, 2lb0_B, 4iak_S, 3uzd_B, 2npm_X, 4iai_S, " \
               "6f09_A, 4bg6_Q, 1a37_P, 6a5q_E, 3l6f_C, 3v7d_E, 4iad_S, 6g8q_P, 5jel_B, 3ux0_P, 3fqr_C, 6fel_E, " \
               "6du2_C, 4iaz_S, 6m90_C, 4jxt_B, 6m93_C, 6fau_B, 3d9n_Y, 6f09_D, 6fbb_P, 6hm5_B, 3ueo_F, 5e2w_P, " \
               "5emb_B, 5n7f_C, 6qh7_Q, 4pz6_Q, 2ghq_C, 4fr3_P, 3huf_E, 4ib0_S, 4hpt_I, 6du3_C, 5hf3_B, 3sp5_P, " \
               "2l0i_B, 2laz_B, 4nm7_C, 5u6k_L, 5ulo_C, 6h06_I, 6bd1_D, 5m35_D, 5lxi_C, 2lay_B, 7tbh_1_III_UA, " \
               "5n5r_P, 2jkr_P, 4jqi_V, 3l41_B, 6faw_D, 5btv_P, 6b67_D, 1t15_B, 7tbh_1_III_CA, 7tbh_1_III_OA, 2lo6_B, " \
               "3axy_E, 6dc9_P, 3bhb_C, 3g2t_C, 6ejl_C, 4wjp_B, 6fcp_P, 3p1q_P, 3p1p_P, 6ttu_I, 3bh9_C, 4jg0_P, " \
               "1t29_B, 4no2_C, 2lax_B, 4wjn_B, 3d9l_Y, 4nnx_C, 6g8i_P, 2ruk_A, 4yk9_B, 5w4s_B, 5jek_C, " \
               "7tbh_1_III_PA, 7tbh_1_III_SA, 6fbw_B, 5dmg_P, 2n9e_A, 5m9d_B, 1p22_C, 6g8j_P, 3shv_C, 6f08_Q, 6eww_E, " \
               "2c63_P, 3u3z_B, 4u4a_D, 3p1n_P, 1ohe_B, 6iht_1_III_B, 3wdz_B, 5w7x_H, 3smm_P, 5k6s_B, 3k15_B, 4hpu_I, " \
               "6bnt_B, 2laj_B, 4x8n_B, 5my9_P, 6gkf_J, 7tbh_1_III_QA, 6mtu_D, 7tbh_1_III_EA, 3zkf_F, 2jkt_P, 4no3_C, " \
               "3o8i_B, 6ef5_P, 2c74_P, 2ovq_C, 5n75_P, 3smn_P, 5nas_C, 3mhr_P, 3t1n_C, 4o46_G"
    TYS_case = "1d3t_H, 1g1s_C, 4z64_P, 1d3d_H, 2zhf_I, 1d4p_H, 5ahg_I, 1no9_I, 1qj1_I, 1bcu_I, 3rm0_I, 1wbg_I, " \
               "2uuf_H, 2bvx_I, 1c1v_I, 1c1w_I, 2zhq_I, 1o5g_I, 1z71_B, 1xm1_H, 1ay6_I, 2zgb_I, 7kme_I, 2zi2_I, " \
               "4bak_D, 6i51_I, 3rly_I, 2zdv_I, 2zfr_I, 4uff_I, 1c4v_3, 1doj_B, 2zg0_I, 4bam_D, 1hgt_I, 3d49_I, " \
               "1ba8_C, 1a4w_I, 1awf_I, 3u98_I, 2pks_D, 1zgi_B, 1k21_I, 1ca8_C, 1twx_C, 1ypg_I, 1nrs_I, 1c4y_3, " \
               "3rmo_I, 3shc_I, 6e4z_P, 1tbz_I, 5mm6_I, 2c90_I, 1h8i_I, 1zrb_B, 4nze_I, 2mpm_B, 4ue7_I, 2ziq_I, " \
               "2c8y_I, 1aht_I, 4ufe_I, 3qx5_I, 1g30_C, 1c5l_I, 1nm6_B, 1ny2_3, 3utu_I, 2zhw_I, 3egk_I, 4e7r_I, " \
               "1mu6_C, 2uuj_H, 2zf0_I, 4loy_I, 4bah_D, 6eo8_I, 2zc9_I, 1o2g_I, 2c8z_I, 3u9a_I, 1ta2_B, 3rmn_I, " \
               "3p17_I, 1c1u_I, 2jh6_H, 3rm2_I, 2jh5_H, 1tmb_I, 3dux_I, 2zff_I, 3eq0_I, 1ghx_I, 2bvs_I, 6s6q_C, " \
               "1tom_I, 1ghy_I, 1g32_C, 1riw_D, 1zgv_B, 1mu8_C, 1c4u_3, 6fjt_I, 1gj5_I, 3qwc_I, 2znk_I, 1k22_I, " \
               "5lce_I, 1ypm_I, 1a2c_I, 1ad8_I, 2zgx_I, 1d6w_I, 1qj6_I, 1kts_C, 5jzy_I, 2uuk_H, 2k3u_B, 5jfd_I, " \
               "1vr1_I, 6rot_A, 4ufg_I, 1b5g_I, 1hdt_P, 1ae8_I, 1ypk_I, 2c8w_I, 1a46_I, 4ban_D, 1bb0_C, 1tmu_J, " \
               "1fpc_I, 1way_I, 1sl3_B, 1d3q_H, 3rml_I, 3hat_I, 5yy4_B, 3rlw_I, 1ta6_B, 4udw_I, 1hah_I, 1nt1_H, " \
               "1aix_I, 2zfp_I, 1t4v_I, 5a2m_I, 4lxb_I, 2bxu_I, 1ypj_I, 1ype_I, 1h8d_I, 5lpd_I, 2zo3_I, 1a61_I, " \
               "4ufd_I, 1gj4_I, 4z5w_Q, 3biv_I, 3rmm_I, 1a5g_I, 2c93_I, 1ai8_I, 3qtv_I, 3uwj_I, 1d3p_H, 1uma_I, " \
               "4z61_Q, 6hsx_I, 3qto_I, 8kme_3, 3t5f_I, 4bao_D, 2bdy_B, 1qbv_E, 4yes_H, 1c5o_I, 1ghw_I, 3sha_I, " \
               "2c8x_I, 1qj7_I, 4n3l_I, 5l6n_I, 3dhk_I, 2zhe_I, 1t4u_I, 2jh0_H, 3f68_I, 1ktt_C, 6eo9_I, 2bvr_I, " \
               "1hag_I, 5mjt_D, 1c5n_I, 4baq_D, 1ghv_I, 4z63_P, 6gbw_I, 2gde_D, 5af9_I, 3c1k_B, 3si4_I, 1d9i_I, " \
               "2zfq_I, 3si3_I, 1mue_C, 1sb1_I, 4ueh_I, 2zda_I, 3biu_I, 1afe_I, 5mls_D, 1ypl_I, 2bxt_I, 1qhr_I, " \
               "3po1_D, 3dt0_I, 3sv2_I"
    PTR_case = "1kc2_B, 3pfv_C, 1een_B, 3ob2_A, 1fyr_I, 3tu7_I, 3tl0_B, 1f1w_B, 2oq1_B, 1yrk_B, 1lkk_B, 1a81_H, " \
               "5njk_G, 1tze_I, 4a4b_B, 2q8y_B, 1ptt_B, 1ayb_P, 1ygr_C, 4k45_B, 4rh5_B, 1mw4_B, 1ka6_B, 2z8p_B, " \
               "2iuh_B, 5gji_B, 1uef_C, 4icz_F, 6r5g_B, 5cdw_f, 1j4k_P, 1fhr_P, 4roj_D, 1a1e_C, 2mrk_B, 4gvc_B, " \
               "1aou_P, 5hz1_A, 1fu5_B, 2g35_B, 1bmb_I, 6kc4_B, 3fxx_B, 3omh_E, 5hz0_A, 4a4c_B, 1m0v_B, 1g1h_B, " \
               "2lct_B, 1ju5_B, 3s8l_B, 1a1c_C, 3da9_D, 5u1m_B, 4rh9_B, 1r1p_E, 1shc_B, 6oi4_E, 3d44_B, 3op0_C, " \
               "1jyq_L, 6i5j_I, 2ci9_L, 6i5j_K, 2cia_L, 2pld_B, 6i5n_I, 3vrr_C, 1csz_B, 4qsy_B, 5cdw_I, 1aot_P, " \
               "1bm2_L, 2iui_C, 6e8k_B, 1ygu_C, 5hl0_B, 2eu0_B, 1ptu_B, 5czh_B, 3d42_B, 4ch8_P, 1sps_D, 1oo4_B, " \
               "3zni_B, 4gfu_F, 3ov1_B, 4rhg_B, 2lnw_B, 4p9v_B, 3vro_B, 4jmh_B, 4u1p_B, 5hkz_B, 1lcj_B, 4p9z_B, " \
               "2mfq_B, 5x94_L, 1d4w_C, 5c26_B, 1xxp_C, 5hky_B, 1a1b_C, 1j4x_D, 1hcs_A, 1r1s_B, 5gjh_B, 4jmg_B, " \
               "2rmx_B, 2hdx_G, 6pxc_U, 5o76_B, 4qum_B, 2cbl_B, 2vif_P, 3ove_B, 1a07_C, 3vrp_B, 2i6o_B, 1eeo_B, " \
               "1shb_B, 5df6_B, 7t1u_2_III_D, 1fpr_B, 1jyr_L, 2ple_B, 2hmh_B, 5aul_B, 1k2m_P, 1csy_B, 1i3z_B, 3mxc_L, " \
               "3bu6_B, 1g1f_B, 5hz3_A, 6roy_D, 1pic_B, 3ob1_A, 1h9o_B, 1ayc_P, 1aya_P, 5hyx_A, 2y1n_D, 6b67_D, " \
               "1hct_A, 1irs_B, 2etz_B, 6roz_B, 1p13_D, 4ch2_P, 3ml4_E, 2fci_B, 8dgo_1_III_C, 4zrt_B, 3n8m_B, 5tq1_B, " \
               "5njj_H, 3tkz_P, 6dm4_E, 2yu7_B, 4s0g_B, 2bbu_B, 1yvh_B, 3olr_E, 4xz1_B, 5czi_B, 3buo_A, 2ror_B, " \
               "1shd_B, 3n84_G, 3n7y_D, 2nmb_B, 1lkl_B, 5x7b_L, 1g1g_B, 1sha_B, 3plf_A, 3mxy_L, 1zfp_I, 5tqs_E, " \
               "4gfv_F, 3u3z_B, 1qg1_I, 6i5n_K, 1tce_B, 2h5k_C, 1x27_I, 1yvl_C, 1r1q_C, 3s8n_B, 4r3p_B, 4gl9_K, " \
               "2hwl_P, 4gpl_A, 3maz_B, 4nnd_F, 4r3r_B, 2l4k_B, 1fbv_B, 1lck_B, 6i4x_D, 6c5x_G, 3s8o_B, 2mc1_B"
    SEP_case = SEP_case.split(", ")
    SEP_case.remove('3ob2_A')
    SEP_case.remove('6b67_D')
    SEP_case.remove('3u3z_B')
    SEP_case.remove('6e4z_P')
    TYS_case = TYS_case.split(", ")
    PTR_case = PTR_case.split(", ")
    random.shuffle(SEP_case)
    random.shuffle(TYS_case)
    random.shuffle(PTR_case)
    NS3_pool = set(SEP_case) | set(TYS_case) | set(PTR_case)

    tgt_split = "/home/guot0b/disk3/guot0b/PepFlowww-main/data/PepMerge_release_analysis/data_split_20241218_v2.json"
    with open(tgt_split, 'r') as f:
        data_split = json.load(f)

    # Check the data first
    assert len(set(data_split['TRAIN']) & set(data_split['TEST'])) == 0
    assert len(set(data_split['TRAIN']) & set(data_split['VAL'])) == 0
    assert len(set(data_split['VAL']) & set(data_split['TEST'])) == 0
    print(f"The original size of TRAIN {len(data_split['TRAIN'])} |"
          f" VAL {len(data_split['VAL'])} |"
          f" TEST {len(data_split['TEST'])}")

    # Spare SEP cases to VAL_set
    SEP_in_test = set(SEP_case) & set(data_split['TEST'])  # 47
    SEP_in_val = set(SEP_case) & set(data_split['VAL'])  # 9
    SEP_in_train = set(SEP_case) & set(data_split['TRAIN'])  # 281
    # len(SEP_case)  # 337
    new_data_split_v1 = copy.deepcopy(data_split)

    num_rec = 0
    for item in SEP_in_test:
        new_data_split_v1['VAL'].append(item)
        new_data_split_v1['TEST'].remove(item)
        num_rec += 1
        if len(set(SEP_case) & set(new_data_split_v1['TEST'])) <= round(0.1 * len(set(SEP_case))) \
                or len(set(SEP_case) & set(new_data_split_v1['VAL'])) >= round(0.1 * len(set(SEP_case))):
            break

    pay_back = random.sample((set(new_data_split_v1['VAL']) - NS3_pool), num_rec)
    for item in pay_back:
        new_data_split_v1['TEST'].append(item)
        new_data_split_v1['VAL'].remove(item)

    num_rec = 0
    for item in SEP_in_train:
        new_data_split_v1['VAL'].append(item)
        new_data_split_v1['TRAIN'].remove(item)
        num_rec += 1
        if len(set(SEP_case) & set(new_data_split_v1['TRAIN'])) <= round(0.8 * len(set(SEP_case))) \
                or len(set(SEP_case) & set(new_data_split_v1['VAL'])) >= round(0.1 * len(set(SEP_case))):
            break

    pay_back = random.sample((set(new_data_split_v1['VAL']) - NS3_pool), num_rec)
    for item in pay_back:
        new_data_split_v1['TRAIN'].append(item)
        new_data_split_v1['VAL'].remove(item)

    # Spare TYS to val & test
    TYS_in_test = set(TYS_case) & set(new_data_split_v1['TEST'])  # 0
    TYS_in_val = set(TYS_case) & set(new_data_split_v1['VAL'])  # 0
    TYS_in_train = set(TYS_case) & set(new_data_split_v1['TRAIN'])  # 219

    num_rec = 0
    for item in TYS_in_train:
        new_data_split_v1['VAL'].append(item)
        new_data_split_v1['TRAIN'].remove(item)
        num_rec += 1
        if len(set(TYS_case) & set(new_data_split_v1['TRAIN'])) <= round(0.8 * len(set(TYS_case))) \
                or len(set(TYS_case) & set(new_data_split_v1['VAL'])) >= round(0.1 * len(set(TYS_case))):
            break

    pay_back = random.sample((set(new_data_split_v1['VAL']) - NS3_pool), num_rec)
    for item in pay_back:
        new_data_split_v1['TRAIN'].append(item)
        new_data_split_v1['VAL'].remove(item)

    num_rec = 0
    for item in (TYS_in_train - set(new_data_split_v1['VAL'])):
        new_data_split_v1['TEST'].append(item)
        new_data_split_v1['TRAIN'].remove(item)
        num_rec += 1
        if len(set(TYS_case) & set(new_data_split_v1['TRAIN'])) <= round(0.8 * len(set(TYS_case))) \
                or len(set(TYS_case) & set(new_data_split_v1['TEST'])) >= round(0.1 * len(set(TYS_case))):
            break

    pay_back = random.sample((set(new_data_split_v1['TEST']) - NS3_pool), num_rec)
    for item in pay_back:
        new_data_split_v1['TRAIN'].append(item)
        new_data_split_v1['TEST'].remove(item)

    # Don't need to Spare PTR
    PTR_in_test = set(PTR_case) & set(new_data_split_v1['TEST'])  # 28
    PTR_in_val = set(PTR_case) & set(new_data_split_v1['VAL'])  # 23
    PTR_in_train = set(PTR_case) & set(new_data_split_v1['TRAIN'])  # 153

    # Check the data finally
    assert len(set(new_data_split_v1['TRAIN']) & set(new_data_split_v1['TEST'])) == 0
    assert len(set(new_data_split_v1['TRAIN']) & set(new_data_split_v1['VAL'])) == 0
    assert len(set(new_data_split_v1['VAL']) & set(new_data_split_v1['TEST'])) == 0
    print(f"The new size of TRAIN {len(new_data_split_v1['TRAIN'])} |"
          f" VAL {len(new_data_split_v1['VAL'])} |"
          f" TEST {len(new_data_split_v1['TEST'])}")
    print(f"The SEP cases between 3 sets are:"
          f" {len(set(SEP_case) & set(new_data_split_v1['TRAIN']))}:"
          f"{len(set(SEP_case) & set(new_data_split_v1['VAL']))}:"
          f"{len(set(SEP_case) & set(new_data_split_v1['TEST']))}")
    print(f"The TYS cases between 3 sets are:"
          f" {len(set(TYS_case) & set(new_data_split_v1['TRAIN']))}:"
          f"{len(set(TYS_case) & set(new_data_split_v1['VAL']))}:"
          f"{len(set(TYS_case) & set(new_data_split_v1['TEST']))}")
    print(f"The PTR cases between 3 sets are:"
          f" {len(set(PTR_case) & set(new_data_split_v1['TRAIN']))}:"
          f"{len(set(PTR_case) & set(new_data_split_v1['VAL']))}:"
          f"{len(set(PTR_case) & set(new_data_split_v1['TEST']))}")

    with open("/home/guot0b/disk3/guot0b/PepFlowww-main/data/PepMerge_release_analysis/data_split_NS3.json", 'w') as f:
        json.dump(new_data_split_v1, f, indent=4)
elif False:
    # Calculate the weight_dict of NS3 dataset
    aa_rec_file = "/home/guot0b/disk3/guot0b/PepFlowww-main/data/PepMerge_release_analysis/aa_rec_NS3.json"
    with open(aa_rec_file, 'r') as f:
        aa_rec = json.load(f)
    train_set = aa_rec['TRAIN']
    pdb.set_trace()
    total_num = sum([value for _, value in train_set.items()])
    density_dict = {key: (value / total_num) for key, value in train_set.items()}
    weight_dict = {key: round(0.01 / value, 3) for key, value in density_dict.items()}
elif False:
    # Check the aatype of generated samples. Select a good case for visualization.
    from pepflow.modules.protein.parsers import get_seq_from_pdb
    from models_con.pep_dataloader import preprocess_structure_v2, preprocess_structure_v3
    N3_case_file = "/ibex/user/guot0b/PepFlowww-NS3/data/NS3_case.txt"
    N3_cases = []
    with open(N3_case_file, 'r') as f:
        for line in f:
            N3_cases.append(line.strip())

    src_dir = "/ibex/user/guot0b/PepFlowww-NS3/logs/2024.12.23_v1_pepflow_NS3_640k/test_640k_num16/"
    case_list = os.listdir(src_dir)
    aa_rec = {}
    NS3_gen_case = {'SEP': [], 'TYS': [], 'PTR': []}
    for case in case_list:
        case_dir = os.path.join(src_dir, case)
        if not os.path.isdir(case_dir):
            continue
        item_list = os.listdir(case_dir)
        for item in item_list:
            if item.endswith(".pdb"):
                item_path = os.path.join(case_dir, item)
                seq_item = get_seq_from_pdb(item_path)
                peptide_seq = seq_item["A"][:-1].split(" ")
                for aa in peptide_seq:
                    if aa not in aa_rec.keys():
                        aa_rec[aa] = 1
                    else:
                        aa_rec[aa] += 1
                    if aa in NS3_gen_case.keys() and case not in N3_cases:
                        try:
                            os.listdir(os.path.join(structure_dir, case))
                            gt_con = torch.load(os.path.join(structure_dir, case, f"{case}_re.pt"))
                            gt_pos = gt_con['pos_heavyatom'][torch.where(gt_con['generate_mask'])]

                            gen_con = preprocess_structure_v3(item_path)
                            gen_pos = gen_con['pep_pos']

                            binder_rmsd = torch.linalg.norm(gen_pos[:, 1] - gt_pos[:, 1], dim=-1).mean()
                            if binder_rmsd < 2:
                                NS3_gen_case[aa].append(item_path)
                                pdb.set_trace()
                        except ValueError:
                            pass
    for aa in NS3_gen_case.keys():
        NS3_gen_case[aa] = set(NS3_gen_case[aa])
elif False:
    # Split the train/val set based on the training set of PepFlow (Nosc)
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1


    src_folder = "/ibex/user/guot0b/PepFlowww-NS3/data/PepMerge_lmdb/"

    all_pdbs = os.listdir(structure_dir)
    """
    

    if not os.path.exists(src_folder):
        os.mkdir(src_folder)
    seq_file = os.path.join(src_folder, 'train_seqs_PepMerge_release.fasta')

    with open(seq_file, 'w') as f:
        for pdb in all_pdbs:
            if pdb not in test_items:
                fasta = get_fasta_from_pdb(os.path.join(structure_dir, pdb, 'receptor.pdb'))
                f.write(f'>{pdb}\n')
                f.write(fasta + '\n')

    # $ mmseqs easy-cluster seqs.fasta clusterRes tmp --min-seq-id 0.4 -c 0.8 --cov-mode 1
    """
    tab = pd.read_csv("/ibex/user/guot0b/PepFlowww-NS3/data/PepMerge_lmdb/clusterRes_cluster.tsv",
                      sep='\t', header=None)
    tab.columns = ['center', 'id']
    tab['cnts'] = tab.groupby('center')['id'].transform('count')
    tab.sort_values('cnts', ascending=False, inplace=True)

    cnts = tab.drop_duplicates('center')
    # tab.to_csv("/home/guot0b/disk3/guot0b/PepFlowww-main/data/PepMerge_release_analysis/item_cluster.csv", index=None)
    # cnts.to_csv("/home/guot0b/disk3/guot0b/PepFlowww-main/data/PepMerge_release_analysis/center.csv", index=None)
    # centers_candidate = set((cnts[(cnts['cnts'] >= 10) & (cnts['cnts'] <= 100)])['center'])
    centers_candidate = cnts[(cnts['cnts'] >= 10) & (cnts['cnts'] <= 100)]

    target_samples = len(test_items)

    # Shuffle the DataFrame to randomize selection order
    centers_candidate = centers_candidate.sample(frac=1).reset_index(drop=True)

    # Select clusters until the sum of 'cnts' is close to the target
    val_clusters = []
    current_sum_val = 0

    for _, row in centers_candidate.iterrows():
        if current_sum_val + row['cnts'] > target_samples:
            # Check if this new row will exceed the target significantly
            if abs(target_samples - (current_sum_val + row['cnts'])) > abs(target_samples - current_sum_val):
                break
        val_clusters.append(row['center'])
        current_sum_val += row['cnts']
        if current_sum_val >= target_samples:
            break
    print(f"The amount of val_set is {current_sum_val}")

    val_samples = []
    for _, row in tab[(tab['cnts'] >= 10) & (tab['cnts'] <= 100)].iterrows():
        if row['center'] in val_clusters:
            val_samples.append(row['id'])

    assert len(val_samples) == current_sum_val

    train_samples = set(all_pdbs) - set(val_samples) - set(test_items)
    train_samples = [item for item in train_samples]

    data_split = {'TRAIN': train_samples,
                  'VAL': val_samples,
                  'TEST': test_items}
    with open(os.path.join(src_folder, 'data_split_Pepflow_train_val_test.json'), 'w') as f:
        json.dump(data_split, f, indent=4)
elif False:
    # Calculate the class weights of BLV
    aa_rec_file = "/ibex/user/guot0b/PepFlowww-NS3/data/PepMerge_lmdb/NoscPep_aa_rec.pt"
    aa_rec = torch.load(aa_rec_file)

    total_num = 0
    for _, value in aa_rec.items():
        total_num += value

    c_dict = {}
    for key, value in aa_rec.items():
        c_dict[key] = np.log(total_num / value).item()

    c_max_key = max(c_dict, key=c_dict.get)
    c_max = c_dict[c_max_key]

    BLV_weight = {}
    for key, _ in aa_rec.items():
        BLV_weight[key] = c_dict[key] / c_max

    # Make into list with specific order
    import pepflow.modules.protein.constants as constants
    BLV_weight_list = []
    for res_idx in range(constants.PAD_RESIDUE_INDEX - 1):
        BLV_weight_list.append(round(BLV_weight[str(constants.AA(res_idx))], 3))
elif False:
    # Calculate the energy of test samples in advance
    from models_con.pep_dataloader import preprocess_structure_v2
    from pepflow.modules.protein.writers import save_pdb


    for case in test_items:
        print(f"Calculating the case {case}...")
        case_dir = os.path.join(structure_dir, case)
        energy_save = os.path.join(case_dir, "energy_pocket_merge_NS3.pt")

        if os.path.exists(energy_save):
            print(f"Energy of {case} has been calculated. Pass it.")
            continue

        pt_file = os.path.join(case_dir, f"{case}_re_NS3.pt")
        if not os.path.exists(pt_file):
            preprocess_structure_v2(case_dir, pt_file, renumber_chain=True)
        try:
            pt_con = torch.load(pt_file)
        except FileNotFoundError:
            continue
        pt_to_pdb = os.path.join(case_dir, "pocket_merge_from_pt_NS3.pdb")
        if not os.path.exists(pt_to_pdb):
            save_pdb(pt_con, path=pt_to_pdb)

        # binder_chain = case.split("_")[-1]
        binder_chain = "A"

        gt_energy_dict = get_rosetta_score_base(pt_to_pdb, chain_id=binder_chain)
        gt_stab = round(gt_energy_dict['stab'], 3)
        gt_aff = round(gt_energy_dict['aff'], 3)
        print(f"Case: {case} with stability {gt_stab} and affinity {gt_aff}")
        energy = {'stab': gt_stab, 'aff': gt_aff}

        torch.save(energy, energy_save)
elif False:
    # Filter the test set
    new_test_items = []
    discard_items = []
    for case in test_items:
        case_dir = os.path.join(structure_dir, case)
        tmp_path = f"tmp_{random.randint(1, 1e4)}.pt"

        preprocess_structure_v2(case_dir, tmp_path, renumber_chain=True)
        if os.path.exists(tmp_path):
            protein = torch.load(tmp_path)
            pep_abs_idx = protein['resseq'][protein['generate_mask']]
            broken_check = pep_abs_idx[1:] - pep_abs_idx[:-1]
            if max(broken_check) >= 3:
                discard_items.append(case)
            else:
                new_test_items.append(case)
            os.remove(tmp_path)
        else:
            discard_items.append(case)
    assert len(new_test_items) + len(discard_items) == len(test_items), "The final number is not correct."
    data_split['TEST'] = new_test_items
    with open('./data/data_split_NS3_v2.json', 'w') as f:
        json.dump(data_split, f, indent=4)
elif False:
    # Filter the current test result
    src_path = "/ibex/user/guot0b/PepFlowww-NS3/logs/2025.1.1_v3_pepflow_NS3_BLV_0.5_320k/test_320k_num16/eval_recovery.csv"
    tgt_path = "/ibex/user/guot0b/PepFlowww-NS3/logs/2025.1.1_v3_pepflow_NS3_BLV_0.5_320k/test_320k_num16/eval_recovery_v2.csv"

    csv_con = pd.read_csv(src_path)
    mask = csv_con['Case'] == test_items[0]
    for idx in range(1, len(test_items)):
        mask = mask | (csv_con['Case'] == test_items[idx])
    csv_con[mask].to_csv(tgt_path, index=False)
elif False:
    # Split another testset that every case contains NS3
    N3_case_file = "/ibex/user/guot0b/PepFlowww-NS3/data/NS3_case.txt"
    N3_cases = []
    with open(N3_case_file, 'r') as f:
        for line in f:
            N3_cases.append(line.strip())

    NS3_test_items = set(test_items) & set(N3_cases)
    NS3_test_save = {'TRAIN':[], 'VAL':[], 'TEST':[case for case in NS3_test_items]}
    with open("/ibex/user/guot0b/PepFlowww-NS3/data/data_split_NS3_v4.json", 'w') as f:
        json.dump(NS3_test_save, f, indent=4)
elif False:
    # Calculate the recovery metrics of NS3_test_items from a general one.
    fold = 4
    csv_path = "/ibex/user/guot0b/PepFlowww-NS3/logs/2025.03.28_v2_NS_pep_no_GSP_v2_320k/test_320k_num16/eval_recovery.csv"
    # csv_path = f"/ibex/user/guot0b/PepFlowww-NS3/logs/2025.01.16_v2_NS_pep_BLV15_320k/test_320k_num16_fold_{fold}/eval_recovery.csv"
    save_path = os.path.join(csv_path.split("eval_recovery.csv")[0], "eval_recovery_NS3.csv")
    csv_con = pd.read_csv(csv_path)
    # with open(f"/ibex/user/guot0b/PepFlowww-NS3/data/Dataset_NS3/data_split_test_fold_{fold}_NS3.json", 'r') as f:
    #     data_split = json.load(f)
    # NS3_test_items = data_split['TEST']    
    NS3_csv = csv_con[csv_con['Case'].isin(NS3_test_items)].copy()

    gen_SEP_all, gt_SEP_all, gen_TYS_all, gt_TYS_all, \
        gen_PTR_all, gt_PTR_all, gen_NS3_all, gt_NS3_all \
        = 0, 0, 0, 0, 0, 0, 0, 0,
    if 'rec_seq(NS3)' in NS3_csv.keys():
        for case in NS3_csv['Case'].values:
            case_data = NS3_csv[NS3_csv['Case'] == case]
            # pdb.set_trace()
            gen_SEP_all += int(case_data['rec_seq(SEP)'].values.item().split('_')[0])
            gt_SEP_all += int(case_data['rec_seq(SEP)'].values.item().split('_')[1])

            gen_TYS_all += int(case_data['rec_seq(TYS)'].values.item().split('_')[0])
            gt_TYS_all += int(case_data['rec_seq(TYS)'].values.item().split('_')[1])

            gen_PTR_all += int(case_data['rec_seq(PTR)'].values.item().split('_')[0])
            gt_PTR_all += int(case_data['rec_seq(PTR)'].values.item().split('_')[1])

            gen_NS3_all += int(case_data['rec_seq(NS3)'].values.item().split('_')[0])
            gt_NS3_all += int(case_data['rec_seq(NS3)'].values.item().split('_')[1])

    temp_df = pd.DataFrame({
        'Case': ['All'],
        'rec_pos': [NS3_csv['rec_pos'].mean()],
        'rec_seq': [NS3_csv['rec_seq'].mean()],
        'rec_ss': [NS3_csv['rec_ss'].mean()],
        'rec_bs': [NS3_csv['rec_bs'].mean()],
        'diversity': [NS3_csv['diversity'].mean()],
        'Clash': [NS3_csv['Clash'].mean()],
        'rec_seq(SEP)': [f"{gen_SEP_all}_{gt_SEP_all}"],
        'rec_seq(TYS)': [f"{gen_TYS_all}_{gt_TYS_all}"],
        'rec_seq(PTR)': [f"{gen_PTR_all}_{gt_PTR_all}"],
        'rec_seq(NS3)': [f"{gen_NS3_all}_{gt_NS3_all}"],
    })

    NS3_csv = pd.concat([NS3_csv, temp_df])
    NS3_csv.to_csv(save_path, index=False)
elif False:
    # Calculate the Energy metrics of NS3_test_items from a general one.
    csv_path = "/ibex/user/guot0b/PepFlowww-NS3/logs/2025.01.03_v1_pepflow_NS3_seesaw_0.8_2.0_320k/test_320k_num16/eval_energy.csv"
    save_path = os.path.join(csv_path.split("eval_energy.csv")[0], "eval_energy_NS3.csv")
    csv_con = pd.read_csv(csv_path)

    NS3_csv = csv_con[csv_con['Case'].isin(NS3_test_items)].copy()

    temp_df = pd.DataFrame({
        'Case': ['All'],
        'Stability': [NS3_csv['Stability'].mean()],
        'Affinity': [NS3_csv['Affinity'].mean()],
        'GT_stab': [NS3_csv['GT_stab'].mean()],
        'Gen_stab': [NS3_csv['Gen_stab'].mean()],
        'GT_aff': [NS3_csv['GT_aff'].mean()],
        'Gen_aff': [NS3_csv['Gen_aff'].mean()],
    })

    NS3_csv = pd.concat([NS3_csv, temp_df])
    NS3_csv.to_csv(save_path, index=False)
elif False:
    # Investigate the impact of NS3 substitution in terms of testset
    # We investigate on general test set and NSAA test set
    NS3_aff_list_general = []
    NS3_aff_list_NSAA = []

    sub_aff_list_general = []
    sub_aff_list_NSAA = []

    for case in test_items:
        case_dir = os.path.join(structure_dir, case)
        NS3_energy_file = os.path.join(case_dir, "energy_pocket_merge_NS3.pt")
        sub_energy_file = os.path.join(case_dir, "energy_pocket_merge.pt")
        if case not in NS3_test_items:
            # There is no difference between NS3 and sub.
            affinity = torch.load(NS3_energy_file)['aff']
            NS3_aff_list_general.append(affinity)
            sub_aff_list_general.append(affinity)
        elif case in NS3_test_items:
            NS3_affinity = torch.load(NS3_energy_file)['aff']
            NS3_aff_list_general.append(NS3_affinity)
            NS3_aff_list_NSAA.append(NS3_affinity)

            sub_affinity = torch.load(sub_energy_file)['aff']
            sub_aff_list_general.append(sub_affinity)
            sub_aff_list_NSAA.append(sub_affinity)

    assert len(NS3_aff_list_general) == len(sub_aff_list_general), 'No!'
    assert len(NS3_aff_list_NSAA) == len(sub_aff_list_NSAA), 'No!'
elif False:
    # Compare the energy between gen_sample with NS3 and gt_sample
    tgt_dir = "/ibex/user/guot0b/PepFlowww-NS3/logs/2025.01.06_v1_pepflow_NS3_BLV_10_acc_weight_320k/test_320k_num16/"
    con = os.listdir(tgt_dir)

    gen_NS3_save_path = os.path.join(tgt_dir, "NS3_sample.pt")
    if not os.path.exists(gen_NS3_save_path):
        gen_sample_with_NS3 = []
        for case in test_items:
            print(f"Calculating {case} now ...")
            case_dir = os.path.join(tgt_dir, case)
            sample_list = os.listdir(case_dir)
            for sample in sample_list:
                sample_path = os.path.join(case_dir, sample)
                rec = rec_aa(sample_path, chain_id='A')
                aatypes = rec[0]

                for item in aatypes:
                    if item in ['SEP', 'TYS', 'PTR']:
                        gen_sample_with_NS3.append(sample_path)
                        break
        torch.save(gen_sample_with_NS3, gen_NS3_save_path)
    else:
        gen_sample_with_NS3 = torch.load(gen_NS3_save_path)

    comparison_rec = []
    for item in gen_sample_with_NS3:
        case = item.split('/')[-2]

        gt_energy_path = os.path.join(structure_dir, case, 'energy_pocket_merge_NS3.pt')
        gt_energy = torch.load(gt_energy_path)
        gt_aff = gt_energy['aff']

        gen_energy = get_rosetta_score_base(item, chain_id='A')
        gen_aff = gen_energy['aff']

        comparison_rec.append(gen_aff < gt_aff)

    # Save the result to the file gen_NS3_save_path
    res = {"NS3_sample": gen_sample_with_NS3, "Aff_success": f"{sum(comparison_rec)}/{len(comparison_rec)}"}
    torch.save(res, gen_NS3_save_path)
    pdb.set_trace()
elif False:
    # Plot the interaction figure
    from models_con.pep_dataloader import preprocess_structure
    task = {'id': '1a81_H', 'pdb_path': "/ibex/user/guot0b/DataPool/Pepflow/1a81_H/"}
    preprocess_structure(task, True)
elif False:
    # Deal with the result of RFD_recovery
    rec_con = pd.read_csv("/ibex/user/guot0b/PepFlowww-ibex/comparison_exp/ProteinGenerator_eval/eval_recovery.csv")

    valid_mask = ~rec_con['rec_pos'].isna()
    pos_avg = rec_con[valid_mask]['rec_pos'].mean()
    seq_avg = rec_con[valid_mask]['rec_seq'].mean()
    ss_avg = rec_con[valid_mask]['rec_ss'].mean()
    bs_avg = rec_con[valid_mask]['rec_bs'].mean()
    div_avg = rec_con[valid_mask]['diversity'].mean()
    print(f"The avg of [pos, seq, ss, bs, div] are {pos_avg}, {seq_avg}, {ss_avg}, {bs_avg}, {div_avg}")
elif False:
    # Figure of energy distribution (Pepflow, NS-Pep, NS-Pep w/o NS3, reference)
    src_pepflow = "/ibex/user/guot0b/PepFlowww-ibex/logs_lr1/2024.12.22_v1_pepflow_640k/test_320k_num16/eval_energy_NS3.csv"
    src_nspep = "/ibex/user/guot0b/PepFlowww-NS3/logs/2025.01.13_v2_NS_pep_SC0.1_rBLV15_320k/test_320k_num16/eval_energy_NS3.csv"
    src_nspep_sub = "/ibex/user/guot0b/PepFlowww-ibex/comparison_exp/NS-Pep_eval(Replace NS3)/eval_energy.csv"

    # Collect the data
    ref_aff_list = []

    for case in NS3_test_items:
        # Reference
        if case == '3p35_D':  # Problematic case
            continue
        case_dir = os.path.join(structure_dir, case)
        energy_save = os.path.join(case_dir, "energy_pocket_merge_NS3.pt")
        ref_con = torch.load(energy_save)
        ref_aff_list.append(ref_con['aff'])

    # Pepflow
    pep_con = pd.read_csv(src_pepflow)
    pepflow_aff_list = pep_con['Gen_aff'][pep_con['Case'] != 'All'].values.tolist()

    # NS-Pep
    NSPep_con = pd.read_csv(src_nspep)
    NSPep_aff_list = NSPep_con['Gen_aff'][NSPep_con['Case'] != 'All'].values.tolist()

    # NS-Pep (sub)
    NSPep_sub_con = pd.read_csv(src_nspep_sub)
    NSPep_sub_aff_list = NSPep_sub_con['Gen_aff'][NSPep_sub_con['Case'] != 'All'].values.tolist()

    # Plot KDE distributions
    # Calculate medians
    pepflow_mean = np.mean(pepflow_aff_list)
    NSPep_sub_mean = np.mean(NSPep_sub_aff_list)
    NSPep_mean = np.mean(NSPep_aff_list)
    ref_mean = np.mean(ref_aff_list)

    # Create figure and subplots
    fig, axes = plt.subplots(nrows=4, ncols=1,
                             sharex=True,
                             figsize=(6, 10))

    # Plot Pepflow KDE
    sns.kdeplot(pepflow_aff_list, fill=True, ax=axes[0], color='C0')
    axes[0].axvline(pepflow_mean, linestyle='--', color='black')  # vertical dashed line at median
    axes[0].set_title('Pepflow')
    axes[0].set_ylabel('Density')

    # Plot NSPep_sub KDE
    sns.kdeplot(NSPep_sub_aff_list, fill=True, ax=axes[1], color='C1')
    axes[1].axvline(NSPep_sub_mean, linestyle='--', color='black')
    axes[1].set_title('NSPep_sub')
    axes[1].set_ylabel('Density')

    # Plot NSPep KDE
    sns.kdeplot(NSPep_aff_list, fill=True, ax=axes[2], color='C2')
    axes[2].axvline(NSPep_mean, linestyle='--', color='black')
    axes[2].set_title('NSPep')
    axes[2].set_ylabel('Density')

    # Plot ref KDE
    sns.kdeplot(ref_aff_list, fill=True, ax=axes[3], color='C3')
    axes[3].axvline(ref_mean, linestyle='--', color='black')
    axes[3].set_title('ref')
    axes[3].set_ylabel('Density')
    axes[3].set_xlabel('Gen_aff')

    plt.tight_layout()
    print(pepflow_mean, NSPep_sub_mean, NSPep_mean, ref_mean)
    plt.savefig('tmp.png')

    # pdb.set_trace()
elif False:
    # Find severl cases not in NS3_test_items but with NS3 and high affinity
    src_dir = "/ibex/user/guot0b/PepFlowww-NS3/logs/2025.01.17_v2_NS_pep_SC0.1_rBLV15_640k/test_320k_num16/"
    candidate = []
    tgt_path = "/ibex/user/guot0b/PepFlowww-NS3/output/CaseStudy_Exp4.2/candidate_list_640k.pt"

    NS3 = set(['SEP', 'TYS', 'PYR'])

    for case in test_items:
        if case not in NS3_test_items:
            case_gen_dir = os.path.join(src_dir, case)
            case_gt_dir = os.path.join(structure_dir, case)

            for gen_item in os.listdir(case_gen_dir):
                if gen_item.endswith(".pdb"):
                    gen_pdb = os.path.join(case_gen_dir, gen_item)

                    # Find the sample with NS3
                    aa_rec, _ = rec_aa(gen_pdb, chain_id='A')

                    flag = len(set(aa_rec) & NS3) > 0
                    if flag:
                        # Compare affinity with the GT
                        gt_energy_file = os.path.join(case_gt_dir, "energy_pocket_merge_NS3.pt")
                        gt_aff = torch.load(gt_energy_file)['aff']

                        gen_energy = get_rosetta_score_base(gen_pdb, chain_id='A')
                        gen_aff = gen_energy['aff']

                        if gen_aff < gt_aff:
                            print(f"Case {gen_pdb} with {gen_aff} < {gt_aff}")
                            candidate.append(gen_pdb)
                        else:
                            print(f"Case {gen_pdb} with {gen_aff} > {gt_aff}")
    torch.save(candidate, tgt_path)
elif False:
    # Plot the bar chart of AAR between different long-tailed method.
    # PepFlow(NS3), PepFlow(NS3) + WS, PepFlow(NS3) + BLV, PepFlow(NS3) + Seesaw, PepFlow(NS3) + RLM
    # AAR_SEP = [0, 0, 0, 0, 0,]
    # AAR_TYS = [6, 1, 7, 4, 48,]
    # AAR_PTR = [1, 15, 0, 4, 16,]
    # AAR_NS3 = [7, 16, 7, 8, 64,]

    # Remaining metrics
    AAR_TYS = [6, 1, 7, 1, 48]
    AAR_PTR = [1, 15, 0, 0, 16]
    AAR_NS3 = [7, 16, 7, 1, 64]

    # Metrics of NS_Pep w/ X, X \in {WS, BLV, Seesaw, RLM}
    # AAR_SEP = [3, 2, 0, 1]
    # AAR_TYS = [8, 69, 0, 72]
    # AAR_PTR = [31, 13, 0, 27]
    # AAR_NS3 = [42, 84, 0, 100]

    # Labels for the x-axis (now we only have 3 metrics)
    metrics = ["TYS", "PTR", "NS3"]

    # Labels for the legend (the 5 methods)
    methods = [
        "PepFlow(NS3)",
        "PepFlow(NS3)+WS",
        "PepFlow(NS3)+BLV",
        "PepFlow(NS3)+Seesaw",
        "PepFlow(NS3)+RLM",
    ]

    # Create an array for the 3 metrics
    x = np.arange(len(metrics))  # -> [0, 1, 2]
    width = 0.15  # width of each bar
    # plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [
        "#2E86C1",  # Blue-ish
        "#58D68D",  # Green-ish
        "#AF7AC5",  # Purple-ish
        "#F5B041",  # Orange-ish
        "#EC7063",  # Reddish
    ]

    x = np.arange(len(metrics))  # [0, 1, 2]
    width = 0.15

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot each method using a distinct color
    ax.bar(x - 2 * width, [AAR_TYS[0], AAR_PTR[0], AAR_NS3[0]],
           width, label=methods[0], color='#96CAC1')
    ax.bar(x - width, [AAR_TYS[1], AAR_PTR[1], AAR_NS3[1]],
           width, label=methods[1], color='#F6F6BC')
    ax.bar(x, [AAR_TYS[2], AAR_PTR[2], AAR_NS3[2]],
           width, label=methods[2], color='#C1BED6')
    ax.bar(x + width, [AAR_TYS[3], AAR_PTR[3], AAR_NS3[3]],
           width, label=methods[3], color='#EA8E83')
    ax.bar(x + 2 * width, [AAR_TYS[4], AAR_PTR[4], AAR_NS3[4]],
           width, label=methods[4], color='#8AAFC9')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('AAR')
    ax.set_title('AAR Comparison Across Long-Tailed Methods (Discarding AAR_SEP)')
    ax.legend()
    plt.grid('on', ls='--', alpha=0.66)

    plt.tight_layout()
    plt.savefig("aar_comparison_no_SEP_v4(seesaw0.8_2.0).png")
elif False:
    # Plot the bar chart of AAR between different long-tailed method.

    # Metrics of NS_Pep w/ X, X \in {WS, BLV, Seesaw, RLM}
    AAR_SEP = [3, 2, 0, 1]
    AAR_TYS = [8, 69, 70, 72]
    AAR_PTR = [31, 13, 21, 27]
    AAR_NS3 = [42, 84, 91, 100]

    # Labels for the x-axis (now we only have 3 metrics)
    metrics = ["SEP", "TYS", "PTR", "NS3"]

    # Labels for the legend (the 5 methods)
    methods = [
        "NS_Pep w/ WS",
        "NS_Pep w/ BLV",
        "NS_Pep w/ Seesaw",
        "NS_Pep w/ RLM",
    ]

    # Create an array for the 3 metrics
    x = np.arange(len(metrics))  # -> [0, 1, 2]
    width = 0.15  # width of each bar
    # plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [
        "#2E86C1",  # Blue-ish
        "#58D68D",  # Green-ish
        "#AF7AC5",  # Purple-ish
        "#F5B041",  # Orange-ish
        "#EC7063",  # Reddish
    ]

    x = np.arange(len(metrics))  # [0, 1, 2]
    width = 0.15

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot each method using a distinct color
    ax.bar(x - 2 * width, [AAR_SEP[0], AAR_TYS[0], AAR_PTR[0], AAR_NS3[0]],
           width, label=methods[0], color='#F6F6BC')
    ax.bar(x - width, [AAR_SEP[1], AAR_TYS[1], AAR_PTR[1], AAR_NS3[1]],
           width, label=methods[1], color='#C1BED6')
    ax.bar(x, [AAR_SEP[2], AAR_TYS[2], AAR_PTR[2], AAR_NS3[2]],
           width, label=methods[2], color='#EA8E83')
    ax.bar(x + width, [AAR_SEP[3], AAR_TYS[3], AAR_PTR[3], AAR_NS3[3]],
           width, label=methods[3], color='#8AAFC9')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('AAR')
    ax.set_title('AAR Comparison Across Long-Tailed Methods (Discarding AAR_SEP)')
    ax.legend()
    plt.grid('on', ls='--', alpha=0.66)

    plt.tight_layout()
    plt.savefig("aar_comparison_NS_Pep.png")
elif False:
    from Bio.SeqUtils import seq1
    # Find cases with NS3 and with low scRMSD
    scRMSD_rec = "/ibex/user/guot0b/PepFlowww-NS3/logs/2025.01.13_v2_NS_pep_SC0.1_rBLV15_320k/test_320k_num16/eval_scRMSD.csv"
    src_dir = "/ibex/user/guot0b/PepFlowww-NS3/logs/2025.01.13_v2_NS_pep_SC0.1_rBLV15_320k/test_320k_num16/"
    sc_con = pd.read_csv(scRMSD_rec)
    low_scRMSD_list = sc_con['Case'][sc_con['scRMSD']<6].values.tolist()
    candidate_v1 = set(low_scRMSD_list) & set(NS3_test_items)
    candidate_v2 = []
    for case in candidate_v1:
        case_gen_dir = os.path.join(src_dir, case)
        case_gt_dir = os.path.join(structure_dir, case)

        for gen_item in os.listdir(case_gen_dir):
            if gen_item.endswith(".pdb"):
                gen_pdb = os.path.join(case_gen_dir, gen_item)

                # Compare affinity with the GT
                gt_energy_file = os.path.join(case_gt_dir, "energy_pocket_merge_NS3.pt")
                gt_aff = torch.load(gt_energy_file)['aff']

                gen_energy = get_rosetta_score_base(gen_pdb, chain_id='A')
                gen_aff = gen_energy['aff']

                if gen_aff < gt_aff:
                # if True:
                    # Calculate the scRMSD
                    gen_pos, gen_seq = extract_pdb(gen_pdb)
                    pep_length = len(gen_seq['A'])
                    gen_pep_ca = gen_pos[:pep_length]
                    gen_seq_rec = ""
                    for key, value in gen_seq.items():
                        if key != "A":
                            for res_name in value:
                                gen_seq_rec += seq1(res_name)

                    chai_pdb = os.path.join(src_dir, "chai_eval", case, gen_item.replace('.pdb', '_chai.cif'))
                    pred_pos, pred_seq = extract_pdb(chai_pdb)
                    pred_pep_ca = pred_pos[:pep_length]
                    pred_res_ca = pred_pos[pep_length:]
                    pred_seq_rec = ""
                    for key, value in pred_seq.items():
                        if key != "A":
                            for res_name in value:
                                pred_seq_rec += seq1(res_name)

                    # Matching between gen_seq_rec and pred_seq_rec
                    _, gen_rec_mask, pred_rec_mask = find_LCS_and_masks(gen_seq_rec, pred_seq_rec)

                    # Align the two complexes w.r.t. anchor pos
                    gen_pos_anchor = gen_pos[pep_length:][gen_rec_mask]
                    pred_pos_anchor = pred_pos[pep_length:][pred_rec_mask]

                    Rot = kabsch_R(gen_pos_anchor, pred_pos_anchor)
                    pred_pep_aligned = (pred_pep_ca - pred_pos_anchor.mean(0)) @ Rot + gen_pos_anchor.mean(0)
                    # pred_rec_a = (pred_pos_anchor - pred_pos_anchor.mean(0)) @ Rot + gen_pos_anchor.mean(0)

                    scRMSD = np.linalg.norm(pred_pep_aligned - gen_pep_ca, axis=-1).mean()

                    candidate_v2.append(gen_pdb + f' GT_aff: {gt_aff} | Gen_aff: {gen_aff} | scRMSD: {scRMSD} \n')
    # torch.save(candidate_v2, 'candidate_list_.pt')
    with open('candidate_list_v4_threshold_6A.txt', 'w') as file:
        file.writelines(candidate_v2)
elif False:
    # Split multi-fold test sets
    ns3 = ['SEP_ns', 'TYS_ns', 'PTR_ns']
    aa_count_csv = '/ibex/user/guot0b/PepFlowww-NS3/data/aa_count_detailed.csv'
    aa_count_csv_con = pd.read_csv(aa_count_csv)

    # Get the case with NS3
    ns3_cases_pre = aa_count_csv_con['Sample'][aa_count_csv_con['aatype'].isin(ns3)].to_list()
    ns3_case_list = []
    for case in ns3_cases_pre:
        items = case.split(', ')
        for item in items:
            if len(item) > 1:
                ns3_case_list.append(item)
    ns3_case_set = set(ns3_case_list)

    val_test_ns3_cases = (set(val_items) | set(test_items)) & ns3_case_set
    val_test_wo_ns3_cases = (set(val_items) | set(test_items)) - ns3_case_set

    multifold_testset = {'TRAIN': [], 'VAL': [], 'TEST': []}
    fold = 5
    for idx in range(fold):
        new_test_items = random.sample(val_test_ns3_cases, 19)
        multifold_testset['TEST'] = new_test_items
        with open(f'/ibex/user/guot0b/PepFlowww-NS3/data/Dataset_NS3/data_split_test_fold_{idx}_NS3.json', 'w') as f:
            json.dump(multifold_testset, f, indent=4)
                    
        new_test_items = new_test_items + random.sample(val_test_wo_ns3_cases, 199)
        multifold_testset['TEST'] = new_test_items
        with open(f'/ibex/user/guot0b/PepFlowww-NS3/data/Dataset_NS3/data_split_test_fold_{idx}.json', 'w') as f:
            json.dump(multifold_testset, f, indent=4)
elif False:
    # Make JSON files for af3
    from Bio import SeqIO
    from Bio.PDB import PDBParser, PDBIO, MMCIFParser, Chain, Residue, Atom
    from Bio.PDB.Model import Model
    from Bio.PDB.Structure import Structure
    from Bio.SeqUtils import seq1    
    from pepflow.modules.protein.constants import AA, non_standard_residue_substitutions

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

    ns3 = {'SEP': 'SER', 'TYS': 'TYR', 'PTR': 'TYR'}
    save_dir = "/ibex/user/guot0b/alphafold3/pepflow_test/json_files"

    # Get the template
    seq_template = {'id': [], 'sequence': '', 'unpairedMsa': '', 'pairedMsa': '', 'templates': []}
    template_json = "/ibex/user/guot0b/alphafold3/pepflow_test/template.json"
    with open(template_json, 'r') as f:
        full_template = json.load(f)
    
    for item in test_items:
        item_folder = os.path.join(structure_dir, item)
        chain_id_alphabet = iter('BCDEFGHIJK')
        seq_list = []

        # Get the peptide sequence
        seq_tmp = seq_template.copy()
        seq_tmp["modifications"] = []
        _, pep_seq_3char = extract_pdb(os.path.join(item_folder, 'peptide.pdb'))
        pep_seq_3char = pep_seq_3char[next(iter(pep_seq_3char.keys()))]
        pep_seq = ''  # Need to preprocess
        for idx, char in enumerate(pep_seq_3char):
            if char in ns3.keys():
                pep_seq += seq1(ns3[char])
                seq_tmp["modifications"].append({"ptmType": char, "ptmPosition": idx+1})
            elif seq1(char) != 'X':
                pep_seq += seq1(char)
            else:
                pep_seq += 'G'  # GLY padding
        seq_tmp['id'] = 'A'
        seq_tmp['sequence'] = pep_seq
        seq_list.append({'protein': seq_tmp})
        

        # Get the receptor sequence
        for con in SeqIO.parse(os.path.join(item_folder, 'receptor.fasta'), "fasta"):
            seq_tmp = seq_template.copy()
            seq_tmp['id'] = next(chain_id_alphabet)
            seq_tmp['sequence'] = str(con.seq)
            seq_list.append({'protein': seq_tmp})
        
        # Get the full json
        _full_template = full_template.copy()
        _full_template['name'] = item
        _full_template['sequences'] = seq_list

        with open(os.path.join(save_dir, f"{item}.json"), 'w') as f:
            json.dump(_full_template, f)
elif False:
    
    # Collate the output of af3
    src_dir = "/ibex/user/guot0b/alphafold3/pepflow_NS3_test"
    tgt_dir = "/ibex/user/guot0b/PepFlowww-ibex/comparison_exp/af3_eval_NS3data"
    num = 16

    from Bio import SeqIO
    from Bio.PDB import PDBParser, PDBIO, MMCIFParser

    def cif2pdb(cif_in: str, pdb_out: str):
        parser = MMCIFParser()
        structure = parser.get_structure("TMP", cif_in)
        io = PDBIO()
        io.set_structure(structure)
        io.save(pdb_out)        

    if not os.path.isdir(tgt_dir):
        os.makedirs(tgt_dir)

    for item in test_items:
        print(item)
        item_dir = os.path.join(src_dir, item.lower())
        save_dir = os.path.join(tgt_dir, item)
        if not os.path.isdir(item_dir):
            continue

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if len(os.listdir(save_dir)) >= num:
            continue

        rec = 0
        flag = True
        for i in range(1, 5):
            if not flag:
                break
            for j in range(5):
                dirname = f"seed-{i}_sample-{j}"
                cif_path = os.path.join(item_dir, dirname, 'model.cif')
                pdb_path = os.path.join(save_dir, f"sample_{rec}.pdb")
                cif2pdb(cif_path, pdb_path)
                rec += 1
                if rec == 16:
                    flag = False
                    break
elif False:
    # Process the LNR dataset of PepGLAD. Step 1: get the json file
    save_dir = "/ibex/user/guot0b/DataPool/PepGLAD/LNR"
    json_path = "/ibex/user/guot0b/DataPool/PepGLAD/LNR_data_split.json"
    with open("/ibex/user/guot0b/DataPool/PepGLAD/LNR/LNR_ori/test.txt", 'r') as f:
        lines = f.readlines()
    data_split_LNR = {'TRAIN': [], 'VAL': [], 'TEST':[]}
    for line in lines:
        pdbid, _, chain, _ = line.split('\t')
        case = f"{pdbid}_{chain}"
        data_split_LNR['TEST'].append(case)
        if not os.path.exists(os.path.join(save_dir, case)):
            os.mkdir(os.path.join(save_dir, case))
    with open(json_path, 'w') as f:
        json.dump(data_split_LNR, f, indent=4)
elif False:
    # Process the LNR dataset of PepGLAD. Step 2: split the peptide and receptor, then get pocket, finally renumber them
    from Bio.PDB import PDBParser, PDBIO, Select, Selection
    from Bio.PDB.NeighborSearch import NeighborSearch
    from data.utils import renumber_chain

    class ChainSelect(Select):
        def __init__(self, target_chains):
            self.target_chains = target_chains

        def accept_chain(self, chain):
            return chain.get_id() in self.target_chains

        def accept_residue(self, residue):
            return residue.id[0] == " "  # Keep only standard residues (no heteroatoms)

    def split_pdb_by_chain(input_pdb, peptide_chain_id, output_dir):
        """
        Split a PDB file into peptide and receptor files based on chain ID.

        Args:
            input_pdb (str): Path to the input PDB file.
            peptide_chain_id (str): Chain ID to extract as peptide.
            output_dir (str): Directory to save the output PDB files.

        Outputs:
            - peptide.pdb: containing only the peptide chain.
            - receptor.pdb: containing the other chains.
        """
        os.makedirs(output_dir, exist_ok=True)

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', input_pdb)

        # Get all chain IDs
        all_chains = {chain.get_id() for model in structure for chain in model}
        receptor_chains = all_chains - {peptide_chain_id}

        # Save peptide
        io = PDBIO()
        io.set_structure(structure)
        peptide_path = os.path.join(output_dir, "peptide.pdb")
        io.save(peptide_path, ChainSelect({peptide_chain_id}))

        # Save receptor
        io.set_structure(structure)
        receptor_path = os.path.join(output_dir, "receptor.pdb")
        io.save(receptor_path, ChainSelect(receptor_chains))
    
    def extract_pocket(data_path, name):
        lig_parser, rec_parser = PDBParser(), PDBParser()
        lig_structure, rec_structure = lig_parser.get_structure('lig', os.path.join(data_path,name,f'peptide.pdb')), rec_parser.get_structure('rec', os.path.join(data_path,name, f'receptor.pdb'))

        lig_atoms = Selection.unfold_entities(lig_structure, 'A')
        rec_atoms = Selection.unfold_entities(rec_structure, 'A')

        search = NeighborSearch(rec_atoms)
        near_res = []
        for atom in lig_atoms:
            near_res += search.search(atom.get_coord(), 10, level='R')
        near_res = list(set([res.get_full_id() for res in near_res])) # get_id() may affect homo dimer
        
        class ResidueSelector(Select):

            def accept_residue(self, residue):
                return residue.get_full_id() in near_res
            
        os.makedirs(os.path.join(data_path,name),exist_ok=True)
        io = PDBIO()
        io.set_structure(rec_structure)
        io.save(os.path.join(data_path,name,'pocket.pdb'), ResidueSelector())


    src_dir = "/ibex/user/guot0b/DataPool/PepGLAD/LNR/LNR_ori/pdbs"
    json_path = "/ibex/user/guot0b/DataPool/PepGLAD/LNR_data_split.json"
    tgt_dir = "/ibex/user/guot0b/DataPool/PepGLAD/LNR"

    with open(json_path, 'r') as f:
        data_split_LNR = json.load(f)
    test_items_LNR = data_split_LNR['TEST']

    # DEBUG
    test_items_LNR_independent = sorted(set(test_items_LNR) - set(train_items))
    eval_con = pd.read_csv("/ibex/user/guot0b/PepFlowww-ibex/logs_lr1/2024.12.22_v1_pepflow_640k/test_320k_num16_LNR/eval_recovery.csv")
    _filter = eval_con['Case'].isin(test_items_LNR_independent)
    eval_con_independent = eval_con[_filter]
    eval_con_independent.to_csv("/ibex/user/guot0b/PepFlowww-ibex/logs_lr1/2024.12.22_v1_pepflow_640k/test_320k_num16_LNR/eval_recovery_independent.csv", index=False)
    # DEBUG

    pdb.set_trace()
    # for case in test_items_LNR:
    #     pdbid, chain_id = case.split('_')
    #     case_output_dir = os.path.join(tgt_dir, case)
    #     pdb_path = os.path.join(src_dir, f"{pdbid}.pdb")
    #     split_pdb_by_chain(pdb_path, chain_id, case_output_dir)
    #     extract_pocket(tgt_dir, case)
    # pdb.set_trace()

    renumber_chain(tgt_dir, test_items_LNR)
elif False:
    # Calculate caRMSD and AAR for specific cases (Visualization)
    from Bio.SeqUtils import seq1
    ID = "7sqr_1_III_K"
    tgt_dir = "/ibex/user/guot0b/PepFlowww-NS3/logs/2025.01.13_v2_NS_pep_SC0.1_rBLV15_320k/test_320k_num16"
    tgt_case_dir = os.path.join(tgt_dir, ID)

    gt_pdb = os.path.join(structure_dir, ID, "pocket_merge_from_pt_NS3.pdb")
    gt_dict = preprocess_structure_v3(gt_pdb)
    gt_pos, gt_seq = gt_dict['pep_pos'], gt_dict['pep_seq']
    gt_pos_rec = gt_dict['rec_pos']
    gt_seq_str = ''.join([seq1(str(constants.AA(gt_seq.numpy()[idx]))) for idx in range(len(gt_seq))])

    print(f"Case {ID} | Seq {gt_seq_str}")

    for item in os.listdir(tgt_case_dir):
        if not item.endswith(".pdb"):
            continue
        item_path = os.path.join(tgt_case_dir, item)

        gen_dict = preprocess_structure_v3(item_path)
        gen_pos, gen_seq = gen_dict['pep_pos'], gen_dict['pep_seq']
        gen_pos_rec = gen_dict['rec_pos']
        gen_seq_str = ''.join([seq1(str(constants.AA(gen_seq.numpy()[idx]))) for idx in range(len(gen_seq))])

        # AAR
        rec_seq = (gt_seq == gen_seq).sum() / len(gen_seq)

        # caRMSD
        caRMSD = torch.linalg.norm(gen_pos[:, 1] - gt_pos[:, 1], dim=-1).mean()

        print(f"Case {ID} | {item} | AAR {round(rec_seq.item(), 5)} | caRMSD {round(caRMSD.item(), 5)} | Seq {gen_seq_str}")
elif False:
    group_labels = ['AAR (%)', 'AAR(NS) (%)', 'caRMSD (Å)']
    data = [
        [22.53, 15.81, 20.99, 17.49],  # AAR
        [29.77, 26.79, 25.00, 4.17],   # AAR(NS)
        [4.92, 5.82, 5.22, 5.08]       # caRMSD
    ]

    x = np.arange(len(group_labels))  # positions for the 3 metric groups
    width = 0.18
    gap = 0.1
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width + gap, 1.5 * width + gap]
    method_labels = ['NS-Pep (RFGM)', 'NS-Pep (RFGM w/o noise)', 'NS-Pep (BLV)', 'NS-Pep (BLV w/o noise)']
    colors = ["#96CAC1",
              "#F6F6BC",
              "#C1BED6",
              "#EA8E83"]
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars for each method in each group
    for i in range(len(method_labels)):
        values = [data[j][i] for j in range(len(group_labels))]  # ith method's performance across metrics
        ax.bar(x + offsets[i], values, width, label=method_labels[i], color=colors[i])

    # Labels and formatting
    ax.set_ylabel('Metric Value', fontsize=16)
    ax.set_title('Ablation Study on noise of RFGM and BLV', fontsize=20)
    ax.tick_params(labelsize=16)  # for tick labels
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.legend(loc='upper right', fontsize=16)

    plt.tight_layout()
    plt.savefig('figures/ablation_noise.png')
elif False:
    src_dir = "/ibex/user/guot0b/PepFlowww-NS3/figures/Folding_wrt_length"
    model_list = ['AlphaFold3', 'ESMFold', 'PepFlow', 'NS-Pep']

    # length ranges (inclusive lower bound, exclusive upper bound)
    ranges = {(3,  10): "Short",   # 3-9
              (10, 15): "Medium",  # 10-14
              (15,26): "Long"}    # 15-25

    # ---------- load & reshape ----------
    records = []
    for model in model_list:
        with open(os.path.join(src_dir, f"{model}_rmsd.json")) as f:
            data = json.load(f)

        for length_str, rmsd_list in data.items():
            L = int(length_str)

            # find which split this length belongs to
            split = next(name for (lo, hi), name in ranges.items() if lo <= L < hi)

            # expand one row per RMSD value (tidy format)
            records.extend(
                {"Model": model, "Split": split, "RMSD": v}
                for v in rmsd_list
            )

    df = pd.DataFrame(records)

    # ---------- draw ----------
    plt.figure(figsize=(8, 5))
    # hue-style grouping: four boxes inside each split
    # (seaborn makes this very convenient, but sticking with matplotlib is fine too)
    import seaborn as sns
    palette = {
        "AlphaFold3": "#ffb6c1",   # light purple,  violet #d1b3ff
        "ESMFold": "#f6c28b",     # light orange
        "PepFlow": "#a2cffe",      # light blue
        "NS-Pep": "#b6e3a4",       # light green
    }

    ax = sns.boxplot(data=df, x="Split", y="RMSD", hue="Model",
                     palette=palette, showcaps=True, fliersize=2, linewidth=1)

    plt.ylabel("")
    plt.xlabel("")
    # plt.title("Peptide folding accuracy vs. length split")
    # plt.legend(title="", frameon=False, loc="upper right")
    plt.legend().remove()
    plt.ylim(0, 30)
    plt.yticks(fontsize=20) 
    plt.xticks([])
    plt.grid('on', ls='--', alpha=0.66)

    for patch in ax.artists:
        patch.set_alpha(0.3)  # adjust to desired transparency (e.g., 0.5 for 50%)

    plt.tight_layout()
    plt.savefig(os.path.join(src_dir, "boxplot.png"))
elif False:
    # Make pocket_merge_re_renum.pdb
    for case in test_items:
        merge_pdb(os.path.join(structure_dir, case, 'peptide.pdb'),
                  os.path.join(structure_dir, case, 'pocket.pdb'),
                  os.path.join(structure_dir, case, 'tmp.pdb'),
                  set_chain_ID='A')
        renumber_pdb_residues(os.path.join(structure_dir, case, 'tmp.pdb'),
                              os.path.join(structure_dir, case, 'pocket_merge_re_renum.pdb'))
        os.remove(os.path.join(structure_dir, case, 'tmp.pdb'))
elif True:
    # Extract the ESM emb for the data
    # TODO: 1. use preprocess_structure_v2 to extract the 'resseq' from pocket and receptor; 2. map from rec to pocket
    # tmp = extract_pocket('1bmb_I')
    from pepflow.modules.protein.parsers import parse_pdb
    from models_con.pep_dataloader import preprocess_structure_v2
    from transformers import AutoTokenizer, AutoModel

    def esm_extractor(seq, tokenizer, model):
        inputs = tokenizer(seq, return_tensors="pt")
        inputs = {key: val.to(model.device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state[0, 1:-1, :]
        return token_embeddings      

    # Initilization for esm2
    model_name = "facebook/esm2_t12_35M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    cases = os.listdir(structure_dir)
    for case in cases:
        print(f"{case}...")
        case_dir = os.path.join(structure_dir, case)
        result_path = os.path.join(case_dir, 'pocket_emb_esm2_t12_480.pt')
        if os.path.exists(result_path):
            continue
        try:
            poc = parse_pdb(os.path.join(case_dir, 'pocket.pdb'))[0]
            rec = parse_pdb(os.path.join(case_dir, 'receptor.pdb'))[0]
        except FileNotFoundError:
            continue
        try:
            poc_in_rec = torch.isin(rec['resseq'], poc['resseq'])
        except TypeError:
            continue

        rec_seq = torch.where(rec['aa']==20, 15, rec['aa'])
        rec_seq = torch.where(rec_seq==21, 19, rec_seq)
        rec_seq = torch.where(rec_seq==22, 19, rec_seq)
        rec_seq_list = []
        for idx in range(len(rec_seq)):
            rec_seq_list.append(seq1(str(constants.AA(rec['aa'][idx].item()))))
        rec_seq = ''.join(rec_seq_list)

        rec_emb = esm_extractor(rec_seq, tokenizer, model)
        poc_emb = rec_emb[poc_in_rec]

        torch.save(poc_emb.cpu(), result_path)


elif False:
    tgt_dir = '/ibex/user/guot0b/DataPool/pepflow_eval'

    for case in test_items:
        src_folder = os.path.join(structure_dir, case)
        tgt_case_dir = os.path.join(tgt_dir, case)

        shutil.copytree(src_folder, tgt_case_dir)
