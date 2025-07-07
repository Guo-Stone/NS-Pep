from transformers import AutoTokenizer, EsmForProteinFolding
from accelerate.test_utils.testing import get_backend
import torch
import pdb
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
import os
import warnings
import sys
warnings.filterwarnings("ignore")
from pathlib import Path
import shutil
from Bio.PDB import PDBParser, PDBIO, MMCIFParser, Chain, Residue, Atom
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure
from Bio.SeqUtils import seq1
import numpy as np
import json
import pandas as pd
import random
from multiprocessing import Pool
from functools import partial
import math

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # For a single GPU
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs

DEVICE, _, _ = get_backend()

tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)

model = model.to(DEVICE)
# model.esm = model.esm.half()

def esmfold_infer(sequences: list, output_path: str, linker_len=36):
    linker = 'G' * linker_len
    cat_seq = ''
    chain_index_list = []
    residue_index_list = []
    mask_index_list = []
    position_ids_list = []
    pos_head = 0
    for idx in range(len(sequences)):
        # TODO: linker_mask, chain_idx, residue_index, position_ids
        cat_seq += sequences[idx]
        chain_index_list += [idx] * len(sequences[idx])
        residue_index_list += [res_id for res_id in range(len(sequences[idx]))]
        mask_index_list += [1] * len(sequences[idx])
        position_ids_list += [res_id for res_id in range(pos_head, pos_head+len(sequences[idx]))]
        pos_head += len(sequences[idx])
        if idx < len(sequences) - 1:
            cat_seq += linker
            chain_index_list += [-1] * len(linker)
            residue_index_list += [res_id for res_id in range(len(linker))]
            mask_index_list += [0] * len(linker)
            position_ids_list += [res_id for res_id in range(pos_head, pos_head+len(linker))]
            pos_head += len(linker)
        pos_head += 512

    tokenized_homodimer = tokenizer([cat_seq], return_tensors="pt", add_special_tokens=False)

    with torch.no_grad():
        position_ids = torch.tensor(position_ids_list, dtype=torch.long)

        tokenized_homodimer['position_ids'] = position_ids.unsqueeze(0)
        tokenized_homodimer = {key: tensor.to(DEVICE) for key, tensor in tokenized_homodimer.items()}

        output = model(**tokenized_homodimer)

    linker_mask = torch.tensor(mask_index_list)[None, :, None]

    output['atom37_atom_exists'] = output['atom37_atom_exists'] * linker_mask.to(output['atom37_atom_exists'].device)
    output['residue_index'] = torch.tensor(residue_index_list, dtype=torch.int64).to(output['atom37_atom_exists'].device)[None, :]
    output['chain_index'] = torch.tensor(chain_index_list, dtype=torch.int64).to(output['atom37_atom_exists'].device)[None, :]
    pdbs = convert_outputs_to_pdb(output)
    with open(output_path, 'w') as f: f.writelines(pdbs)

"""
seqs = ["DSIQAEEWYFGKITRRESERLLLNAENPRGTFLVRESETTKGAYSLSVSDFDNAKGLNVKHYKIRKLDSGGFYITSRTQFNSLQQLVAYYSKHADGLCHRLTTVCP", "DSIQAEEWYFGKITRRESERLLLNAENPRGTFLVRESETTKGAYSLSVSDFDNAKGLNVKHYKIRKLDSGGFYITSRTQFNSLQQLVAYYSKHADGLCHRLTTVCP", "DSIQAEEWYFGKITRRESERLLLNAENPRGTFLVRESETTKGAYSLSVSDFDNAKGLNVKHYKIRKLDSGGFYITSRTQFNSLQQLVAYYSKHADGLCHRLTTVCP"]

esmfold_infer(seqs, '/ibex/user/guot0b/PepFlowww-NS3/tmp.pdb')
"""

"""
linker = 'G' * 36

cat_seq = ''
chain_index_list = []
residue_index_list = []
mask_index_list = []
position_ids_list = []
pos_head = 0
for idx in range(len(seqs)):
    # TODO: linker_mask, chain_idx, residue_index, position_ids
    cat_seq += seqs[idx]
    chain_index_list += [idx] * len(seqs[idx])
    residue_index_list += [res_id for res_id in range(len(seqs[idx]))]
    mask_index_list += [1] * len(seqs[idx])
    position_ids_list += [res_id for res_id in range(pos_head, pos_head+len(seqs[idx]))]
    pos_head += len(seqs[idx])
    if idx < len(seqs) - 1:
        cat_seq += linker
        chain_index_list += [-1] * len(linker)
        residue_index_list += [res_id for res_id in range(len(linker))]
        mask_index_list += [0] * len(linker)
        position_ids_list += [res_id for res_id in range(pos_head, pos_head+len(linker))]
        pos_head += len(linker)
    pos_head += 512
tokenized_homodimer = tokenizer([cat_seq], return_tensors="pt", add_special_tokens=False)

with torch.no_grad():
    # position_ids = torch.arange(len(homodimer_sequence), dtype=torch.long)
    # position_ids[len(sequence) + len(linker):] += 512
    position_ids = torch.tensor(position_ids_list, dtype=torch.long)

    tokenized_homodimer['position_ids'] = position_ids.unsqueeze(0)
    tokenized_homodimer = {key: tensor.to(DEVICE) for key, tensor in tokenized_homodimer.items()}

    output = model(**tokenized_homodimer)

# linker_mask = torch.tensor([1] * len(sequence) + [0] * len(linker) + [1] * len(sequence))[None, :, None]
linker_mask = torch.tensor(mask_index_list)[None, :, None]

output['atom37_atom_exists'] = output['atom37_atom_exists'] * linker_mask.to(output['atom37_atom_exists'].device)
output['residue_index'] = torch.tensor(residue_index_list, dtype=torch.int64).to(output['atom37_atom_exists'].device)[None, :]
output['chain_index'] = torch.tensor(chain_index_list, dtype=torch.int64).to(output['atom37_atom_exists'].device)[None, :]
pdbs = convert_outputs_to_pdb(output)
with open('/ibex/user/guot0b/PepFlowww-NS3/tmp.pdb', 'w') as f: f.writelines(pdbs)
"""

if __name__ == '__main__':
    structure_dir = "/ibex/user/guot0b/DataPool/Pepflow/"
    output_dir = '/ibex/user/guot0b/PepFlowww-ibex/comparison_exp/esmfold_NS3data_fixed_seq_v2'
    test_path = "/ibex/user/guot0b/PepFlowww-NS3/data/data_split_NS3_v3.json"
    num = 16
    linker_len = 48

    with open(test_path, 'r') as f:
        data_split = json.load(f)
    test_items = data_split['TEST']
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for case in test_items:
        print(f"{case}...")
        case_output_dir = os.path.join(output_dir, case)
        if not os.path.exists(case_output_dir):
            os.mkdir(case_output_dir)        
        if len(os.listdir(case_output_dir)) >= num:
            print(f"Pass the case {case} because it exists.")
            continue

        # Obtain the sequence
        pep_fasta = os.path.join(structure_dir, case, 'peptide.fasta')
        rec_fasta = os.path.join(structure_dir, case, 'receptor.fasta')
        
        seqs = []
        with open(pep_fasta, 'r') as f:
            con = f.readlines()
        seqs.append(con[1].strip())

        with open(rec_fasta, 'r') as f:
            con = f.readlines()
        for line in con:
            if line.startswith('>'):
                continue
            else:
                seqs.append(line.strip())

        for idx in range(num):
            set_seed(idx)
            item_path = os.path.join(case_output_dir, f"sample_{idx}.pdb")
            try:
                esmfold_infer(seqs, item_path)
            except torch.cuda.OutOfMemoryError:
                break

        


