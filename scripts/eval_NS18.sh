#!/bin/bash
# Evaluation script for NS-Pep (NS18 version), adapted from our cluster script
# (pepflow_eval_rec_NS18.sh) with all Slurm/cluster-specific lines removed.
# Generates 16 peptides per test case with flow_model.NS_Pep_v3 and computes
# AAR (rec_seq), C-RMSD / L-RMSD (rec_pos), SSR (rec_ss), BSR (rec_bs),
# diversity and per-NSAA sequence recovery for the 18 NSAAs.
#
# Usage (from the repository root):
#   conda activate bio
#   bash scripts/eval_NS18.sh [DATA_DIR] [OUTPUT_DIR]

DATA_DIR=${1:-./data/Pepflow/}       # unzipped dataset, one folder per test case
OUTPUT_DIR=${2:-./outputs/NS18_test_320k_num16}
CKPT=./checkpoints/NS-Pep_NS18_320k.pt
CONFIG=./configs/NS-Pep_NS18_320k.yaml
TEST_LIST=./data/data_split_NS18_v4_seed1177_curated.json

python eval_recovery.py \
    --ckpt $CKPT \
    --config $CONFIG \
    --data_dir $DATA_DIR \
    --test_list $TEST_LIST \
    --output_dir $OUTPUT_DIR \
    --num 16 \
    --device cuda:0
