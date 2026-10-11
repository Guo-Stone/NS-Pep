#!/bin/bash
# Evaluation script for NS-Pep (NS3 version), adapted from our cluster script
# (pepflow_eval_rec_NS3.sh) with all Slurm/cluster-specific lines removed.
# Generates 16 peptides per test case with flow_model.NS_Pep and computes
# AAR (rec_seq), RMSD (rec_pos), SSR (rec_ss), BSR (rec_bs) and diversity.
#
# Usage (from the repository root):
#   conda activate bio
#   bash scripts/eval_NS3.sh [DATA_DIR] [OUTPUT_DIR]

DATA_DIR=${1:-./data/Pepflow/}       # unzipped dataset, one folder per test case
OUTPUT_DIR=${2:-./outputs/test_320k_num16}
CKPT=./checkpoints/NS-Pep_320k.pt
CONFIG=./configs/NS-Pep_320k.yaml
TEST_LIST=./data/data_split_NS3_v3.json

python eval_recovery.py \
    --ckpt $CKPT \
    --config $CONFIG \
    --data_dir $DATA_DIR \
    --test_list $TEST_LIST \
    --output_dir $OUTPUT_DIR \
    --num 16 \
    --device cuda:0
