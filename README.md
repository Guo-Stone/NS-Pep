# NS-Pep: *De novo* Peptide Design with Non-Standard Amino Acids

**NS-Pep** is a unified flow-matching framework for co-designing peptide sequences and structures that incorporate non-standard amino acids (NSAAs). It introduces Residue Frequency-Guided Modification (RFGM), Progressive Side-chain Perception (PSP), and Interaction-Aware Weighting (IAW) to tackle the long-tailed NSAA distribution and to capture fine-grained side-chain geometry.

## Peptide Generation on the General Test Set

Table 1 of the paper: comparison for peptide generation on the **General** test set.

| Method | AAR % ↑ | AAR(S) % ↑ | RMSD Å ↓ | SSR % ↑ | BSR % ↑ | AFF % ↑ | scRMSD Å ↓ | Diversity ↑ |
|---|---|---|---|---|---|---|---|---|
| RFdiffusion | 8.66 | 8.69 | 17.57 | 71.75 | 6.52 | 12.06 | 19.64 | 0.41 |
| ProteinGenerator | 7.08 | 7.12 | 17.26 | 75.30 | 4.28 | 8.91 | 23.02 | 0.39 |
| PPFLOW | 8.63 | 8.87 | 9.45 | 0.11 | 75.14 | 6.53 | 15.77 | *0.66* |
| DiffPepBuilder | 6.71 | 6.77 | 8.36 | 75.49 | **85.86** | *12.11* | 14.65 | 0.51 |
| PepGLAD | 6.82 | 6.90 | 7.78 | 78.72 | 81.48 | 10.36 | 13.43 | **0.72** |
| PepFlow | *16.30* | *16.76* | 4.98 | **85.80** | 81.50 | 11.60 | 12.32 | 0.50 |
| PepFlow* | 15.88 | 15.88 | *4.94* | 83.40 | 84.36 | 11.39 | *11.67* | 0.51 |
| **NS-Pep (ours)** | **22.53** | **22.35** | **4.92** | *85.00* | *81.62* | **17.23** | **11.50** | 0.49 |

**Bold** = best; *italic* = second best.

## Model Weights & Evaluation

The checkpoint and evaluation files below reproduce the **NS-Pep** row of the table above (AAR = 22.53%).

| Item | Path |
|---|---|
| Checkpoint (320k iterations; model weights + config only) | `checkpoints/NS-Pep_320k.pt` (Git LFS) |
| Model class | `models_con/flow_model.py` → `NS_Pep` |
| Evaluation code | `eval_recovery.py` |
| Evaluation script | `scripts/eval_NS3.sh` |
| Config (used to train this checkpoint) | `configs/NS-Pep_320k.yaml` |
| NSAA constants (SEP / TYS / PTR) | `pepflow/modules/protein/constants.py` |
| Dataset split (TRAIN / VAL / TEST = 8228 / 1065 / 219) | `data/data_split_NS3_v3.json` |
| Reference results (16 samples per test case) | `results/test_320k_num16/eval_recovery.csv` |
| Environment (`bio`) | `environment/bio_environment.yml`, `environment/bio_conda_list.txt`, `environment/bio_pip_freeze.txt` |

### Data

The dataset can be downloaded from Google Drive:
**https://drive.google.com/file/d/1j459qA_2MFAvEHaZ6CceT5B8a3z8HKw0/view?usp=sharing**

After unzipping, each test case is a folder named by its ID (e.g. `1a46_I/`). It contains the receptor/pocket/peptide PDB files and the preprocessed `<ID>_re_NS3.pt`. If that `.pt` file is missing, `eval_recovery.py` rebuilds it from `peptide_re.pdb` / `pocket_re.pdb`.

### Run

```bash
git lfs pull                          # fetch the checkpoint
conda env create -f environment/bio_environment.yml
conda activate bio
bash scripts/eval_NS3.sh /path/to/Pepflow/ ./outputs/test_320k_num16
```

The script writes `eval_recovery.csv` to the output directory, with one row per test case and an `All` summary row:
`rec_seq` = AAR, `rec_pos` = RMSD, `rec_ss` = SSR, `rec_bs` = BSR. The `Clash` column is disabled and always reads 100.
Sampling is stochastic, so a fresh run gives a slightly different value. Our re-run on a V100 gave an AAR of 22.78%, compared with the 22.53% reference.

Notes:
- `eval_recovery.py` and the config files still contain our cluster paths as defaults. `scripts/eval_NS3.sh` overrides every path that inference needs.
- The model class is selected by the uncommented `model = flow_model.NS_Pep(...)` line in `eval_recovery.py`.

## Citation

```bibtex
@article{guo2025ns,
  title={NS-Pep: De novo Peptide Design with Non-Standard Amino Acids},
  author={Guo, Tao and Yin, Junbo and Wang, Yu and Gao, Xin},
  journal={arXiv preprint arXiv:2510.03326},
  year={2025}
}
```
