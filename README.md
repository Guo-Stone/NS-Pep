# NS-Pep: *De novo* Peptide Design with Non-Standard Amino Acids

**NS-Pep** is a unified flow-matching framework for co-designing peptide sequences and structures that incorporate non-standard amino acids (NSAAs). It introduces Residue Frequency-Guided Modification (RFGM), Progressive Side-chain Perception (PSP), and Interaction-Aware Weighting (IAW) to tackle the long-tailed NSAA distribution and to capture fine-grained side-chain geometry.

> **Branch `NS18`** — release of the NS-Pep model extended to **18 non-standard amino acids**.
> The 3-NSAA version (SEP / TYS / PTR) used for Table 1 of the paper is on the [`master`](https://github.com/Guo-Stone/NS-Pep/tree/master) branch.

The 18 NSAAs (`data/NS18_list.json`; constants in `pepflow/modules/protein/constants.py`):
`SEP, TYS, PTR, MLE, M3L, ALY, TPO, DAL, HYP, MVA, DLY, DLE, SAR, NLE, BMT, DGL, DPR, DTR`

## Results on the NS18 Test Set

From `results/NS18_test_320k_num16/eval_recovery.csv` (`All` row; 16 samples per case, 187 test cases):

| AAR % ↑ | AAR on NSAA positions % ↑ | C-RMSD Å ↓ | L-RMSD Å ↓ | SSR % ↑ | BSR % ↑ | Diversity ↑ |
|---|---|---|---|---|---|---|
| 18.73 | 17.76 (844 / 4752) | 5.56 | 2.79 | 95.86 | 81.41 | 0.47 |

## Model Weights & Evaluation

| Item | Path |
|---|---|
| Checkpoint (320k iterations; model weights + config only) | `checkpoints/NS-Pep_NS18_320k.pt` (Git LFS) |
| Model class | `models_con/flow_model.py` → `NS_Pep_v3` |
| Evaluation code | `eval_recovery.py` |
| Evaluation script | `scripts/eval_NS18.sh` |
| Config (used to train this checkpoint) | `configs/NS-Pep_NS18_320k.yaml` |
| NSAA list / constants | `data/NS18_list.json`, `pepflow/modules/protein/constants.py` |
| Dataset split (TRAIN / TEST = 9665 / 188) | `data/data_split_NS18_v4_seed1177_curated.json` |
| Reference results | `results/NS18_test_320k_num16/eval_recovery.csv` |
| Environment (`bio`) | `environment/bio_environment.yml`, `environment/bio_conda_list.txt`, `environment/bio_pip_freeze.txt` |

### Data

The dataset can be downloaded from Google Drive:
**https://drive.google.com/file/d/1j459qA_2MFAvEHaZ6CceT5B8a3z8HKw0/view?usp=sharing**

After unzipping, each test case is a folder named by its ID. It contains the receptor/pocket/peptide PDB files and the preprocessed `<ID>_re_NS18.pt`. If that `.pt` file is missing, `eval_recovery.py` rebuilds it from `peptide_re.pdb` / `pocket_re.pdb`.

### Run

```bash
git lfs pull                          # fetch the checkpoint
conda env create -f environment/bio_environment.yml
conda activate bio
bash scripts/eval_NS18.sh /path/to/Pepflow/ ./outputs/NS18_test_320k_num16
```

The script writes `eval_recovery.csv` to the output directory, with one row per test case and an `All` summary row.
Its columns are `rec_seq` (AAR), `rec_pos(C-RMSD)`, `rec_pos(L-RMSD)`, `rec_ss` (SSR), `rec_bs` (BSR), `diversity`, and `rec_seq(<NSAA>)`. Each `rec_seq(<NSAA>)` cell is written as `correct_total`.
Sampling is stochastic, so a fresh run gives a slightly different value. Our re-run on a V100 gave an AAR of 18.82%, compared with the 18.73% reference.

Notes:
- `eval_recovery.py` and the config files still contain our cluster paths as defaults. `scripts/eval_NS18.sh` overrides every path that inference needs.
- The model class is selected by the uncommented `model = flow_model.NS_Pep_v3(...)` line in `eval_recovery.py`.

## Citation

```bibtex
@article{guo2025ns,
  title={NS-Pep: De novo Peptide Design with Non-Standard Amino Acids},
  author={Guo, Tao and Yin, Junbo and Wang, Yu and Gao, Xin},
  journal={arXiv preprint arXiv:2510.03326},
  year={2025}
}
```
