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

*Model weights, training/inference code, and related files will be released later.*

## Citation

```bibtex
@article{guo2025ns,
  title={NS-Pep: De novo Peptide Design with Non-Standard Amino Acids},
  author={Guo, Tao and Yin, Junbo and Wang, Yu and Gao, Xin},
  journal={arXiv preprint arXiv:2510.03326},
  year={2025}
}
```
