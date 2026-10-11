import pdb

import torch
import enum

## others
NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE

PAD_RESIDUE_INDEX = 39  # 21, 24(NS3)

# WEIGHT_DICT = {'LEU': 0.107, 'ALA': 0.119, 'ARG': 0.134,
#                'PRO': 0.144, 'SER': 0.157, 'GLU': 0.16,
#                'GLY': 0.162, 'LYS': 0.174, 'ILE': 0.183,
#                'THR': 0.191, 'ASP': 0.193, 'VAL': 0.198,
#                'PHE': 0.245, 'GLN': 0.246, 'ASN': 0.289,
#                'TYR': 0.314, 'HIS': 0.452, 'MET': 0.563,
#                'TRP': 0.668, 'CYS': 0.776, 'SEP': 2.68,
#                'TYS': 4.827, 'PTR': 5.232}  # derived from script.py

WEIGHT_DICT = {'LEU': 0.109, 'ALA': 0.12, 'ARG': 0.134, 
               'PRO': 0.145, 'SER': 0.158, 'GLU': 0.161, 
               'GLY': 0.162, 'LYS': 0.177, 'ILE': 0.186, 
               'THR': 0.192, 'ASP': 0.196, 'VAL': 0.202, 
               'PHE': 0.249, 'GLN': 0.25, 'ASN': 0.296, 
               'TYR': 0.319, 'HIS': 0.455, 'MET': 0.579, 
               'TRP': 0.681, 'CYS': 0.784, 'SEP': 2.427, 
               'TYS': 4.393, 'PTR': 4.827, 'M3L': 5.525, 
               'HYP': 5.896, 'ALY': 6.143, 'TPO': 7.382, 
               'DAL': 9.549, 'DLY': 16.269, 'NLE': 18.692, 
               'DLE': 20.43, 'MLE': 22.526, 'MVA': 25.1, 
               'DGL': 29.284, 'DPR': 32.537, 'DTR': 35.14, 
               'SAR': 51.677, 'BMT': 146.418}  # NS18

# BLV_weight_list = [0.392, 0.66, 0.469, 0.439, 0.511, 0.449,
#                    0.58, 0.47, 0.446, 0.359, 0.627, 0.535,
#                    0.422, 0.504, 0.408, 0.442, 0.459, 0.478,
#                    0.668, 0.547, 0.881, 0.969, 1.0]  # NS3
BLV_weight_list = [0.308, 0.517, 0.367, 0.345, 0.401, 0.349, 
                   0.454, 0.365, 0.349, 0.283, 0.492, 0.417, 
                   0.328, 0.396, 0.319, 0.344, 0.362, 0.37, 
                   0.52, 0.428, 0.689, 0.759, 0.764, 0.777, 
                   0.796, 0.796, 0.802, 0.805, 0.811, 0.894, 
                   0.906, 0.923, 0.943, 0.943, 0.97, 0.993, 
                   0.996, 1.0]  # NS18

##
# Residue identities

non_standard_residue_substitutions = {
    '2AS': 'ASP', '3AH': 'HIS', '5HP': 'GLU', 'ACL': 'ARG', 'AGM': 'ARG', 'AIB': 'ALA', 'ALM': 'ALA', 'ALO': 'THR',
    'ALY': 'ALY', 'ARM': 'ARG',
    'ASA': 'ASP', 'ASB': 'ASP', 'ASK': 'ASP', 'ASL': 'ASP', 'ASQ': 'ASP', 'AYA': 'ALA', 'BCS': 'CYS', 'BHD': 'ASP',
    'BMT': 'BMT', 'BNN': 'ALA',
    'BUC': 'CYS', 'BUG': 'LEU', 'C5C': 'CYS', 'C6C': 'CYS', 'CAS': 'CYS', 'CCS': 'CYS', 'CEA': 'CYS', 'CGU': 'GLU',
    'CHG': 'ALA', 'CLE': 'LEU', 'CME': 'CYS',
    'CSD': 'ALA', 'CSO': 'CYS', 'CSP': 'CYS', 'CSS': 'CYS', 'CSW': 'CYS', 'CSX': 'CYS', 'CXM': 'MET', 'CY1': 'CYS',
    'CY3': 'CYS', 'CYG': 'CYS',
    'CYM': 'CYS', 'CYQ': 'CYS', 'DAH': 'PHE', 'DAL': 'DAL', 'DAR': 'ARG', 'DAS': 'ASP', 'DCY': 'CYS', 'DGL': 'DGL',
    'DGN': 'GLN', 'DHA': 'ALA',
    'DHI': 'HIS', 'DIL': 'ILE', 'DIV': 'VAL', 'DLE': 'DLE', 'DLY': 'DLY', 'DNP': 'ALA', 'DPN': 'PHE', 'DPR': 'DPR',
    'DSN': 'SER', 'DSP': 'ASP',
    'DTH': 'THR', 'DTR': 'DTR', 'DTY': 'TYR', 'DVA': 'VAL', 'EFC': 'CYS', 'FLA': 'ALA', 'FME': 'MET', 'GGL': 'GLU',
    'GL3': 'GLY', 'GLZ': 'GLY',
    'GMA': 'GLU', 'GSC': 'GLY', 'HAC': 'ALA', 'HAR': 'ARG', 'HIC': 'HIS', 'HIP': 'HIS', 'HMR': 'ARG', 'HPQ': 'PHE',
    'HTR': 'TRP', 'HYP': 'HYP',
    'IAS': 'ASP', 'IIL': 'ILE', 'IYR': 'TYR', 'KCX': 'LYS', 'LLP': 'LYS', 'LLY': 'LYS', 'LTR': 'TRP', 'LYM': 'LYS',
    'LYZ': 'LYS', 'MAA': 'ALA', 'MEN': 'ASN',
    'MHS': 'HIS', 'MIS': 'SER', 'MLE': 'MLE', 'MPQ': 'GLY', 'MSA': 'GLY', 'MSE': 'MET', 'MVA': 'MVA', 'NEM': 'HIS',
    'NEP': 'HIS', 'NLE': 'NLE',
    'NLN': 'LEU', 'NLP': 'LEU', 'NMC': 'GLY', 'OAS': 'SER', 'OCS': 'CYS', 'OMT': 'MET', 'PAQ': 'TYR', 'PCA': 'GLU',
    'PEC': 'CYS', 'PHI': 'PHE',
    'PHL': 'PHE', 'PR3': 'CYS', 'PRR': 'ALA', 'PTR': 'PTR', 'PYX': 'CYS', 'SAC': 'SER', 'SAR': 'SAR', 'SCH': 'CYS',
    'SCS': 'CYS', 'SCY': 'CYS',
    'SEL': 'SER', 'SET': 'SER', 'SHC': 'CYS', 'SHR': 'LYS', 'SMC': 'CYS', 'SOC': 'CYS', 'STY': 'TYR',
    'SVA': 'SER', 'TIH': 'ALA',
    'TPL': 'TRP', 'TPO': 'TPO', 'TPQ': 'ALA', 'TRG': 'LYS', 'TRO': 'TRP', 'TYB': 'TYR', 'TYI': 'TYR', 'TYQ': 'TYR',
    'TYS': 'TYS', 'TYY': 'TYR',
    'ALA': 'ALA', 'CYS': 'CYS', 'ASP': 'ASP', 'GLU': 'GLU', 'PHE': 'PHE', 'GLY': 'GLY', 'HIS': 'HIS', 'ILE': 'ILE',
    'LYS': 'LYS', 'LEU': 'LEU',
    'MET': 'MET', 'ASN': 'ASN', 'PRO': 'PRO', 'GLN': 'GLN', 'ARG': 'ARG', 'SER': 'SER', 'THR': 'THR', 'VAL': 'VAL',
    'TRP': 'TRP', 'TYR': 'TYR', 'M3L': 'M3L',
    'UNK': 'UNK', 'SEP': 'SEP', # The nsaa we could deal with
}

ressymb_to_resindex = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
    'X': 20,
}

resindex_to_ressymb = {}
for k, v in ressymb_to_resindex.items(): resindex_to_ressymb[v] = k

BACKBONE_FRAME = 0
OMEGA_FRAME = 1
PHI_FRAME = 2
PSI_FRAME = 3
CHI1_FRAME, CHI2_FRAME, CHI3_FRAME, CHI4_FRAME, CHI5_FRAME, CHI6_FRAME = 4, 5, 6, 7, 8, 9


class AA(enum.IntEnum):
    ALA = 0
    CYS = 1
    ASP = 2
    GLU = 3
    PHE = 4
    GLY = 5
    HIS = 6
    ILE = 7
    LYS = 8
    LEU = 9
    MET = 10
    ASN = 11
    PRO = 12
    GLN = 13
    ARG = 14
    SER = 15
    THR = 16
    VAL = 17
    TRP = 18
    TYR = 19
    SEP = 20
    TYS = 21
    PTR = 22
    MLE = 23
    M3L = 24
    ALY = 25
    TPO = 26
    DAL = 27
    HYP = 28
    MVA = 29
    DLY = 30
    DLE = 31
    SAR = 32
    NLE = 33
    BMT = 34
    DGL = 35
    DPR = 36
    DTR = 37
    UNK = 38

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str) and len(value) == 3:  # three representation
            if value in non_standard_residue_substitutions:
                value = non_standard_residue_substitutions[value]
            if value in cls._member_names_:
                return getattr(cls, value)
        elif isinstance(value, str) and len(value) == 1:  # one representation
            if value in ressymb_to_resindex:
                return cls(ressymb_to_resindex[value])

        return super()._missing_(value)

    def __str__(self):
        return self.name

    @classmethod
    def is_aa(cls, value):
        return (value in ressymb_to_resindex) or \
            (value in non_standard_residue_substitutions) or \
            (value in cls._member_names_)


num_aa_types = len(AA)


##
# Atom identities

max_num_heavyatoms = 17
max_num_hydrogens = 22
max_num_allatoms = max_num_heavyatoms + max_num_hydrogens

class BBHeavyAtom(enum.IntEnum):
    N = 0;
    CA = 1;
    C = 2;
    O = 3;
    CB = 4;
    OXT = max_num_heavyatoms-1;



restype_to_heavyatom_names = {
    AA.ALA: ['N', 'CA', 'C', 'O', 'CB', '', '', '', '', '', '', '', '', '', '', '', 'OXT'],
    AA.ARG: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2', '', '', '', '', '', 'OXT'],
    AA.ASN: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2', '', '', '', '', '', '', '', '', 'OXT'],
    AA.ASP: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2', '', '', '', '', '', '', '', '', 'OXT'],
    AA.CYS: ['N', 'CA', 'C', 'O', 'CB', 'SG', '', '', '', '', '', '', '', '', '', '', 'OXT'],
    AA.GLN: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2', '', '', '', '', '', '', '', 'OXT'],
    AA.GLU: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2', '', '', '', '', '', '', '', 'OXT'],
    AA.GLY: ['N', 'CA', 'C', 'O', '', '', '', '', '', '', '', '', '', '', '', '', 'OXT'],
    AA.HIS: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2', '', '', '', '', '', '', 'OXT'],
    AA.ILE: ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '', '', '', '', '', '', '', '', 'OXT'],
    AA.LEU: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', '', '', '', '', '', '', '', '', 'OXT'],
    AA.LYS: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', '', '', '', '', '', '', '', 'OXT'],
    AA.MET: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE', '', '', '', '', '', '', '', '', 'OXT'],
    AA.PHE: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', '', '', '', '', '', 'OXT'],
    AA.PRO: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', '', '', '', '', '', '', '', '', '', 'OXT'],
    AA.SER: ['N', 'CA', 'C', 'O', 'CB', 'OG', '', '', '', '', '', '', '', '', '', '', 'OXT'],
    AA.SEP: ['N', 'CA', 'C', 'O', 'CB', 'OG', 'P', 'O1P', 'O2P', 'O3P', '', '', '', '', '', '', 'OXT'],
    AA.THR: ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '', '', '', '', '', '', '', '', '', 'OXT'],
    AA.TRP: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2', '', '', 'OXT'],
    AA.TYR: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', '', '', '', '', 'OXT'],
    AA.TYS: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', 'S', 'O1', 'O2', 'O3', 'OXT'],
    AA.PTR: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', 'P', 'O1P', 'O2P', 'O3P', 'OXT'],
    AA.VAL: ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '', '', '', '', '', '', '', '', '', 'OXT'],
    AA.MLE: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CN', '', '', '', '', '', '', '', 'OXT'],
    AA.M3L: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', 'CM1', 'CM2', 'CM3', '', '', '', '', 'OXT'],
    AA.ALY: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', 'CH', 'CH3', 'OH', '', '', '', '', 'OXT'],
    AA.TPO: ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', 'P', 'O1P', 'O2P', 'O3P', '', '', '', '', '', 'OXT'],
    AA.DAL: ['N', 'CA', 'C', 'O', 'CB', '', '', '', '', '', '', '', '', '', '', '', 'OXT'],
    AA.HYP: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OD1', '', '', '', '', '', '', '', '', 'OXT'],
    AA.MVA: ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CN', '', '', '', '', '', '', '', '', 'OXT'],
    AA.DLY: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', '', '', '', '', '', '', '', 'OXT'],
    AA.DLE: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', '', '', '', '', '', '', '', '', 'OXT'],
    AA.SAR: ['N', 'CA', 'C', 'O', 'CN', '', '', '', '', '', '', '', '', '', '', '', 'OXT'],
    AA.NLE: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', '', '', '', '', '', '', '', '', 'OXT'],
    AA.BMT: ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', 'CD1', 'CD2', 'CE', 'CZ', 'CH', 'CN', '', '', '', 'OXT'],
    AA.DGL: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2', '', '', '', '', '', '', '', 'OXT'],
    AA.DPR: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', '', '', '', '', '', '', '', '', '', 'OXT'],
    AA.DTR: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2', '', '', 'OXT'],
    AA.UNK: ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
}
for names in restype_to_heavyatom_names.values(): assert len(names) == max_num_heavyatoms

restype_to_hydrogen_names = {
    AA.ALA: ['H', 'H2', 'H3', 'HA', 'HB1', 'HB2', 'HB3', 'HXT', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
    AA.CYS: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HG', 'HXT', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
    AA.ASP: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HD2', 'HXT', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
    AA.GLU: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HG2', 'HG3', 'HE2', 'HXT', '', '', '', '', '', '', '', '', '', '', '', ''],
    AA.PHE: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HD1', 'HD2', 'HE1', 'HE2', 'HZ', 'HXT', '', '', '', '', '', '', '', '', '', ''],
    AA.GLY: ['H', 'H2', 'H3', 'HA2', 'HA3', 'HXT', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
    AA.HIS: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HD1', 'HD2', 'HE1', 'HE2', 'HXT', '', '', '', '', '', '', '', '', '', '', ''],
    AA.ILE: ['H', 'H2', 'H3', 'HA', 'HB', 'HG12', 'HG13', 'HG21', 'HG22', 'HG23', 'HD11', 'HD12', 'HD13', 'HXT', '', '', '', '', '', '', '', ''],
    AA.LYS: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HG2', 'HG3', 'HD2', 'HD3', 'HE2', 'HE3', 'HZ1', 'HZ2', 'HZ3', 'HXT', '', '', '', '', '', ''],
    AA.LEU: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HG', 'HD11', 'HD12', 'HD13', 'HD21', 'HD22', 'HD23', 'HXT', '', '', '', '', '', '', '', ''],
    AA.MET: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HG2', 'HG3', 'HE1', 'HE2', 'HE3', 'HXT', '', '', '', '', '', '', '', '', '', ''],
    AA.ASN: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HD21', 'HD22', 'HXT', '', '', '', '', '', '', '', '', '', '', '', '', ''],
    AA.PRO: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HG2', 'HG3', 'HD2', 'HD3', 'HXT', '', '', '', '', '', '', '', '', '', '', ''],
    AA.GLN: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HG2', 'HG3', 'HE21', 'HE22', 'HXT', '', '', '', '', '', '', '', '', '', '', ''],
    AA.ARG: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HG2', 'HG3', 'HD2', 'HD3', 'HE', 'HH11', 'HH12', 'HH21', 'HH22', 'HXT', '', '', '', '', '', ''],
    AA.SER: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HG', 'HXT', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
    AA.SEP: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HXT', 'HOP2', 'HOP3', '', '', '', '', '', '', '', '', '', '', '', '', ''],
    AA.THR: ['H', 'H2', 'H3', 'HA', 'HB', 'HG1', 'HG21', 'HG22', 'HG23', 'HXT', '', '', '', '', '', '', '', '', '', '', '', ''],
    AA.VAL: ['H', 'H2', 'H3', 'HA', 'HB', 'HG11', 'HG12', 'HG13', 'HG21', 'HG22', 'HG23', 'HXT', '', '', '', '', '', '', '', '', '', ''],
    AA.TRP: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HD1', 'HE1', 'HE3', 'HZ2', 'HZ3', 'HH2', 'HXT', '', '', '', '', '', '', '', '', ''],
    AA.TYR: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HD1', 'HD2', 'HE1', 'HE2', 'HH', 'HXT', '', '', '', '', '', '', '', '', '', ''],
    AA.TYS: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HD1', 'HD2', 'HE1', 'HE2', 'HO3', 'HXT', '', '', '', '', '', '', '', '', '', ''],
    AA.PTR: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HD1', 'HD2', 'HE1', 'HE2', 'HO2P', 'HO3P', 'HXT', '', '', '', '', '', '', '', '', ''],
    AA.MLE: ['H', 'H2', 'HA', 'HB2', 'HB3', 'HG', 'HD11', 'HD12', 'HD13', 'HD21', 'HD22', 'HD23', 'HXT', 'HN1', 'HN2', 'HN3', '', '', '', '', '', ''],
    AA.M3L: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HG2', 'HG3', 'HD2', 'HD3', 'HE2', 'HE3', 'HM11', 'HM12', 'HM13', 'HM21', 'HM22', 'HM23', 'HM31', 'HM32', 'HM33', 'HXT'],
    AA.ALY: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HG2', 'HG3', 'HD2', 'HD3', 'HE2', 'HE3', 'HZ', 'HH31', 'HH32', 'HH33', 'HXT', '', '', '', '', ''],
    AA.TPO: ['H', 'H2', 'H3', 'HA', 'HB', 'HG21', 'HG22', 'HG23', 'HOP2', 'HOP3', 'HXT', '', '', '', '', '', '', '', '', '', '', ''],
    AA.DAL: ['H', 'H2', 'H3', 'HA', 'HB1', 'HB2', 'HB3', 'HXT', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
    AA.HYP: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HG', 'HD1', 'HD22', 'HD23', 'HXT', '', '', '', '', '', '', '', '', '', '', ''],
    AA.MVA: ['H', 'H2', 'HA', 'HB', 'HG11', 'HG12', 'HG13', 'HG21', 'HG22', 'HG23', 'HN1', 'HN2', 'HN3', 'HXT', '', '', '', '', '', '', '', ''],
    AA.DLY: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HG2', 'HG3', 'HD2', 'HD3', 'HE2', 'HE3', 'HZ1', 'HZ2', 'HZ3', 'HXT', '', '', '', '', '', ''],
    AA.DLE: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HG', 'HD11', 'HD12', 'HD13', 'HD21', 'HD22', 'HD23', 'HXT', '', '', '', '', '', '', '', ''],
    AA.SAR: ['H', 'H2', 'HA2', 'HA3', 'HN1', 'HN2', 'HN3', 'HXT', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
    AA.NLE: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HG2', 'HG3', 'HD2', 'HD3', 'HE1', 'HE2', 'HE3', 'HXT', '', '', '', '', '', '', '', ''],
    AA.BMT: ['H', 'H2', 'HA', 'HB', 'HG1', 'HG2', 'HD11', 'HD12', 'HD13', 'HD22', 'HD23', 'HE', 'HZ', 'HH1', 'HH2', 'HH3', 'HN1', 'HN2', 'HN3', 'HXT', '', ''],
    AA.DGL: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HG2', 'HG3', 'HE2', 'HXT', '', '', '', '', '', '', '', '', '', '', '', ''],
    AA.DPR: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HG2', 'HG3', 'HD2', 'HD3', 'HXT', '', '', '', '', '', '', '', '', '', '', ''],
    AA.DTR: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HD1', 'HE1', 'HE3', 'HZ2', 'HZ3', 'HH2', 'HXT', '', '', '', '', '', '', '', '', ''],
    AA.UNK: ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
}

for names in restype_to_hydrogen_names.values(): assert len(names) == max_num_hydrogens

restype_to_allatom_names = {
    restype: restype_to_heavyatom_names[restype] + restype_to_hydrogen_names[restype]
    for restype in AA
}

restype_atom14_name_to_index = {
    resname: {name: index for index, name in enumerate(atoms) if name != ""}
    for resname, atoms in restype_to_heavyatom_names.items()
}


##
# Bond identities

class BondType(enum.IntEnum):
    NoBond = 0
    Single = 1
    Double = 2
    Triple = 3
    AromaticSingle = 5
    AromaticDouble = 6


BT = BondType
restype_to_bonded_atom_name_pairs = {
    AA.ALA: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle),
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('CB', 'HB1', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle),
        ('OXT', 'HXT', BT.AromaticSingle), ],
    AA.CYS: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle),
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('CB', 'SG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle),
        ('SG', 'HG', BT.AromaticSingle), ('OXT', 'HXT', BT.AromaticSingle), ],
    AA.ASP: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle),
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle),
        ('CG', 'OD1', BT.AromaticDouble), ('CG', 'OD2', BT.AromaticSingle), ('OD2', 'HD2', BT.AromaticSingle),
        ('OXT', 'HXT', BT.AromaticSingle), ],
    AA.GLU: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle),
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle),
        ('CG', 'CD', BT.AromaticSingle), ('CG', 'HG2', BT.AromaticSingle), ('CG', 'HG3', BT.AromaticSingle),
        ('CD', 'OE1', BT.AromaticDouble), ('CD', 'OE2', BT.AromaticSingle), ('OE2', 'HE2', BT.AromaticSingle),
        ('OXT', 'HXT', BT.AromaticSingle), ],
    AA.PHE: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle),
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle),
        ('CG', 'CD1', BT.AromaticDouble), ('CG', 'CD2', BT.AromaticSingle), ('CD1', 'CE1', BT.AromaticSingle),
        ('CD1', 'HD1', BT.AromaticSingle), ('CD2', 'CE2', BT.AromaticDouble), ('CD2', 'HD2', BT.AromaticSingle),
        ('CE1', 'CZ', BT.AromaticDouble), ('CE1', 'HE1', BT.AromaticSingle), ('CE2', 'CZ', BT.AromaticSingle),
        ('CE2', 'HE2', BT.AromaticSingle), ('CZ', 'HZ', BT.AromaticSingle), ('OXT', 'HXT', BT.AromaticSingle), ],
    AA.GLY: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'HA2', BT.AromaticSingle),
        ('CA', 'HA3', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('OXT', 'HXT', BT.AromaticSingle), ],
    AA.HIS: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle),
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle),
        ('CG', 'ND1', BT.AromaticSingle), ('CG', 'CD2', BT.AromaticDouble), ('ND1', 'CE1', BT.AromaticDouble),
        ('ND1', 'HD1', BT.AromaticSingle), ('CD2', 'NE2', BT.AromaticSingle), ('CD2', 'HD2', BT.AromaticSingle),
        ('CE1', 'NE2', BT.AromaticSingle), ('CE1', 'HE1', BT.AromaticSingle), ('NE2', 'HE2', BT.AromaticSingle),
        ('OXT', 'HXT', BT.AromaticSingle), ],
    AA.ILE: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle),
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('CB', 'CG1', BT.AromaticSingle), ('CB', 'CG2', BT.AromaticSingle), ('CB', 'HB', BT.AromaticSingle),
        ('CG1', 'CD1', BT.AromaticSingle), ('CG1', 'HG12', BT.AromaticSingle), ('CG1', 'HG13', BT.AromaticSingle),
        ('CG2', 'HG21', BT.AromaticSingle), ('CG2', 'HG22', BT.AromaticSingle), ('CG2', 'HG23', BT.AromaticSingle),
        ('CD1', 'HD11', BT.AromaticSingle), ('CD1', 'HD12', BT.AromaticSingle), ('CD1', 'HD13', BT.AromaticSingle),
        ('OXT', 'HXT', BT.AromaticSingle), ],
    AA.LYS: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle),
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle),
        ('CG', 'CD', BT.AromaticSingle), ('CG', 'HG2', BT.AromaticSingle), ('CG', 'HG3', BT.AromaticSingle),
        ('CD', 'CE', BT.AromaticSingle), ('CD', 'HD2', BT.AromaticSingle), ('CD', 'HD3', BT.AromaticSingle),
        ('CE', 'NZ', BT.AromaticSingle), ('CE', 'HE2', BT.AromaticSingle), ('CE', 'HE3', BT.AromaticSingle),
        ('NZ', 'HZ1', BT.AromaticSingle), ('NZ', 'HZ2', BT.AromaticSingle), ('NZ', 'HZ3', BT.AromaticSingle),
        ('OXT', 'HXT', BT.AromaticSingle), ],
    AA.LEU: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle),
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle),
        ('CG', 'CD1', BT.AromaticSingle), ('CG', 'CD2', BT.AromaticSingle), ('CG', 'HG', BT.AromaticSingle),
        ('CD1', 'HD11', BT.AromaticSingle), ('CD1', 'HD12', BT.AromaticSingle), ('CD1', 'HD13', BT.AromaticSingle),
        ('CD2', 'HD21', BT.AromaticSingle), ('CD2', 'HD22', BT.AromaticSingle), ('CD2', 'HD23', BT.AromaticSingle),
        ('OXT', 'HXT', BT.AromaticSingle), ],
    AA.MET: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle),
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle),
        ('CG', 'SD', BT.AromaticSingle), ('CG', 'HG2', BT.AromaticSingle), ('CG', 'HG3', BT.AromaticSingle),
        ('SD', 'CE', BT.AromaticSingle), ('CE', 'HE1', BT.AromaticSingle), ('CE', 'HE2', BT.AromaticSingle),
        ('CE', 'HE3', BT.AromaticSingle), ('OXT', 'HXT', BT.AromaticSingle), ],
    AA.ASN: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle),
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle),
        ('CG', 'OD1', BT.AromaticDouble), ('CG', 'ND2', BT.AromaticSingle), ('ND2', 'HD21', BT.AromaticSingle),
        ('ND2', 'HD22', BT.AromaticSingle), ('OXT', 'HXT', BT.AromaticSingle), ],
    AA.PRO: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('N', 'CD', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle),
        ('CA', 'CB', BT.AromaticSingle), ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble),
        ('C', 'OXT', BT.AromaticSingle), ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle),
        ('CB', 'HB3', BT.AromaticSingle), ('CG', 'CD', BT.AromaticSingle), ('CG', 'HG2', BT.AromaticSingle),
        ('CG', 'HG3', BT.AromaticSingle), ('CD', 'HD2', BT.AromaticSingle), ('CD', 'HD3', BT.AromaticSingle),
        ('OXT', 'HXT', BT.AromaticSingle), ],
    AA.GLN: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle),
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle),
        ('CG', 'CD', BT.AromaticSingle), ('CG', 'HG2', BT.AromaticSingle), ('CG', 'HG3', BT.AromaticSingle),
        ('CD', 'OE1', BT.AromaticDouble), ('CD', 'NE2', BT.AromaticSingle), ('NE2', 'HE21', BT.AromaticSingle),
        ('NE2', 'HE22', BT.AromaticSingle), ('OXT', 'HXT', BT.AromaticSingle), ],
    AA.ARG: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle),
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle),
        ('CG', 'CD', BT.AromaticSingle), ('CG', 'HG2', BT.AromaticSingle), ('CG', 'HG3', BT.AromaticSingle),
        ('CD', 'NE', BT.AromaticSingle), ('CD', 'HD2', BT.AromaticSingle), ('CD', 'HD3', BT.AromaticSingle),
        ('NE', 'CZ', BT.AromaticSingle), ('NE', 'HE', BT.AromaticSingle), ('CZ', 'NH1', BT.AromaticSingle),
        ('CZ', 'NH2', BT.AromaticDouble), ('NH1', 'HH11', BT.AromaticSingle), ('NH1', 'HH12', BT.AromaticSingle),
        ('NH2', 'HH21', BT.AromaticSingle), ('NH2', 'HH22', BT.AromaticSingle), ('OXT', 'HXT', BT.AromaticSingle), ],
    AA.SER: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle),
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('CB', 'OG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle),
        ('OG', 'HG', BT.AromaticSingle), ('OXT', 'HXT', BT.AromaticSingle), ],
    AA.SEP: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle),
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('CB', 'OG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle),
        ('OG', 'P', BT.AromaticSingle), ('P', 'O1P', BT.AromaticDouble), ('P', 'O2P', BT.AromaticSingle),
        ('P', 'O3P', BT.AromaticSingle), ('O2P', 'HOP2', BT.AromaticSingle), ('O3P', 'HOP3', BT.AromaticSingle),
        ('OXT', 'HXT', BT.AromaticSingle), ],
    AA.THR: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle),
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('CB', 'OG1', BT.AromaticSingle), ('CB', 'CG2', BT.AromaticSingle), ('CB', 'HB', BT.AromaticSingle),
        ('OG1', 'HG1', BT.AromaticSingle), ('CG2', 'HG21', BT.AromaticSingle), ('CG2', 'HG22', BT.AromaticSingle),
        ('CG2', 'HG23', BT.AromaticSingle), ('OXT', 'HXT', BT.AromaticSingle), ],
    AA.VAL: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle),
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('CB', 'CG1', BT.AromaticSingle), ('CB', 'CG2', BT.AromaticSingle), ('CB', 'HB', BT.AromaticSingle),
        ('CG1', 'HG11', BT.AromaticSingle), ('CG1', 'HG12', BT.AromaticSingle), ('CG1', 'HG13', BT.AromaticSingle),
        ('CG2', 'HG21', BT.AromaticSingle), ('CG2', 'HG22', BT.AromaticSingle), ('CG2', 'HG23', BT.AromaticSingle),
        ('OXT', 'HXT', BT.AromaticSingle), ],
    AA.TRP: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle),
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle),
        ('CG', 'CD1', BT.AromaticDouble), ('CG', 'CD2', BT.AromaticSingle), ('CD1', 'NE1', BT.AromaticSingle),
        ('CD1', 'HD1', BT.AromaticSingle), ('CD2', 'CE2', BT.AromaticDouble), ('CD2', 'CE3', BT.AromaticSingle),
        ('NE1', 'CE2', BT.AromaticSingle), ('NE1', 'HE1', BT.AromaticSingle), ('CE2', 'CZ2', BT.AromaticSingle),
        ('CE3', 'CZ3', BT.AromaticDouble), ('CE3', 'HE3', BT.AromaticSingle), ('CZ2', 'CH2', BT.AromaticDouble),
        ('CZ2', 'HZ2', BT.AromaticSingle), ('CZ3', 'CH2', BT.AromaticSingle), ('CZ3', 'HZ3', BT.AromaticSingle),
        ('CH2', 'HH2', BT.AromaticSingle), ('OXT', 'HXT', BT.AromaticSingle), ],
    AA.TYR: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle),
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle),
        ('CG', 'CD1', BT.AromaticDouble), ('CG', 'CD2', BT.AromaticSingle), ('CD1', 'CE1', BT.AromaticSingle),
        ('CD1', 'HD1', BT.AromaticSingle), ('CD2', 'CE2', BT.AromaticDouble), ('CD2', 'HD2', BT.AromaticSingle),
        ('CE1', 'CZ', BT.AromaticDouble), ('CE1', 'HE1', BT.AromaticSingle), ('CE2', 'CZ', BT.AromaticSingle),
        ('CE2', 'HE2', BT.AromaticSingle), ('CZ', 'OH', BT.AromaticSingle), ('OH', 'HH', BT.AromaticSingle),
        ('OXT', 'HXT', BT.AromaticSingle), ],
    AA.TYS: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle),
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle),
        ('CG', 'CD1', BT.AromaticDouble), ('CG', 'CD2', BT.AromaticSingle), ('CD1', 'CE1', BT.AromaticSingle),
        ('CD1', 'HD1', BT.AromaticSingle), ('CD2', 'CE2', BT.AromaticDouble), ('CD2', 'HD2', BT.AromaticSingle),
        ('CE1', 'CZ', BT.AromaticDouble), ('CE1', 'HE1', BT.AromaticSingle), ('CE2', 'CZ', BT.AromaticSingle),
        ('CE2', 'HE2', BT.AromaticSingle), ('CZ', 'OH', BT.AromaticSingle), ('OH', 'S', BT.AromaticSingle),
        ('S', 'O1', BT.AromaticDouble), ('S', 'O2', BT.AromaticDouble), ('S', 'O3', BT.AromaticSingle),
        ('O3', 'HO3', BT.AromaticSingle), ('OXT', 'HXT', BT.AromaticSingle), ],
    AA.PTR: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle),
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle),
        ('CG', 'CD1', BT.AromaticDouble), ('CG', 'CD2', BT.AromaticSingle), ('CD1', 'CE1', BT.AromaticSingle),
        ('CD1', 'HD1', BT.AromaticSingle), ('CD2', 'CE2', BT.AromaticDouble), ('CD2', 'HD2', BT.AromaticSingle),
        ('CE1', 'CZ', BT.AromaticDouble), ('CE1', 'HE1', BT.AromaticSingle), ('CE2', 'CZ', BT.AromaticSingle),
        ('CE2', 'HE2', BT.AromaticSingle), ('CZ', 'OH', BT.AromaticSingle), ('OH', 'P', BT.AromaticSingle),
        ('P', 'O1P', BT.AromaticDouble), ('P', 'O2P', BT.AromaticSingle), ('P', 'O3P', BT.AromaticSingle),
        ('O2P', 'HO2P', BT.AromaticSingle), ('O3P', 'HO3P', BT.AromaticSingle), ('OXT', 'HXT', BT.AromaticSingle), ],
    AA.MLE: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'CN', BT.Single),
        ('CN', 'HN1', BT.AromaticSingle), ('CN', 'HN2', BT.AromaticSingle), ('CN', 'HN3', BT.AromaticSingle),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle),
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle),
        ('CG', 'CD1', BT.AromaticSingle), ('CG', 'CD2', BT.AromaticSingle), ('CG', 'HG', BT.AromaticSingle),
        ('CD1', 'HD11', BT.AromaticSingle), ('CD1', 'HD12', BT.AromaticSingle), ('CD1', 'HD13', BT.AromaticSingle),
        ('CD2', 'HD21', BT.AromaticSingle), ('CD2', 'HD22', BT.AromaticSingle), ('CD2', 'HD23', BT.AromaticSingle),
        ('OXT', 'HXT', BT.AromaticSingle), ],
    AA.M3L: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle),
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle),
        ('CG', 'CD', BT.AromaticSingle), ('CG', 'HG2', BT.AromaticSingle), ('CG', 'HG3', BT.AromaticSingle),
        ('CD', 'CE', BT.AromaticSingle), ('CD', 'HD2', BT.AromaticSingle), ('CD', 'HD3', BT.AromaticSingle),
        ('CE', 'NZ', BT.AromaticSingle), ('CE', 'HE2', BT.AromaticSingle), ('CE', 'HE3', BT.AromaticSingle),
        ('NZ', 'CM1', BT.AromaticSingle), ('NZ', 'CM2', BT.AromaticSingle), ('NZ', 'CM3', BT.AromaticSingle),
        ('CM1', 'HM11', BT.AromaticSingle), ('CM1', 'HM12', BT.AromaticSingle), ('CM1', 'HM13', BT.AromaticSingle),
        ('CM2', 'HM21', BT.AromaticSingle), ('CM2', 'HM23', BT.AromaticSingle), ('CM2', 'HM23', BT.AromaticSingle),
        ('CM3', 'HM31', BT.AromaticSingle), ('CM3', 'HM33', BT.AromaticSingle), ('CM3', 'HM33', BT.AromaticSingle),
        ('OXT', 'HXT', BT.AromaticSingle), ],    
    AA.ALY: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle),
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle),
        ('CG', 'CD', BT.AromaticSingle), ('CG', 'HG2', BT.AromaticSingle), ('CG', 'HG3', BT.AromaticSingle),
        ('CD', 'CE', BT.AromaticSingle), ('CD', 'HD2', BT.AromaticSingle), ('CD', 'HD3', BT.AromaticSingle),
        ('CE', 'NZ', BT.AromaticSingle), ('CE', 'HE2', BT.AromaticSingle), ('CE', 'HE3', BT.AromaticSingle),
        ('NZ', 'HZ', BT.AromaticSingle), ('NZ', 'CH', BT.AromaticSingle), ('CH', 'OH', BT.AromaticSingle),
        ('CH', 'CH3', BT.AromaticSingle), ('CH3', 'HH31', BT.AromaticSingle), ('CH3', 'HH32', BT.AromaticSingle),
        ('CH3', 'HH33', BT.AromaticSingle), ('OXT', 'HXT', BT.AromaticSingle), ], 
    AA.TPO: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle),
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('CB', 'OG1', BT.AromaticSingle), ('CB', 'CG2', BT.AromaticSingle), ('CB', 'HB', BT.AromaticSingle),
        ('OG1', 'P', BT.AromaticSingle), ('CG2', 'HG21', BT.AromaticSingle), ('CG2', 'HG22', BT.AromaticSingle),
        ('CG2', 'HG23', BT.AromaticSingle), ('OXT', 'HXT', BT.AromaticSingle), ('P', 'O1P', BT.AromaticDouble), 
        ('P', 'O2P', BT.AromaticSingle), ('P', 'O3P', BT.AromaticSingle), ('O2P', 'HOP2', BT.AromaticSingle), ('O3P', 'HOP3', BT.AromaticSingle), ],    
    AA.DAL: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle),
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('CB', 'HB1', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle),
        ('OXT', 'HXT', BT.AromaticSingle), ],
    AA.HYP: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('N', 'CD', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle),
        ('CA', 'CB', BT.AromaticSingle), ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble),
        ('C', 'OXT', BT.AromaticSingle), ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle),
        ('CB', 'HB3', BT.AromaticSingle), ('CG', 'CD', BT.AromaticSingle), ('CG', 'HG', BT.AromaticSingle),
        ('CG', 'OD1', BT.AromaticSingle), ('CD', 'HD22', BT.AromaticSingle), ('CD', 'HD23', BT.AromaticSingle),
        ('OXT', 'HXT', BT.AromaticSingle), ('HD1', 'OD1', BT.AromaticSingle),],
    AA.MVA: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'CN', BT.AromaticSingle),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle),
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('CB', 'CG1', BT.AromaticSingle), ('CB', 'CG2', BT.AromaticSingle), ('CB', 'HB', BT.AromaticSingle),
        ('CG1', 'HG11', BT.AromaticSingle), ('CG1', 'HG12', BT.AromaticSingle), ('CG1', 'HG13', BT.AromaticSingle),
        ('CG2', 'HG21', BT.AromaticSingle), ('CG2', 'HG22', BT.AromaticSingle), ('CG2', 'HG23', BT.AromaticSingle),
        ('OXT', 'HXT', BT.AromaticSingle), ('CN', 'HN1', BT.AromaticSingle), ('CN', 'HN2', BT.AromaticSingle),
        ('CN', 'HN3', BT.AromaticSingle),],        
    AA.DLY: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle),
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle),
        ('CG', 'CD', BT.AromaticSingle), ('CG', 'HG2', BT.AromaticSingle), ('CG', 'HG3', BT.AromaticSingle),
        ('CD', 'CE', BT.AromaticSingle), ('CD', 'HD2', BT.AromaticSingle), ('CD', 'HD3', BT.AromaticSingle),
        ('CE', 'NZ', BT.AromaticSingle), ('CE', 'HE2', BT.AromaticSingle), ('CE', 'HE3', BT.AromaticSingle),
        ('NZ', 'HZ1', BT.AromaticSingle), ('NZ', 'HZ2', BT.AromaticSingle), ('NZ', 'HZ3', BT.AromaticSingle),
        ('OXT', 'HXT', BT.AromaticSingle), ],        
    AA.DLE: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle),
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle),
        ('CG', 'CD1', BT.AromaticSingle), ('CG', 'CD2', BT.AromaticSingle), ('CG', 'HG', BT.AromaticSingle),
        ('CD1', 'HD11', BT.AromaticSingle), ('CD1', 'HD12', BT.AromaticSingle), ('CD1', 'HD13', BT.AromaticSingle),
        ('CD2', 'HD21', BT.AromaticSingle), ('CD2', 'HD22', BT.AromaticSingle), ('CD2', 'HD23', BT.AromaticSingle),
        ('OXT', 'HXT', BT.AromaticSingle), ],    
    AA.SAR: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'CN', BT.AromaticSingle),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'HA2', BT.AromaticSingle),
        ('CA', 'HA3', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('OXT', 'HXT', BT.AromaticSingle), ('CN', 'HN1', BT.AromaticSingle), ('CN', 'HN2', BT.AromaticSingle),
        ('CN', 'HN3', BT.AromaticSingle),],    
    AA.NLE: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle),
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle),
        ('CG', 'CD', BT.AromaticSingle), ('CG', 'HG2', BT.AromaticSingle), ('CG', 'HG3', BT.AromaticSingle),
        ('CD', 'HD2', BT.AromaticSingle), ('CD', 'HD3', BT.AromaticSingle), ('CE', 'CD', BT.AromaticSingle),
        ('CE', 'HE1', BT.AromaticSingle), ('CE', 'HE2', BT.AromaticSingle), ('CE', 'HE3', BT.AromaticSingle),
        ('OXT', 'HXT', BT.AromaticSingle), ],
    AA.BMT: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'CN', BT.AromaticSingle),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle),
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('CB', 'OG1', BT.AromaticSingle), ('CB', 'CG2', BT.AromaticSingle), ('CB', 'HB', BT.AromaticSingle),
        ('OG1', 'HG1', BT.AromaticSingle), ('CG2', 'CD1', BT.AromaticSingle), ('CD1', 'HD11', BT.AromaticSingle),
        ('CD1', 'HD12', BT.AromaticSingle), ('CD1', 'HD13', BT.AromaticSingle), ('CG2', 'CD2', BT.AromaticSingle),
        ('CG2', 'HG2', BT.AromaticSingle), ('CD2', 'CE', BT.AromaticSingle), ('CD2', 'HD22', BT.AromaticSingle),
        ('CD2', 'HD23', BT.AromaticSingle), ('CE', 'CZ', BT.AromaticSingle), ('CE', 'HE', BT.AromaticSingle),
        ('CZ', 'HZ', BT.AromaticSingle), ('CZ', 'CH', BT.AromaticSingle), ('CH', 'HH1', BT.AromaticSingle),
        ('CH', 'HH2', BT.AromaticSingle), ('CH', 'HH3', BT.AromaticSingle), ('CN', 'HN1', BT.AromaticSingle),
        ('CN', 'HN2', BT.AromaticSingle), ('CN', 'HN3', BT.AromaticSingle), ('OXT', 'HXT', BT.AromaticSingle), ],        
    AA.DGL: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle),
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle),
        ('CG', 'CD', BT.AromaticSingle), ('CG', 'HG2', BT.AromaticSingle), ('CG', 'HG3', BT.AromaticSingle),
        ('CD', 'OE1', BT.AromaticDouble), ('CD', 'OE2', BT.AromaticSingle), ('OE2', 'HE2', BT.AromaticSingle),
        ('OXT', 'HXT', BT.AromaticSingle), ],
    AA.DPR: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('N', 'CD', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle),
        ('CA', 'CB', BT.AromaticSingle), ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble),
        ('C', 'OXT', BT.AromaticSingle), ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle),
        ('CB', 'HB3', BT.AromaticSingle), ('CG', 'CD', BT.AromaticSingle), ('CG', 'HG2', BT.AromaticSingle),
        ('CG', 'HG3', BT.AromaticSingle), ('CD', 'HD2', BT.AromaticSingle), ('CD', 'HD3', BT.AromaticSingle),
        ('OXT', 'HXT', BT.AromaticSingle), ],
    AA.DTR: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single),
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle),
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle),
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle),
        ('CG', 'CD1', BT.AromaticDouble), ('CG', 'CD2', BT.AromaticSingle), ('CD1', 'NE1', BT.AromaticSingle),
        ('CD1', 'HD1', BT.AromaticSingle), ('CD2', 'CE2', BT.AromaticDouble), ('CD2', 'CE3', BT.AromaticSingle),
        ('NE1', 'CE2', BT.AromaticSingle), ('NE1', 'HE1', BT.AromaticSingle), ('CE2', 'CZ2', BT.AromaticSingle),
        ('CE3', 'CZ3', BT.AromaticDouble), ('CE3', 'HE3', BT.AromaticSingle), ('CZ2', 'CH2', BT.AromaticDouble),
        ('CZ2', 'HZ2', BT.AromaticSingle), ('CZ3', 'CH2', BT.AromaticSingle), ('CZ3', 'HZ3', BT.AromaticSingle),
        ('CH2', 'HH2', BT.AromaticSingle), ('OXT', 'HXT', BT.AromaticSingle), ],                    
    AA.UNK: [],
}

restype_to_allatom_bond_matrix = {
    restype: torch.zeros([max_num_allatoms, max_num_allatoms], dtype=torch.long)
    for restype in AA
}
restype_to_heavyatom_bond_matrix = {
    restype: torch.zeros([max_num_heavyatoms, max_num_heavyatoms], dtype=torch.long)
    for restype in AA
}


def _make_bond_matrices():
    for restype in AA:
        for atom1_name, atom2_name, bond_type in restype_to_bonded_atom_name_pairs[restype]:
            idx1 = restype_to_allatom_names[restype].index(atom1_name)            
            idx2 = restype_to_allatom_names[restype].index(atom2_name)
            restype_to_allatom_bond_matrix[restype][idx1, idx2] = bond_type
            restype_to_allatom_bond_matrix[restype][idx2, idx1] = bond_type
            if atom1_name in restype_to_heavyatom_names[restype] and \
                    atom2_name in restype_to_heavyatom_names[restype]:
                jdx1 = restype_to_heavyatom_names[restype].index(atom1_name)
                jdx2 = restype_to_heavyatom_names[restype].index(atom2_name)
                restype_to_heavyatom_bond_matrix[restype][jdx1, jdx2] = bond_type
                restype_to_heavyatom_bond_matrix[restype][jdx2, jdx1] = bond_type


_make_bond_matrices()


##
# Torsion geometry and ideal coordinates

class Torsion(enum.IntEnum):
    Backbone = 0
    Omega = 1
    Phi = 2
    Psi = 3
    Chi1 = 4
    Chi2 = 5
    Chi3 = 6
    Chi7 = 7


chi_angles_atoms = {
    AA.ALA: [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
    AA.ARG: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
             ['CB', 'CG', 'CD', 'NE'], ['CG', 'CD', 'NE', 'CZ']],
    AA.ASN: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    AA.ASP: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    AA.CYS: [['N', 'CA', 'CB', 'SG']],
    AA.GLN: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
             ['CB', 'CG', 'CD', 'OE1']],
    AA.GLU: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
             ['CB', 'CG', 'CD', 'OE1']],
    AA.GLY: [],
    AA.HIS: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']],
    AA.ILE: [['N', 'CA', 'CB', 'CG1'], ['CA', 'CB', 'CG1', 'CD1']],
    AA.LEU: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    AA.LYS: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
             ['CB', 'CG', 'CD', 'CE'], ['CG', 'CD', 'CE', 'NZ']],
    AA.MET: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'SD'],
             ['CB', 'CG', 'SD', 'CE']],
    AA.PHE: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    AA.PRO: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD']],
    AA.SER: [['N', 'CA', 'CB', 'OG']],
    AA.SEP: [['N', 'CA', 'CB', 'OG'], ['CA', 'CB', 'OG', 'P'],
             ['CB', 'OG', 'P', 'O1P']],
    AA.THR: [['N', 'CA', 'CB', 'OG1']],
    AA.TRP: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    AA.TYR: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    AA.TYS: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1'],
             ['CE1', 'CZ', 'OH', 'S'], ['CZ', 'OH', 'S', 'O1']],
    AA.PTR: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1'],
             ['CE1', 'CZ', 'OH', 'P'], ['CZ', 'OH', 'P', 'O1P']],
    AA.VAL: [['N', 'CA', 'CB', 'CG1']],
    AA.MLE: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1'],
             ['C', 'CA', 'N', 'CN']],
    AA.M3L: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
             ['CB', 'CG', 'CD', 'CE'], ['CG', 'CD', 'CE', 'NZ'],
             ['CD', 'CE', 'NZ', 'CM1']],
    AA.ALY: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
             ['CB', 'CG', 'CD', 'CE'], ['CG', 'CD', 'CE', 'NZ'],
             ['CD', 'CE', 'NZ', 'CH'], ['CE', 'NZ', 'CH', 'CH3']], 
    AA.TPO: [['N', 'CA', 'CB', 'OG1'], ['CA', 'CB', 'OG1', 'P'],
             ['CB', 'OG1', 'P', 'O1P']],
    AA.DAL: [],
    AA.HYP: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD']],
    AA.MVA: [['N', 'CA', 'CB', 'CG1'], ['C', 'CA', 'N', 'CN']],
    AA.DLY: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
             ['CB', 'CG', 'CD', 'CE'], ['CG', 'CD', 'CE', 'NZ']],
    AA.DLE: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    AA.SAR: [['C', 'CA', 'N', 'CN']],
    AA.NLE: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
             ['CB', 'CG', 'CD', 'CE']],
    AA.BMT: [['N', 'CA', 'CB', 'OG1'], ['CA', 'CB', 'CG2', 'CD2'],
             ['CB', 'CG2', 'CD2', 'CE'], ['CG2', 'CD2', 'CE', 'CZ'],
             ['C', 'CA', 'N', 'CN']],
    AA.DGL: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
             ['CB', 'CG', 'CD', 'OE1']],
    AA.DPR: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD']],
    AA.DTR: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],                                                    

}

chi_angles_mask = {
    AA.ALA: [False, False, False, False, False, False],  # ALA
    AA.ARG: [True, True, True, True, False, False],  # ARG
    AA.ASN: [True, True, False, False, False, False],  # ASN
    AA.ASP: [True, True, False, False, False, False],  # ASP
    AA.CYS: [True, False, False, False, False, False],  # CYS
    AA.GLN: [True, True, True, False, False, False],  # GLN
    AA.GLU: [True, True, True, False, False, False],  # GLU
    AA.GLY: [False, False, False, False, False, False],  # GLY
    AA.HIS: [True, True, False, False, False, False],  # HIS
    AA.ILE: [True, True, False, False, False, False],  # ILE
    AA.LEU: [True, True, False, False, False, False],  # LEU
    AA.LYS: [True, True, True, True, False, False],  # LYS
    AA.MET: [True, True, True, False, False, False],  # MET
    AA.PHE: [True, True, False, False, False, False],  # PHE
    AA.PRO: [True, True, False, False, False, False],  # PRO
    AA.SER: [True, False, False, False, False, False],  # SER
    AA.SEP: [True, True, True, False, False, False],  # SEP
    AA.THR: [True, False, False, False, False, False],  # THR
    AA.TRP: [True, True, False, False, False, False],  # TRP
    AA.TYR: [True, True, False, False, False, False],  # TYR
    AA.TYS: [True, True, True, True, False, False],  # TYS
    AA.PTR: [True, True, True, True, False, False],  # PTR
    AA.VAL: [True, False, False, False, False, False],  # VAL
    AA.MLE: [True, True, True, False, False, False],  # MLE, 23
    AA.M3L: [True, True, True, True, True, False],
    AA.ALY: [True, True, True, True, True, True],
    AA.TPO: [True, True, True, False, False, False],
    AA.DAL: [False, False, False, False, False, False],
    AA.HYP: [True, True, False, False, False, False],
    AA.MVA: [True, True, False, False, False, False],  # MVA, 29
    AA.DLY: [True, True, True, True, False, False],
    AA.DLE: [True, True, False, False, False, False],
    AA.SAR: [True, False, False, False, False, False],
    AA.NLE: [True, True, True, False, False, False],
    AA.BMT: [True, True, True, True, True, False],
    AA.DGL: [True, True, True, False, False, False],
    AA.DPR: [True, True, False, False, False, False],
    AA.DTR: [True, True, False, False, False, False],
    AA.UNK: [False, False, False, False, False, False],  # UNK
}

chi_pi_periodic = {
    AA.ALA: [False, False, False, False, False, False],  # ALA
    AA.ARG: [False, False, False, False, False, False],  # ARG
    AA.ASN: [False, False, False, False, False, False],  # ASN
    AA.ASP: [False, True, False, False, False, False],  # ASP
    AA.CYS: [False, False, False, False, False, False],  # CYS
    AA.GLN: [False, False, False, False, False, False],  # GLN
    AA.GLU: [False, False, True, False, False, False],  # GLU
    AA.GLY: [False, False, False, False, False, False],  # GLY
    AA.HIS: [False, False, False, False, False, False],  # HIS
    AA.ILE: [False, False, False, False, False, False],  # ILE
    AA.LEU: [False, False, False, False, False, False],  # LEU
    AA.LYS: [False, False, False, False, False, False],  # LYS
    AA.MET: [False, False, False, False, False, False],  # MET
    AA.PHE: [False, True, False, False, False, False],  # PHE
    AA.PRO: [False, False, False, False, False, False],  # PRO
    AA.SER: [False, False, False, False, False, False],  # SER
    AA.SEP: [False, False, False, False, False, False],  # SEP
    AA.THR: [False, False, False, False, False, False],  # THR
    AA.TRP: [False, False, False, False, False, False],  # TRP
    AA.TYR: [False, True, False, False, False, False],  # TYR
    AA.TYS: [False, True, False, False, False, False],  # TYS
    AA.PTR: [False, True, False, False, False, False],  # PTR
    AA.VAL: [False, False, False, False, False, False],  # VAL
    AA.MLE: [False, False, False, False, False, False],
    AA.M3L: [False, False, False, False, False, False],
    AA.ALY: [False, False, False, False, False, False],
    AA.TPO: [False, False, False, False, False, False],
    AA.DAL: [False, False, False, False, False, False],
    AA.HYP: [False, False, False, False, False, False],
    AA.MVA: [False, False, False, False, False, False],
    AA.DLY: [False, False, False, False, False, False],
    AA.DLE: [False, False, False, False, False, False],
    AA.SAR: [False, False, False, False, False, False],
    AA.NLE: [False, False, False, False, False, False],
    AA.BMT: [False, False, False, False, False, False],
    AA.DGL: [False, False, True, False, False, False],
    AA.DPR: [False, False, False, False, False, False],
    AA.DTR: [False, False, False, False, False, False],
    AA.UNK: [False, False, False, False, False, False],  # UNK
}

# 10: rigid group of CN
rigid_group_heavy_atom_positions = {
    AA.ALA: [
        ['N', 0, (-0.525, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, -0.000, -0.000)],
        ['CB', 0, (-0.529, -0.774, -1.205)],
        ['O', 3, (0.627, -1.062, 0.000)],
    ],
    AA.ARG: [
        ['N', 0, (-0.524, 1.362, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, -0.000)],
        ['CB', 0, (-0.524, -0.778, -1.209)],
        ['O', 3, (0.626, -1.062, 0.000)],
        ['CG', 4, (0.616, 1.390, -0.000)],
        ['CD', 5, (0.564, 1.414, 0.000)],
        ['NE', 6, (0.539, 1.357, -0.000)],
        ['NH1', 7, (0.206, 2.301, 0.000)],
        ['NH2', 7, (2.078, 0.978, -0.000)],
        ['CZ', 7, (0.758, 1.093, -0.000)],
    ],
    AA.ASN: [
        ['N', 0, (-0.536, 1.357, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, -0.000, -0.000)],
        ['CB', 0, (-0.531, -0.787, -1.200)],
        ['O', 3, (0.625, -1.062, 0.000)],
        ['CG', 4, (0.584, 1.399, 0.000)],
        ['ND2', 5, (0.593, -1.188, 0.001)],
        ['OD1', 5, (0.633, 1.059, 0.000)],
    ],
    AA.ASP: [
        ['N', 0, (-0.525, 1.362, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, 0.000, -0.000)],
        ['CB', 0, (-0.526, -0.778, -1.208)],
        ['O', 3, (0.626, -1.062, -0.000)],
        ['CG', 4, (0.593, 1.398, -0.000)],
        ['OD1', 5, (0.610, 1.091, 0.000)],
        ['OD2', 5, (0.592, -1.101, -0.003)],
    ],
    AA.CYS: [
        ['N', 0, (-0.522, 1.362, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.524, 0.000, 0.000)],
        ['CB', 0, (-0.519, -0.773, -1.212)],
        ['O', 3, (0.625, -1.062, -0.000)],
        ['SG', 4, (0.728, 1.653, 0.000)],
    ],
    AA.GLN: [
        ['N', 0, (-0.526, 1.361, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, 0.000, 0.000)],
        ['CB', 0, (-0.525, -0.779, -1.207)],
        ['O', 3, (0.626, -1.062, -0.000)],
        ['CG', 4, (0.615, 1.393, 0.000)],
        ['CD', 5, (0.587, 1.399, -0.000)],
        ['NE2', 6, (0.593, -1.189, -0.001)],
        ['OE1', 6, (0.634, 1.060, 0.000)],
    ],
    AA.GLU: [
        ['N', 0, (-0.528, 1.361, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, -0.000, -0.000)],
        ['CB', 0, (-0.526, -0.781, -1.207)],
        ['O', 3, (0.626, -1.062, 0.000)],
        ['CG', 4, (0.615, 1.392, 0.000)],
        ['CD', 5, (0.600, 1.397, 0.000)],
        ['OE1', 6, (0.607, 1.095, -0.000)],
        ['OE2', 6, (0.589, -1.104, -0.001)],
    ],
    AA.GLY: [
        ['N', 0, (-0.572, 1.337, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.517, -0.000, -0.000)],
        ['O', 3, (0.626, -1.062, -0.000)],
    ],
    AA.HIS: [
        ['N', 0, (-0.527, 1.360, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, 0.000, 0.000)],
        ['CB', 0, (-0.525, -0.778, -1.208)],
        ['O', 3, (0.625, -1.063, 0.000)],
        ['CG', 4, (0.600, 1.370, -0.000)],
        ['CD2', 5, (0.889, -1.021, 0.003)],
        ['ND1', 5, (0.744, 1.160, -0.000)],
        ['CE1', 5, (2.030, 0.851, 0.002)],
        ['NE2', 5, (2.145, -0.466, 0.004)],
    ],
    AA.ILE: [
        ['N', 0, (-0.493, 1.373, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, -0.000, -0.000)],
        ['CB', 0, (-0.536, -0.793, -1.213)],
        ['O', 3, (0.627, -1.062, -0.000)],
        ['CG1', 4, (0.534, 1.437, -0.000)],
        ['CG2', 4, (0.540, -0.785, -1.199)],
        ['CD1', 5, (0.619, 1.391, 0.000)],
    ],
    AA.LEU: [
        ['N', 0, (-0.520, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, -0.000)],
        ['CB', 0, (-0.522, -0.773, -1.214)],
        ['O', 3, (0.625, -1.063, -0.000)],
        ['CG', 4, (0.678, 1.371, 0.000)],
        ['CD1', 5, (0.530, 1.430, -0.000)],
        ['CD2', 5, (0.535, -0.774, 1.200)],
    ],
    AA.LYS: [
        ['N', 0, (-0.526, 1.362, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, 0.000, 0.000)],
        ['CB', 0, (-0.524, -0.778, -1.208)],
        ['O', 3, (0.626, -1.062, -0.000)],
        ['CG', 4, (0.619, 1.390, 0.000)],
        ['CD', 5, (0.559, 1.417, 0.000)],
        ['CE', 6, (0.560, 1.416, 0.000)],
        ['NZ', 7, (0.554, 1.387, 0.000)],
    ],
    AA.MET: [
        ['N', 0, (-0.521, 1.364, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, 0.000, 0.000)],
        ['CB', 0, (-0.523, -0.776, -1.210)],
        ['O', 3, (0.625, -1.062, -0.000)],
        ['CG', 4, (0.613, 1.391, -0.000)],
        ['SD', 5, (0.703, 1.695, 0.000)],
        ['CE', 6, (0.320, 1.786, -0.000)],
    ],
    AA.PHE: [
        ['N', 0, (-0.518, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.524, 0.000, -0.000)],
        ['CB', 0, (-0.525, -0.776, -1.212)],
        ['O', 3, (0.626, -1.062, -0.000)],
        ['CG', 4, (0.607, 1.377, 0.000)],
        ['CD1', 5, (0.709, 1.195, -0.000)],
        ['CD2', 5, (0.706, -1.196, 0.000)],
        ['CE1', 5, (2.102, 1.198, -0.000)],
        ['CE2', 5, (2.098, -1.201, -0.000)],
        ['CZ', 5, (2.794, -0.003, -0.001)],
    ],
    AA.PRO: [
        ['N', 0, (-0.566, 1.351, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, -0.000, 0.000)],
        ['CB', 0, (-0.546, -0.611, -1.293)],
        ['O', 3, (0.621, -1.066, 0.000)],
        ['CG', 4, (0.382, 1.445, 0.0)],
        ['CD', 5, (0.477, 1.424, 0.0)],  # manually made angle 2 degrees larger
    ],
    AA.SER: [
        ['N', 0, (-0.529, 1.360, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, -0.000)],
        ['CB', 0, (-0.518, -0.777, -1.211)],
        ['O', 3, (0.626, -1.062, -0.000)],
        ['OG', 4, (0.503, 1.325, 0.000)],
    ],
    AA.SEP: [
        ["N", 0, (-0.445, 1.389, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.521, -0.000, -0.000)],
        ["CB", 0, (-0.621, -0.568, -1.251)],
        ["O", 3, (0.608, -1.066, -0.000)],
        ["OG", 4, (0.462, 1.355, 0.000)],  # 1.42~1.44
        ["P", 5, (0.677, 1.476, 0.000)],  # 1.58~1.62
        ["O1P", 6, (0.611, 1.347, 0.000)],  # 1.45~1.50
        ["O2P", 6, (0.400, -0.792, 1.292)],  # 1.58~1.62
        ["O3P", 6, (0.267, -0.878, -1.195)],  # 1.58~1.62
    ],
    AA.THR: [
        ['N', 0, (-0.517, 1.364, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, 0.000, -0.000)],
        ['CB', 0, (-0.516, -0.793, -1.215)],
        ['O', 3, (0.626, -1.062, 0.000)],
        ['CG2', 4, (0.550, -0.718, -1.228)],
        ['OG1', 4, (0.472, 1.353, 0.000)],
    ],
    AA.TRP: [
        ['N', 0, (-0.521, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, 0.000)],
        ['CB', 0, (-0.523, -0.776, -1.212)],
        ['O', 3, (0.627, -1.062, 0.000)],
        ['CG', 4, (0.609, 1.370, -0.000)],
        ['CD1', 5, (0.824, 1.091, 0.000)],
        ['CD2', 5, (0.854, -1.148, -0.005)],
        ['CE2', 5, (2.186, -0.678, -0.007)],
        ['CE3', 5, (0.622, -2.530, -0.007)],
        ['NE1', 5, (2.140, 0.690, -0.004)],
        ['CH2', 5, (3.028, -2.890, -0.013)],
        ['CZ2', 5, (3.283, -1.543, -0.011)],
        ['CZ3', 5, (1.715, -3.389, -0.011)],
    ],
    AA.TYR: [
        ['N', 0, (-0.522, 1.362, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.524, -0.000, -0.000)],
        ['CB', 0, (-0.522, -0.776, -1.213)],
        ['O', 3, (0.627, -1.062, -0.000)],
        ['CG', 4, (0.607, 1.382, -0.000)],
        ['CD1', 5, (0.716, 1.195, -0.000)],
        ['CD2', 5, (0.713, -1.194, -0.001)],
        ['CE1', 5, (2.107, 1.200, -0.002)],
        ['CE2', 5, (2.104, -1.201, -0.003)],
        ['OH', 5, (4.168, -0.002, -0.005)],
        ['CZ', 5, (2.791, -0.001, -0.003)],
    ],
    AA.TYS: [
        ["N", 0, (-0.648, 1.313, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.520, -0.000, -0.000)],
        ["CB", 0, (-0.494, -0.798, -1.215)],
        ["O", 3, (0.659, 1.040, -0.000)],
        ["CG", 4, (0.609, 1.396, -0.000)],
        ["CD1", 5, (0.689, 1.206, -0.000)],
        ["CD2", 5, (0.672, -1.206, -0.022)],
        ["CE1", 5, (2.062, 1.214, -0.007)],
        ["CE2", 5, (2.049, -1.204, -0.022)],
        ["OH", 5, (3.958, -0.000, -0.019)],
        ["CZ", 5, (2.724, -0.003, -0.013)],
        ["S", 6, (0.725, -0.015, 1.256)],
        ["O1", 7, (0.506, 1.346, 0.004)],
        ["O2", 7, (0.453, -0.612, 1.215)],
        ["O3", 7, (0.556, -0.727, -1.103)],
    ],
    AA.PTR: [
        ["N", 0, (-0.462, 1.404, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.521, -0.000, -0.000)],
        ["CB", 0, (-0.560, -0.735, -1.222)],
        ["O", 3, (0.597, 1.097, -0.000)],
        ["CG", 4, (0.659, 1.367, -0.000)],
        ["CD1", 5, (0.717, 1.200, -0.000)],
        ["CD2", 5, (0.718, -1.203, -0.088)],
        ["CE1", 5, (2.122, 1.210, -0.099)],
        ["CE2", 5, (2.123, -1.218, -0.189)],
        ["OH", 5, (4.246, 0.026, -0.344)],
        ["CZ", 5, (2.816, -0.009, -0.197)],
        ["P", 6, (0.873, 0.044, 1.239)],
        ["O1P", 7, (0.576, 1.331, -0.041)],
        ["O2P", 7, (0.493, -0.814, 1.122)],
        ["O3P", 7, (0.209, -0.705, -1.257)],
    ],
    AA.VAL: [
        ['N', 0, (-0.494, 1.373, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, -0.000, -0.000)],
        ['CB', 0, (-0.533, -0.795, -1.213)],
        ['O', 3, (0.627, -1.062, -0.000)],
        ['CG1', 4, (0.540, 1.429, -0.000)],
        ['CG2', 4, (0.533, -0.776, 1.203)],
    ],
    AA.MLE: [
        ['N', 0, (-0.534, 1.370, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, -0.000)],
        ['CB', 0, (-0.537, -0.597, -1.302)],
        ['O', 3, (0.625, -1.063, -0.000)],
        ['CG', 4, (0.645, 1.388, 0.000)],
        ['CD1', 5, (0.510, 1.422, -0.000)],
        ['CD2', 5, (0.524, -0.747, 1.221)],
        ['CN', 6, (0.760, -1.291, 0.000)]  # Chi3
    ],  # 2oju
    AA.M3L: [
        ['N', 0, (-0.549, 1.348, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.534, 0.000, 0.000)],
        ['CB', 0, (-0.518, -0.747, -1.234)],
        ['O', 3, (0.638, -1.060, -0.000)],
        ['CG', 4, (0.640, 1.370, 0.000)],
        ['CD', 5, (0.544, 1.419, 0.000)],
        ['CE', 6, (0.605, 1.412, 0.000)],
        ['NZ', 7, (0.489, 1.394, 0.002)],
        ['CM1', 8, (0.497, 1.415, -0.001)],
        ['CM2', 8, (0.449, -0.722, -1.210)],
        ['CM3', 8, (0.689, -0.680, 1.119)],
    ],  # 3mp1
    AA.ALY: [
        ['N', 0, (-0.389, 1.401, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.534, 0.000, 0.000)],
        ['CB', 0, (-0.528, -0.564, -1.326)],
        ['O', 3, (0.520, -1.103, -0.000)],
        ['CG', 4, (0.569, 1.384, 0.000)],
        ['CD', 5, (0.423, 1.481, 0.000)],
        ['CE', 6, (0.564, 1.423, 0.000)],
        ['NZ', 7, (0.516, 1.368, 0.001)],
        ['CH', 8, (0.706, 1.121, 0.000)],
        ['OH', 9, (0.604, -1.030, 0.048)],
        ['CH3', 9, (0.738, 1.305, 0.000)],
    ],  # 4fz3
    AA.TPO: [
        ['N', 0, (-0.574, 1.352, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.529, 0.000, -0.000)],
        ['CB', 0, (-0.678, -0.626, -1.236)],
        ['O', 3, (0.628, -1.060, 0.000)],
        ['CG2', 4, (0.673, -0.705, -1.194)],
        ['OG1', 4, (0.329, 1.419, 0.000)],
        ['P', 5, (0.722, 1.378, 0.000)],
        ['O1P', 6, (0.427, 1.484, -0.001)],
        ['O2P', 6, (0.461, -0.694, -1.292)],
        ['O3P', 6, (0.610, -0.703, 1.197)],
    ],  # 6gy2
    AA.DAL: [
        ['N', 0, (-0.412, 1.394, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.513, -0.000, -0.000)],
        ['CB', 0, (-0.553, -0.728, 1.213)],
        ['O', 3, (0.608, -1.069, 0.000)],
    ],  # 2oju
    AA.HYP: [
        ['N', 0, (-0.505, 1.377, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.507, -0.000, 0.000)],
        ['CB', 0, (-0.490, -0.671, -1.282)],
        ['O', 3, (0.584, -1.059, 0.000)],
        ['CG', 4, (0.375, 1.476, 0.0)],
        ['CD', 5, (0.697, 1.462, 0.0)],
        ['OD1', 5, (0.489, -0.638, -1.154)],
    ],  # 7tpt_1_III
    AA.MVA: [
        ['N', 0, (-0.439, 1.400, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.515, -0.000, -0.000)],
        ['CB', 0, (-0.521, -0.848, -1.177)],
        ['O', 3, (0.631, -1.059, -0.000)],
        ['CG1', 4, (0.532, 1.422, -0.000)],
        ['CG2', 4, (0.519, -0.743, -1.224)],  # 1.224?
        ['CN', 5, (0.737, -1.301, 0.000)],  # Chi2
    ],  # 2oju
    AA.DLY: [
        ['N', 0, (-0.695, 1.262, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.519, 0.000, 0.000)],
        ['CB', 0, (-0.468, -0.714, 1.279)],
        ['O', 3, (0.565, -1.096, -0.000)],
        ['CG', 4, (0.732, 1.342, 0.000)],
        ['CD', 5, (0.603, 1.401, 0.000)],
        ['CE', 6, (0.555, 1.423, 0.000)],
        ['NZ', 7, (0.593, 1.370, 0.000)],
    ],  # 5ney
    AA.DLE: [
        ['N', 0, (-0.524, 1.358, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, -0.000, -0.000)],
        ['CB', 0, (-0.514, -0.774, 1.217)],
        ['O', 3, (0.621, -1.059, -0.000)],
        ['CG', 4, (0.671, 1.371, 0.001)],
        ['CD1', 5, (0.534, 1.420, -0.000)],
        ['CD2', 5, (0.541, -0.794, 1.182)],
    ],  # 6q8g
    AA.SAR: [
        ['N', 0, (-0.544, 1.353, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, -0.000, -0.000)],
        ['O', 3, (0.627, -1.064, -0.000)],
        ['CN', 4, (0.656, 1.348, 0.000)],  # Chi1
    ],  # 2oju
    AA.NLE: [
        ['N', 0, (-0.710, 1.279, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.547, -0.000, -0.000)],
        ['CB', 0, (-0.599, -0.795, -1.177)],
        ['O', 3, (0.655, -1.071, -0.000)],
        ['CG', 4, (0.584, 1.426, -0.000)],
        ['CD', 5, (0.585, 1.419, 0.000)],
        ['CE', 6, (0.664, 1.382, -0.002)],
    ],  # 4y1d
    AA.BMT: [
        ['N', 0, (-0.561, 1.354, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.515, 0.000, -0.000)],
        ['CB', 0, (-0.558, -0.872, -1.197)],
        ['O', 3, (0.623, -1.063, 0.000)],
        ['CG2', 4, (0.629, -0.774, -1.223)],
        ['OG1', 4, (0.473, 1.369, 0.000)],
        ['CD1', 5, (0.595, -0.861, 1.148)],
        ['CD2', 5, (0.684, 1.385, 0.000)],
        ['CE', 6, (0.698, 1.365, 0.000)],
        ['CZ', 7, (0.734, 1.217, 0.000)],
        ['CH', 7, (2.257, 1.192, -0.002)],
        ['CN', 8, (0.777, -1.271, 0.000)],  # Chi5
    ],  # 2oju                
    AA.DGL: [
        ['N', 0, (-0.611, 1.330, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, -0.000)],
        ['CB', 0, (-0.568, -0.913, 1.104)],
        ['O', 3, (0.587, -1.103, 0.000)],
        ['CG', 4, (0.608, 1.413, 0.000)],
        ['CD', 5, (0.590, 1.379, 0.000)],
        ['OE1', 6, (0.668, 1.018, -0.000)],
        ['OE2', 6, (0.509, -1.208, -0.006)],
    ],  # 4thn
    AA.DPR: [
        ['N', 0, (-0.542, 1.375, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.544, -0.000, 0.000)],
        ['CB', 0, (-0.570, -0.622, 1.289)],
        ['O', 3, (0.602, -1.080, 0.000)],
        ['CG', 4, (0.425, 1.450, 0.000)],
        ['CD', 5, (0.427, 1.433, 0.000)],
    ],  # 6qfk
    AA.DTR: [
        ['N', 0, (-0.603, 1.305, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.528, -0.000, 0.000)],
        ['CB', 0, (-0.372, -0.625, 1.353)],
        ['O', 3, (0.662, -1.005, 0.000)],
        ['CG', 4, (0.563, 1.459, -0.000)],
        ['CD1', 5, (0.779, 1.109, 0.001)],
        ['CD2', 5, (0.819, -1.313, -0.006)],
        ['CE2', 5, (2.216, -0.758, -0.005)],
        ['CE3', 5, (0.654, -2.701, -0.008)],
        ['NE1', 5, (1.986, 0.626, -0.006)],
        ['CH2', 5, (3.104, -2.997, -0.022)],
        ['CZ2', 5, (3.314, -1.621, -0.017)],
        ['CZ3', 5, (1.826, -3.521, -0.011)],
    ],  # 5ney
}

# The following tensors are initialized by `_make_rigid_group_constants`
num_rigid_groups = 10  # 8
restype_rigid_group_rotation = torch.zeros([PAD_RESIDUE_INDEX, num_rigid_groups, 3, 3])
restype_rigid_group_translation = torch.zeros([PAD_RESIDUE_INDEX, num_rigid_groups, 3])
restype_heavyatom_to_rigid_group = torch.zeros([PAD_RESIDUE_INDEX, max_num_heavyatoms-1], dtype=torch.long)
restype_heavyatom_rigid_group_positions = torch.zeros([PAD_RESIDUE_INDEX, max_num_heavyatoms-1, 3])


def _make_rigid_group_constants():
    def _make_rotation_matrix(ex, ey):
        # Gram-Schmidt orthogonalization
        ex_normalized = ex / torch.linalg.norm(ex)

        # make ey perpendicular to ex
        ey_normalized = ey - torch.dot(ey, ex_normalized) * ex_normalized
        ey_normalized /= torch.linalg.norm(ey_normalized)

        eznorm = torch.cross(ex_normalized, ey_normalized)
        m = torch.stack([ex_normalized, ey_normalized, eznorm]).transpose(0, 1)  # (3, 3_index)
        return m

    for restype in AA:
        if restype == AA.UNK: continue

        atom_groups = {
            name: group
            for name, group, _ in rigid_group_heavy_atom_positions[restype]
        }
        atom_positions = {
            name: torch.FloatTensor(pos)
            for name, _, pos in rigid_group_heavy_atom_positions[restype]
        }

        # Atom 14 rigid group positions
        for atom_idx, atom_name in enumerate(restype_to_heavyatom_names[restype]):
            if (atom_name == '') or (atom_name not in atom_groups): continue
            restype_heavyatom_to_rigid_group[restype, atom_idx] = atom_groups[atom_name]
            restype_heavyatom_rigid_group_positions[restype, atom_idx, :] = atom_positions[atom_name]

        # 0: backbone to backbone
        restype_rigid_group_rotation[restype, Torsion.Backbone, :, :] = torch.eye(3)
        restype_rigid_group_translation[restype, Torsion.Backbone, :] = torch.zeros([3])

        # 1: omega-frame to backbone
        restype_rigid_group_rotation[restype, Torsion.Omega, :, :] = torch.eye(3)
        restype_rigid_group_translation[restype, Torsion.Omega, :] = torch.zeros([3])

        # 2: phi-frame to backbone
        restype_rigid_group_rotation[restype, Torsion.Phi, :, :] = _make_rotation_matrix(
            ex=atom_positions['N'] - atom_positions['CA'],
            ey=torch.FloatTensor([1., 0., 0.]),
        )
        restype_rigid_group_translation[restype, Torsion.Phi, :] = atom_positions['N']

        # 3: psi-frame to backbone
        restype_rigid_group_rotation[restype, Torsion.Psi, :, :] = _make_rotation_matrix(
            ex=atom_positions['C'] - atom_positions['CA'],
            ey=atom_positions['CA'] - atom_positions['N'],  # In accordance to the definition of psi angle
        )
        restype_rigid_group_translation[restype, Torsion.Psi, :] = atom_positions['C']

        # 4: chi1-frame to backbone
        if chi_angles_mask[restype][0]:
            base_atom_names = chi_angles_atoms[restype][0]
            base_atom_positions = [atom_positions[name] for name in base_atom_names]
            restype_rigid_group_rotation[restype, Torsion.Chi1, :, :] = _make_rotation_matrix(
                ex=base_atom_positions[2] - base_atom_positions[1],
                ey=base_atom_positions[0] - base_atom_positions[1],
            )
            restype_rigid_group_translation[restype, Torsion.Chi1, :] = base_atom_positions[2]

        # chi2-chi1
        # chi3-chi2
        # chi4-chi3
        for chi_idx in range(1, 6):
            if chi_angles_mask[restype][chi_idx]:
                axis_end_atom_name = chi_angles_atoms[restype][chi_idx][2]
                axis_end_atom_position = atom_positions[axis_end_atom_name]
                restype_rigid_group_rotation[restype, Torsion.Chi1 + chi_idx, :, :] = _make_rotation_matrix(
                    ex=axis_end_atom_position,
                    ey=torch.FloatTensor([-1., 0., 0.]),
                )
                restype_rigid_group_translation[restype, Torsion.Chi1 + chi_idx, :] = axis_end_atom_position


_make_rigid_group_constants()

"""
# The following tensors are taken from diffab
"""
backbone_atom_coordinates = {
    AA.ALA: [
        (-0.525, 1.363, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.526, -0.0, -0.0),  # C
    ],
    AA.ARG: [
        (-0.524, 1.362, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.525, -0.0, -0.0),  # C
    ],
    AA.ASN: [
        (-0.536, 1.357, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.526, -0.0, -0.0),  # C
    ],
    AA.ASP: [
        (-0.525, 1.362, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.527, 0.0, -0.0),  # C
    ],
    AA.CYS: [
        (-0.522, 1.362, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.524, 0.0, 0.0),  # C
    ],
    AA.GLN: [
        (-0.526, 1.361, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.526, 0.0, 0.0),  # C
    ],
    AA.GLU: [
        (-0.528, 1.361, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.526, -0.0, -0.0),  # C
    ],
    AA.GLY: [
        (-0.572, 1.337, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.517, -0.0, -0.0),  # C
    ],
    AA.HIS: [
        (-0.527, 1.36, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.525, 0.0, 0.0),  # C
    ],
    AA.ILE: [
        (-0.493, 1.373, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.527, -0.0, -0.0),  # C
    ],
    AA.LEU: [
        (-0.52, 1.363, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.525, -0.0, -0.0),  # C
    ],
    AA.LYS: [
        (-0.526, 1.362, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.526, 0.0, 0.0),  # C
    ],
    AA.MET: [
        (-0.521, 1.364, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.525, 0.0, 0.0),  # C
    ],
    AA.PHE: [
        (-0.518, 1.363, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.524, 0.0, -0.0),  # C
    ],
    AA.PRO: [
        (-0.566, 1.351, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.527, -0.0, 0.0),  # C
    ],
    AA.SER: [
        (-0.529, 1.36, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.525, -0.0, -0.0),  # C
    ],
    AA.THR: [
        (-0.517, 1.364, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.526, 0.0, -0.0),  # C
    ],
    AA.TRP: [
        (-0.521, 1.363, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.525, -0.0, 0.0),  # C
    ],
    AA.TYR: [
        (-0.522, 1.362, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.524, -0.0, -0.0),  # C
    ],
    AA.VAL: [
        (-0.494, 1.373, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.527, -0.0, -0.0),  # C
    ],
}

bb_oxygen_coordinate = {
    AA.ALA: (2.153, -1.062, 0.0),
    AA.ARG: (2.151, -1.062, 0.0),
    AA.ASN: (2.151, -1.062, 0.0),
    AA.ASP: (2.153, -1.062, 0.0),
    AA.CYS: (2.149, -1.062, 0.0),
    AA.GLN: (2.152, -1.062, 0.0),
    AA.GLU: (2.152, -1.062, 0.0),
    AA.GLY: (2.143, -1.062, 0.0),
    AA.HIS: (2.15, -1.063, 0.0),
    AA.ILE: (2.154, -1.062, 0.0),
    AA.LEU: (2.15, -1.063, 0.0),
    AA.LYS: (2.152, -1.062, 0.0),
    AA.MET: (2.15, -1.062, 0.0),
    AA.PHE: (2.15, -1.062, 0.0),
    AA.PRO: (2.148, -1.066, 0.0),
    AA.SER: (2.151, -1.062, 0.0),
    AA.THR: (2.152, -1.062, 0.0),
    AA.TRP: (2.152, -1.062, 0.0),
    AA.TYR: (2.151, -1.062, 0.0),
    AA.VAL: (2.154, -1.062, 0.0),
}

backbone_atom_coordinates_tensor = torch.zeros([21, 3, 3])
bb_oxygen_coordinate_tensor = torch.zeros([21, 3])


def make_coordinate_tensors():
    for restype, atom_coords in backbone_atom_coordinates.items():
        for atom_id, atom_coord in enumerate(atom_coords):
            backbone_atom_coordinates_tensor[restype][atom_id] = torch.FloatTensor(atom_coord)

    for restype, bb_oxy_coord in bb_oxygen_coordinate.items():
        bb_oxygen_coordinate_tensor[restype] = torch.FloatTensor(bb_oxy_coord)


make_coordinate_tensors()

"""
# Convert atom14 to atom37 for clash evaluation in AttnPacker
"""

from eval.attnpacker.protein_constants import ALL_ATOMS

atom37_indices = {atom: idx for idx, atom in enumerate(ALL_ATOMS)}

restype_to_atom37_map = {}


# Function to map atom14 to atom37
def make_atom14_to_atom37_map():
    for restype, atom14_names in restype_to_heavyatom_names.items():
        # Map each atom14 to its index in atom37, or -1 if it doesn't exist
        atom37_map = [
            atom37_indices[atom] if atom in atom37_indices else -1
            for atom in atom14_names
        ]
        restype_to_atom37_map[int(restype)] = atom37_map


make_atom14_to_atom37_map()
