"""
A utility module that allows access to the amino acid attributes needed for descriptors and encodings.
The list of supported amino acids and post-translations are at the bottom of this page
and the data for all attributes are under the [peptidy.data](https://github.com/molML/peptidy/tree/main/peptidy/data).

Attributes
----------
aromatic_aas: list
    A list of aromatic amino acids retrieve based on [Lobry,1994](https://academic.oup.com/nar/article-abstract/22/15/3174/1087817).
blosum62_scores: dict
    A dictionary that contains the BLOSUM62 matrix.
descriptor_per_aas: dict
    A dictionary that contains the all descriptor values per amino acid.
formulas: dict
    A dictionary that contains the closed formulas of the amino acids.
hydrophobic_aas: list
    A list of hydrophobic amino acids ([Nelson & Cox, 2004](https://mis.kp.ac.rw/admin/admin_panel/kp_lms/files/digital/Core%20Books/Core%20Books%20In%20Nursing%20%20And%20%20Midwifery/H106_%20Biochemistry_Lehninger%20Principles%20of%20Biochemistry,%20Fourth%20Edition%20-%20David%20L.%20Nelson,%20Michael%20M.%20Cox.pdf)).
instabilities: dict
    A dictionary that contains the instability of amino acids pairs per [Guruprasad, Reddy & Pandit, 1990](https://academic.oup.com/peds/article-abstract/4/2/155/1491271).
n_h_acceptors: dict
    A dictionary that contains the number of hydrogen acceptors per amino acid according to [PubChem](https://pubchem.ncbi.nlm.nih.gov/).
n_h_donors: dict
    A dictionary that contains the number of hydrogen donors per amino acid according to [PubChem](https://pubchem.ncbi.nlm.nih.gov/).
n_rotatable_bonds: dict
    A dictionary that contains the number of rotatable bonds in each amino acid ([PubChem](https://pubchem.ncbi.nlm.nih.gov/)).
neg_pks: dict
    A dictionary that contains the negative pKa values of the amino acids.
pos_pks: dict
    A dictionary that contains the positive pKa values of the amino acids.
token2label: dict
    A dictionary that contains the token to label mapping for label encoding.
    The amino acids are indexed from 1 to 20 in alphabetical order and
    the modifications are indexed from 21 to 28.
tpsas: dict
    A dictionary that contains the topological polar surface area of the amino acids, as retrieved from [Adhav & Saikrishnan, 2023](https://pubs.acs.org/doi/10.1021/acsomega.3c00205).
weights: dict
    A dictionary that contains the molecular weights of the amino acids.
x_logps: dict
    A dictionary that contains the XLogP values of the amino acids per [PubChem](https://pubchem.ncbi.nlm.nih.gov/).

Supported Amino Acids, Post-translations, and Their Labels
----------------------------------------------------------
```
{
    "A": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "K": 9,
    "L": 10,
    "M": 11,
    "N": 12,
    "P": 13,
    "Q": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "V": 18,
    "W": 19,
    "Y": 20,
    "S_p": 21,  # Phosphorylated Serine
    "T_p": 22,  # Phosphorylated Threonine
    "Y_p": 23,  # Phosphorylated Tyrosine
    "C_m": 24,  # Methylated Cysteine
    "R_m": 25,  # Methylated Arginine
    "R_d": 26,  # Dimethylated Arginine
    "R_s": 27,  # Symmetrically dimethylated Arginine
    "K_a": 28   # Acetylated Lysine
}
```
"""

import json
import os
import pathlib
import sys

__PACKAGE_PATH = str(pathlib.Path(__file__).parent.resolve())

__ATTR_NAMES = {fname[:-5] for fname in os.listdir(f"{__PACKAGE_PATH}/data")}

__ATTRS = dict()

for attr_name in __ATTR_NAMES:
    with open(f"{__PACKAGE_PATH}/data/{attr_name}.json") as f:
        attribute = json.load(f)
    __ATTRS[attr_name] = attribute
    setattr(sys.modules[__name__], attr_name, attribute.copy())

setattr(
    sys.modules[__name__],
    "aminoacids",
    list(__ATTRS["token_to_label"].keys()).copy(),
)
