import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

import peptidy


class TestEncodings:
    p1 = "WYGHP"
    p2 = "S_pSKAC"
    p3 = "A_D_m"
    p4 = "AXGHP"

    def test_one_hot_encoding(self):
        tag = "one_hot"

        exp1 = np.loadtxt(f"./tests/encoding_matrices/{tag}_0.txt").tolist()
        exp1_pad = np.loadtxt(f"./tests/encoding_matrices/{tag}_1.txt").tolist()
        exp2 = np.loadtxt(f"./tests/encoding_matrices/{tag}_2.txt").tolist()

        assert peptidy.encoding.one_hot_encoding(self.p1) == exp1
        assert peptidy.encoding.one_hot_encoding(self.p1, padding_len=7) == exp1_pad
        assert (
            peptidy.encoding.one_hot_encoding(self.p2, add_generative_tokens=True)
            == exp2
        )

        with pytest.raises(ValueError) as e_info:
            peptidy.encoding.one_hot_encoding(self.p3)
        assert (
            e_info.exconly(tryshort=True)
            == "ValueError: Unknown amino acid(s) in peptide: {'A_D_m'}"
        )
        with pytest.raises(ValueError) as e_info:
            peptidy.encoding.one_hot_encoding(self.p4)
        assert (
            e_info.exconly(tryshort=True)
            == "ValueError: Unknown amino acid(s) in peptide: {'X'}"
        )

    def test_label_encoding(self):
        exp1 = [19, 20, 6, 7, 13]
        exp1_pad = [19, 20, 6, 7, 13, 0, 0]
        exp2 = [29, 21, 16, 9, 1, 2, 30]

        assert peptidy.encoding.label_encoding(self.p1) == exp1
        assert peptidy.encoding.label_encoding(self.p1, padding_len=7) == exp1_pad
        assert (
            peptidy.encoding.label_encoding(self.p2, add_generative_tokens=True) == exp2
        )

        with pytest.raises(ValueError) as e_info:
            peptidy.encoding.one_hot_encoding(self.p3)
        assert (
            e_info.exconly(tryshort=True)
            == "ValueError: Unknown amino acid(s) in peptide: {'A_D_m'}"
        )
        with pytest.raises(ValueError) as e_info:
            peptidy.encoding.one_hot_encoding(self.p4)
        assert (
            e_info.exconly(tryshort=True)
            == "ValueError: Unknown amino acid(s) in peptide: {'X'}"
        )

    def test_blosum62_encoding(self):
        tag = "blosum62"

        exp1 = np.loadtxt(f"./tests/encoding_matrices/{tag}_0.txt").tolist()
        exp1_pad = np.loadtxt(f"./tests/encoding_matrices/{tag}_1.txt").tolist()
        exp2 = np.loadtxt(f"./tests/encoding_matrices/{tag}_2.txt").tolist()
        exp2_not_pt = np.loadtxt(f"./tests/encoding_matrices/{tag}_3.txt").tolist()
        exp2_add_special = np.loadtxt(f"./tests/encoding_matrices/{tag}_4.txt").tolist()

        assert peptidy.encoding.blosum62_encoding(self.p1) == exp1
        assert peptidy.encoding.blosum62_encoding(self.p1, padding_len=7) == exp1_pad
        assert peptidy.encoding.blosum62_encoding(self.p2) == exp2
        assert (
            peptidy.encoding.blosum62_encoding(self.p2, encode_post_translation=False)
            == exp2_not_pt
        )
        assert (
            peptidy.encoding.blosum62_encoding(
                self.p2, add_generative_tokens=True, generative_token_value=-2
            )
            == exp2_add_special
        )

        with pytest.raises(ValueError) as e_info:
            peptidy.encoding.blosum62_encoding(self.p3)
        assert (
            e_info.exconly(tryshort=True)
            == "ValueError: Unknown amino acid(s) in peptide: {'A_D_m'}"
        )
        with pytest.raises(ValueError) as e_info:
            peptidy.encoding.blosum62_encoding(self.p4)
        assert (
            e_info.exconly(tryshort=True)
            == "ValueError: Unknown amino acid(s) in peptide: {'X'}"
        )

    def test_aminoacid_descriptor_encoding(self):
        # possible descriptors:
        # ['aliphatic_index', 'aromaticity', 'charge', 'charge_density',
        # 'hydrophobic_aa_ratio', 'isoelectric_point', 'n_C', 'n_H', 'n_N',
        # 'n_O', 'n_S', 'n_P', 'molecular_weight', 'n_h_donors', 'n_h_acceptors',
        # 'topological_polar_surface_area', 'energy_based_on_logP', 'average_number_rotatable_bonds']

        # WYGHP
        exp1 = list()
        n_h_don_list = {"W": 3, "Y": 3, "G": 2, "H": 3, "P": 2}
        n_h_acc_list = {"W": 3, "Y": 4, "G": 3, "H": 4, "P": 3}
        for aa in self.p1:
            aa_mol = Chem.MolFromSequence(aa)
            hydrophob = 0.0
            mol_weight = Descriptors.MolWt(aa_mol)
            n_h_don = n_h_don_list[aa]
            n_h_acc = n_h_acc_list[aa]
            num_rot_bonds = Lipinski.NumRotatableBonds(aa_mol)
            exp1.append([hydrophob, mol_weight, n_h_don, n_h_acc, num_rot_bonds])

        res1 = peptidy.encoding.aminoacid_descriptor_encoding(
            self.p1,
            descriptor_names=[
                "hydrophobic_aa_ratio",
                "molecular_weight",
                "n_h_donors",
                "n_h_acceptors",
                "average_number_rotatable_bonds",
            ],
        )

        # S_pSKAC
        exp2 = list()
        n_h_don_list = {"p": 4, "S": 3, "K": 3, "A": 2, "C": 3}
        n_h_acc_list = {"p": 7, "S": 4, "K": 4, "A": 3, "C": 4}
        for aa in self.p2[2:]:
            if aa == "p":
                aa_mol = Chem.MolFromSmiles("C(C(C(=O)O)N)OP(=O)(O)O")
            else:
                aa_mol = Chem.MolFromSequence(aa)
            hydrophob = aa == "A" or aa == "C"
            hydrophob = float(hydrophob)
            mol_weight = Descriptors.MolWt(aa_mol)
            n_h_don = n_h_don_list[aa]
            n_h_acc = n_h_acc_list[aa]
            num_rot_bonds = Lipinski.NumRotatableBonds(aa_mol)
            exp2.append([hydrophob, mol_weight, n_h_don, n_h_acc, num_rot_bonds])

        print(self.p2)
        res2 = peptidy.encoding.aminoacid_descriptor_encoding(
            self.p2,
            descriptor_names=[
                "hydrophobic_aa_ratio",
                "molecular_weight",
                "n_h_donors",
                "n_h_acceptors",
                "average_number_rotatable_bonds",
            ],
        )

        np.testing.assert_almost_equal(res1, exp1, decimal=2)
        np.testing.assert_almost_equal(res2, exp2, decimal=2)

    def test_peptide_descriptor_encoding(self):
        # ['aliphatic_index', 'freq_A', 'freq_C', 'freq_D', 'freq_E', 'freq_F',
        # 'freq_G', 'freq_H', 'freq_I', 'freq_K', 'freq_L', 'freq_M', 'freq_N',
        # 'freq_P', 'freq_Q', 'freq_R', 'freq_S', 'freq_T', 'freq_V', 'freq_W',
        # 'freq_Y', 'freq_S_p', 'freq_T_p', 'freq_Y_p', 'freq_C_m', 'freq_R_m',
        # 'freq_R_d', 'freq_R_s', 'freq_K_a', 'aromaticity', 'average_number_rotatable_bonds',
        # 'charge', 'charge_density', 'energy_based_on_logP', 'hydrophobic_aa_ratio',
        # 'instability_index', 'isoelectric_point', 'length', 'n_C', 'n_H', 'n_N', 'n_O',
        # 'n_S', 'n_P', 'molecular_weight', 'n_h_donors', 'n_h_acceptors', 'topological_polar_surface_area']

        # 0: WYGHP
        # 1: S_pSKAC
        peptides = [self.p1, self.p2]
        exp = list()

        n_h_don_list = [
            {"W": 3, "Y": 3, "G": 2, "H": 3, "P": 2},
            {"S_p": 4, "S": 3, "K": 3, "A": 2, "C": 3},
        ]
        n_h_acc_list = [
            {"W": 3, "Y": 4, "G": 3, "H": 4, "P": 3},
            {"S_p": 7, "S": 4, "K": 4, "A": 3, "C": 4},
        ]

        for i in range(2):
            aa = (
                Chem.MolFromSequence(peptides[i])
                if i == 0
                else Chem.MolFromSmiles(
                    "N[C@@]([H])(COP(=O)(O)O)C(=O)N[C@@]([H])(CO)C(=O)N[C@@]([H])(CCCCN)C(=O)N[C@@]([H])(C)C(=O)N[C@@]([H])(CS)C(=O)O"
                )
            )
            hydrophob = 0.0 if i == 0 else 0.4
            mol_weight = Descriptors.MolWt(aa)
            n_h_don = sum(n_h_don_list[i].values())
            n_h_acc = sum(n_h_acc_list[i].values())
            if "P" in peptides[i][1:]:
                num_rot_bonds = (
                    Lipinski.NumRotatableBonds(aa) - len(peptides[i]) + 2
                ) / len(peptides[i])
            elif "_" in peptides[i]:
                # [SR]: looks a bit ugly because of the S_p in the second peptide
                num_rot_bonds = (
                    Lipinski.NumRotatableBonds(aa) - (len(peptides[i]) - 2) + 1
                ) / (len(peptides[i]) - 2)
            exp.append([hydrophob, mol_weight, n_h_don, n_h_acc, num_rot_bonds])

        res1 = peptidy.encoding.peptide_descriptor_encoding(
            self.p1,
            descriptor_names=[
                "hydrophobic_aa_ratio",
                "molecular_weight",
                "n_h_donors",
                "n_h_acceptors",
                "average_number_rotatable_bonds",
            ],
        )
        res2 = peptidy.encoding.peptide_descriptor_encoding(
            self.p2,
            descriptor_names=[
                "hydrophobic_aa_ratio",
                "molecular_weight",
                "n_h_donors",
                "n_h_acceptors",
                "average_number_rotatable_bonds",
            ],
        )

        np.testing.assert_almost_equal(res1, exp[0], decimal=1)
        np.testing.assert_almost_equal(res2, exp[1], decimal=2)
