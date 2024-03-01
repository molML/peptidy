import json
import os

from peptidy.encoding import (
    aminoacid_descriptor_encoding,
    blosum62_encoding,
    one_hot_encoding,
    peptide_descriptor_encoding,
)
from peptidy.tokenizer import tokenize_peptide

from peptidy.tokenizer import tokenize_peptide


class TestEncodings:
    homedir = os.path.dirname(__file__).split("tests")[0]

    with open(homedir + "peptidy/data/token_to_label.json") as f:
        data = json.load(f)
        len_tokens = len(data)
        alphabet_aa = list(data.keys())

    def test_one_hot_encoding(self):
        # Test case 1: one_hot_encoding
        test_peptide_1 = "ADEFGHIKM"
        test_peptide_2 = "ACDR_mS_p"
        test_peptide_3 = "AR_mPLM"

        for test_peptide in [test_peptide_1, test_peptide_2, test_peptide_3]:
            output = one_hot_encoding(test_peptide)
            test_peptide = tokenize_peptide(test_peptide)
            expected = [(self.len_tokens + 1) * [0] for _ in range(len(test_peptide))]
            # look up index of amino acid in the alphabet and set the corresponding index in the list to 1
            for i in range(len(test_peptide)):
                expected[i][self.alphabet_aa.index(test_peptide[i]) + 1] = 1
            assert (
                output == expected
            ), f"Expected \n{expected}, but got\n{output}\n length of expected: {len(expected)}, length of output: {len(output)}"

    def test_blosum62_encoding(self):
        # Test case 2: BLOSUM
        # 2.1 without post translation
        test_peptide_1 = "ADEFGHIKM"

        with open(self.homedir + "peptidy/data/blosum62_scores.json") as f:
            blos_data = json.load(f)

        for test_peptide in [test_peptide_1]:
            output = blosum62_encoding(test_peptide, encode_post_translation=False)
            test_peptide = tokenize_peptide(test_peptide)
            expected = [blos_data[aa] for aa in test_peptide]
            assert (
                output == expected
            ), f"Expected \n{expected}, but got\n{output}\n length of expected: {len(expected)}, length of output: {len(output)}"

        # 2.2 with post translation
        test_peptide_2 = "ACDR_mS_p"
        test_peptide_3 = "AR_mPLM"
        for test_peptide in [test_peptide_1, test_peptide_2, test_peptide_3]:
            output = blosum62_encoding(test_peptide, encode_post_translation=True)
            test_peptide = tokenize_peptide(test_peptide)
            expected = list()
            for aa in test_peptide:
                expected.append(
                    blos_data[aa.split("_")[0]] + [1]
                    if "_" in aa
                    else blos_data[aa] + [0]
                )
        assert (
            output == expected
        ), f"Expected \n{expected}, but got\n{output}\n length of expected: {len(expected)}, length of output: {len(output)}"
        # L This gives an error, but I do not really understand why the expected answer contains 3 zeros behind the normal vector. Also, if you rerun encode
        # L also if you repeat this test multiple times, blosum encoding gives a different output (see below)
        for _ in range(10):
            assert expected == blosum62_encoding(
                test_peptide, encode_post_translation=True
            )

        # 2.3 with padding
        for test_peptide in [test_peptide_1]:
            output = blosum62_encoding(
                test_peptide, encode_post_translation=False, padding_len=10
            )
            test_peptide = tokenize_peptide(test_peptide)
            expected = [blos_data[aa] for aa in test_peptide]
            for pad in range(10 - len(test_peptide)):
                expected.append([0] * len(blos_data[aa]))

            assert (
                output == expected
            ), f"Expected \n{expected}, but got\n{output}\n length of expected: {len(expected)}, length of output: {len(output)}"

    def test_aminoacid_descriptor_encoding(self):
        # Test case 3:
        test_peptide_1 = "R_mKGFS_p"
        output = aminoacid_descriptor_encoding(
            test_peptide_1,
            descriptor_names=["charge"],
            padding_len=10,
            add_generative_tokens=False,
            generative_token_value=-1,
            return_dimension_names=False,
        )

        test_peptide_1 = "AAA"
        output = aminoacid_descriptor_encoding(
            test_peptide_1,
            descriptor_names=["charge"],
            padding_len=10,
            add_generative_tokens=False,
            generative_token_value=0,
            return_dimension_names=False,
        )

        assert len(output) == 10, f"Expected length of 10, but got {len(output)}"
        assert output[-2][0] == 0, f"Expected 0, but got {output[-2]}"
        assert output[0][0] != 0, f"Expected not 0, but got {output[0]}"

        test_peptide_2 = "PKM"
        output = aminoacid_descriptor_encoding(
            test_peptide_2,
            descriptor_names=["energy_based_on_logP"],
            padding_len=10,
            add_generative_tokens=False,
            generative_token_value=-1,
            return_dimension_names=False,
        )

        assert len(output) == 10, f"Expected length of 10, but got {len(output)}"
        assert output[-2][0] == -1, f"Expected 0, but got {output[-2]}"
        assert float(output[0][0]) == float(2.5), f"Expected not 0, but got {output[0]}"

        test_peptide_3 = "DC_m"
        output, names = aminoacid_descriptor_encoding(
            test_peptide_3,
            descriptor_names=["molecular_formula", "isoelectric_point"],
            padding_len=10,
            add_generative_tokens=False,
            generative_token_value=-1,
            return_dimension_names=True,
        )
        index_NH = names.index("n_H")

        assert (
            int(output[0][index_NH]) == 7
        ), f"Expected 7, but got {output[0][index_NH]}"
        # This raises a key error already, it can not find the descriptor molecular_formula--> resolved by directly translating molecular formula to n_H etc.

        output = aminoacid_descriptor_encoding(
            test_peptide_3,
            padding_len=10,
            add_generative_tokens=False,
            generative_token_value=-1,
            return_dimension_names=True,
        )
        output, names = aminoacid_descriptor_encoding("A", return_dimension_names=True)
        index_NH = names.index("n_H")

        assert int(output[0][index_NH]) == 7, f"Expected 0, but got {output[0]}"

        output, names = aminoacid_descriptor_encoding("D", return_dimension_names=True)
        index_NH = names.index("n_H")

        assert int(output[0][index_NH]) == 7, f"Expected 0, but got {output[0]}"

    def test_peptide_descriptor_encoding(self):
        test_peptide_1 = "S_pS_pS_p"

        output_3aa = peptide_descriptor_encoding(
            test_peptide_1, descriptor_names=["charge"]
        )
        test_peptide_1 = "S_p"
        output_1aa = peptide_descriptor_encoding(
            test_peptide_1, descriptor_names=["charge"]
        )

        assert (
            output_3aa[0] != 3 * output_1aa[0]
        ), f"Expected roughly \n{3*output_1aa[0]}, but got\n{output_3aa[0]}"

        # THis is not an issue in hindsight, I forgot that of course the alanine does not contribute to the charge, so I had to test it with P instead of A

        test_peptide_2 = "PKM"
        output_PKM, names = peptide_descriptor_encoding(
            test_peptide_2,
            descriptor_names=[
                "energy_based_on_logP",
                "isoelectric_point",
                "molecular_formula",
            ],
            return_dimension_names=True,
        )
        expected = {
            "energy_based_on_logP": 7.4 / len(test_peptide_2),
            "isoelectric_point": 8.75,
        }
        for key in expected.keys():
            index = names.index(key)
            print(key)
            assert (
                output_PKM[index] == expected[key]
            ), f"Expected {expected[key]}, but got {output_PKM[index]}"
