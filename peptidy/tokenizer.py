from typing import List

from peptidy import biology

__AAs = set(biology.aminoacids)


def tokenize_peptide(peptide: str) -> List[str]:
    """
    Tokenize a peptide sequence into its constituent amino acids.
    The amino acids are represented by their upper-case one-letter codes.
    Post-translational modifications are also supported,
    and are represented as `"<aa>_<mod>"` where `<aa>` is the amino acid and `<mod>` is the modification.
    The list supported amino acids and modifications can be found in the `peptidy.biology` module.

    Parameters
    ----------
    peptide : str
        A peptide sequence.

    Returns
    -------
    List[str]
        A list of tokens, each representing an amino acid
        (possibly with a post-translational modification) in the peptide sequence.

    Raises
    ------
    ValueError
        If the peptide sequence contains unknown amino acids or a syntax error.

    Examples
    --------
    >>> tokenize_peptide("ACDEF")
    ['A', 'C', 'D', 'E', 'F']
    >>> tokenize_peptide("ACK_aDGH")
    ['A', 'C', 'K_a', 'D', 'G', 'H']
    >>> tokenize_peptide("S_pT_p")
    ['S_p', 'T_p']
    >>> tokenize_peptide('ACD')
    ['A', 'C', 'D']
    >>> tokenize_peptide('R_mRGD')
    ['R_m', 'R', 'G', 'D']
    >>> tokenize_peptide('AXR')
    Traceback (most recent call last):
        ...
    ValueError: Unknown amino acid(s) in peptide: {'X'}
    >>> tokenize_peptide('A_C_D')
    Traceback (most recent call last):
        ...
    ValueError: Unknown amino acid(s) in peptide: {'A_C_D'}
    """

    if "_" not in peptide:
        tokens = list(peptide)
        diff = set(tokens).difference(__AAs)
        if len(diff) > 0:
            raise ValueError("Unknown amino acid(s) in peptide: " + str(diff))
        return tokens

    seq_len = len(peptide)
    tokens = list()
    char_ix = 0
    while char_ix < seq_len:
        char = peptide[char_ix]
        if char == "_":
            tokens[-1] = tokens[-1] + char + peptide[char_ix + 1]
            char_ix = char_ix + 2
        else:
            tokens.append(char)
            char_ix = char_ix + 1
    diff = set(tokens).difference(__AAs)
    if len(diff) > 0:
        raise ValueError("Unknown amino acid(s) in peptide: " + str(diff))

    return tokens
