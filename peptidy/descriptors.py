import sys
from collections import Counter
from typing import Dict, List, Union

from peptidy import biology, tokenizer


def aliphatic_index(peptide: str) -> float:
    """
    Calculate the aliphatic index of a peptide.

    The aliphatic index is a measure of the thermal stability of a peptide. It is defined as the volume
    of a protein that is occupied by aliphatic side chains of amino acids, such as alanine, valine,
    isoleucine, and leucine ([Ikai, 1980](https://academic.oup.com/jb/article-abstract/88/6/1895/773432?redirectedFrom=fulltext)).

    Aliphatic index is calculated as:

    $\\textit{aliphatic_index} = 100 * (f_A + 2.9 f_V + 3.9 f_I + 3.9 f_L)$,

    where $f_A$, $f_V$, $f_I$, and $f_L$ are the frequencies of alanine, valine, isoleucine, and leucine, respectively.

    Parameters
    ----------
    peptide : str
        Amino acid sequence of the peptide.

    Returns
    -------
    float
        The aliphatic index of the peptide.

    Raises
    ------
    ValueError
        If the peptide sequence contains unknown amino acids or a syntax error.

    Examples
    --------
    >>> aliphatic_index("AVIL")
    292.5
    >>> # doubling the length of the peptide (with no aliphatic amino acids) halves the aliphatic index
    >>> aliphatic_index('AVILMNPS_p')
    146.25
    >>> aliphatic_index("AKLVT")
    156.0
    >>> aliphatic_index('ACD')
    33.3...
    >>> # Equals to 0, if the peptide contains no aliphatic amino acids
    >>> aliphatic_index("WYGHP")
    0.0
    >>> aliphatic_index('DEFR_mSC')
    0.0
    """
    tokenized_peptide = tokenizer.tokenize_peptide(peptide)
    aa_counts = dict(Counter(tokenized_peptide))
    aa_counts = {aa: count / length(peptide) * 100 for aa, count in aa_counts.items()}
    return (
        aa_counts.get("A", 0)
        + 2.9 * aa_counts.get("V", 0)
        + 3.9 * (aa_counts.get("I", 0) + aa_counts.get("L", 0))
    )


def aminoacid_frequencies(peptide: str) -> Dict[str, float]:
    """
    Calculate the frequency (count / peptide length) of all amino acids in the input sequence.

    Parameters
    ----------
    peptide : str
        Amino acid sequence of the peptide.

    Returns
    -------
    Dict[str, float]
        A dictionary that contains the frequencies of all amino acids and post-translations,
        not only the ones present in the sequence. The keys have the format `freq_<AA>` (`AA` = letter code of the
        amino acid).

    Raises
    ------
    ValueError
        If the peptide sequence contains unknown amino acids or a syntax error.

    Examples
    --------
    >>> freqs = aminoacid_frequencies("AVIL")
    >>> freqs["freq_A"]
    0.25
    >>> freqs["freq_V"]
    0.25
    >>> freqs["freq_C_m"]
    0.0
    >>> freqs["freq_R"]
    0.0
    >>> freqs = aminoacid_frequencies('AC_mD')
    >>> freqs["freq_C_m"]
    0.3...
    >>> freqs["freq_D"]
    0.3...
    >>> freqs["freq_C"]
    0.0
    >>> freqs["freq_W"]
    0.0
    >>> aminoacid_frequencies('AXR')
    Traceback (most recent call last):
        ...
    ValueError: Unknown amino acid(s) in peptide: {'X'}
    """
    tokenized_peptide = tokenizer.tokenize_peptide(peptide)
    aa_counts = dict(Counter(tokenized_peptide))
    peptide_len = length(tokenized_peptide)
    return {
        f"freq_{aa}": aa_counts.get(aa, 0) / peptide_len for aa in biology.aminoacids
    }


def aromaticity(peptide: str) -> float:
    """
    Calculate the sum of the frequencies of aromatic amino-acids
    ("F", "W", "Y", and "Y_p") as a measure of aromaticity of a peptide
    ([Lobry,1994](https://academic.oup.com/nar/article-abstract/22/15/3174/1087817)).

    Parameters
    ----------
    peptide : str
        Amino acid sequence of the peptide.

    Returns
    -------
    float
        Frequency of aromatic residues in the peptide.

    Raises
    ------
    ValueError
        If the peptide sequence contains unknown amino acids or a syntax error.

    Examples
    --------
    >>> aromaticity("AVIL")
    0.0
    >>> aromaticity("WYGHP")
    0.4
    >>> aromaticity('ACDEF')
    0.2
    >>> aromaticity('DAY_pCDFWY')
    0.5
    """
    peptide = tokenizer.tokenize_peptide(peptide)
    counts = dict(Counter(peptide))
    return sum([counts.get(aa, 0) for aa in biology.aromatic_aas]) / length(peptide)


def average_n_rotatable_bonds(
    peptide: str,
) -> float:
    """
    Calculate the number of total rotatable bonds divided by the number of amino acids in the peptide.

    Chain flexibility is known to play a role in binding [Francesca Peccati & Gonzalo Jiménez-Osés, 2021]{https://pubs.acs.org/doi/10.1021/acsomega.1c00485}.
    The number of rotatable bonds per amino acid was retrieved from PubChem [Kim et al., 2023](https://academic.oup.com/nar/article/51/D1/D1373/6777787?login=true).

    Parameters
    ----------
    peptide : str
        Amino acid sequence of the peptide.

    Returns
    -------
    float
        Average number of rotatable bonds in the peptide.

    Raises
    ------
    ValueError
        If the peptide sequence contains unknown amino acids or a syntax error.

    Examples
    --------
    >>> average_n_rotatable_bonds("AVIL")
    2.25
    >>> average_n_rotatable_bonds("WYGHP")
    2.2
    >>> average_n_rotatable_bonds('ACD')
    2.0
    """
    tokenized_peptide = tokenizer.tokenize_peptide(peptide)
    return sum([biology.n_rotatable_bonds[aa] for aa in tokenized_peptide]) / length(
        peptide
    )


def charge(peptide: str, pH: float = 7) -> float:
    """
    Calculate the total charge of the sequence.

    The method used is first described by Bjellqvist [Bjellqvist et al., 1993][https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/elps.11501401163].
    The total charge is then calculated based on the Henderson-Hasselbach equation [Aronson, 1983](https://www.sciencedirect.com/science/article/pii/0307441283900468?via%3Dihub).
    Pka of phosphoserine and phosphothreonine were retrieved from [Xie,Jiang & Ben-Amotz, 2005](https://www.sciencedirect.com/science/article/pii/S0003269705004124?via%3Dihub).
    Pka of phosphotyrosine was taken from [Wojciechowski M et al., 2003](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1302655/#:~:text=The%20pKa%20value%20for,et%20al.%2C%201994)
    Pka of the posttranslational of arginine was kept equal to the pka of arginine, based on [Evich M et al.,2015](https://onlinelibrary.wiley.com/doi/full/10.1002/pro.2838)
    Parameters
    ----------
    peptide : str
        Amino acid sequence of the peptide.
    pH : float, optional
        pH at which to calculate charge, by default 7.

    Returns
    -------
    float
        The total charge of the sequence.

    Examples
    --------
    >>> charge('ACD', pH=13)
    -2.999...
    >>> charge('NNNNRKTNGDDSLF')
    -0.238...
    """
    tokenized_peptide = tokenizer.tokenize_peptide(peptide)
    aa_counts = dict(Counter(tokenized_peptide))
    aa_counts["Nterm"] = 1.0
    aa_counts["Cterm"] = 1.0

    pos_crs = {aa: 10 ** (pk - pH) for aa, pk in biology.pos_pks.items()}
    pos_partial_charges = {aa: cr / (cr + 1) for aa, cr in pos_crs.items()}
    pos_charge = sum(
        [aa_counts.get(aa, 0) * pc for aa, pc in pos_partial_charges.items()]
    )

    neg_crs = {aa: 10 ** (pH - pk) for aa, pk in biology.neg_pks.items()}
    neg_partial_charges = {aa: cr / (cr + 1) for aa, cr in neg_crs.items()}
    neg_charge = sum(
        [aa_counts.get(aa, 0) * pc for aa, pc in neg_partial_charges.items()]
    )

    return pos_charge - neg_charge


def charge_density(peptide: str, pH: float = 7) -> float:
    """
    Calculate the charge of the peptide normalized by weight, *i.e.,* charge / molecular weight.

    Parameters
    ----------
    peptide : str
        Amino acid sequence of the peptide.
    pH : float, optional
        pH at which to calculate charge, by default 7.

    Returns
    -------
    float
        Charge density.

    Examples
    --------
    >>> charge_density('KTTENGD')
    -0.00161...
    >>> charge_density('FPAL', pH=13)
    -0.00223...
    """
    return charge(peptide, pH) / molecular_weight(peptide)


def hydrophobic_aa_ratio(peptide: str) -> float:
    """
    Calculate the total ratio of hydrophobic amino-acids (A, C, C_m, F, I, L, M, and V) in a peptide.
    ([Nelson & Cox, 2004](https://mis.kp.ac.rw/admin/admin_panel/kp_lms/files/digital/Core%20Books/Core%20Books%20In%20Nursing%20%20And%20%20Midwifery/H106_%20Biochemistry_Lehninger%20Principles%20of%20Biochemistry,%20Fourth%20Edition%20-%20David%20L.%20Nelson,%20Michael%20M.%20Cox.pdf))

    Parameters
    ----------
    peptide : str
        Amino acid sequence of the peptide.

    Returns
    -------
    float
        Total ratio of hydrophobic amino-acids of the peptide.

    Examples
    --------
    >>> hydrophobic_aa_ratio('FC_mPR_mS_pA')
    0.5
    >>> hydrophobic_aa_ratio('FPR_mXS_pA')
    Traceback (most recent call last):
        ...
    ValueError: Unknown amino acid(s) in peptide: {'X'}
    """
    tokenized_peptide = tokenizer.tokenize_peptide(peptide)
    aa_counts = dict(Counter(tokenized_peptide))
    return sum([aa_counts.get(aa, 0) for aa in biology.hydrophobic_aas]) / length(
        peptide
    )


def instability_index(peptide: str) -> float:
    """
    Calculate the instability index of the peptide.

    The instability index is based on amino acid compositions and computed by summing the instability coefficient
    of all dipeptide combinations in the peptide. It is based on the frequency of the dipeptide
    occurring in stable versus unstable proteins [Guruprasad, Reddy & Pandit, 1990](https://academic.oup.com/peds/article-abstract/4/2/155/1491271).
    A value of 1 is used for amino acid pairs whose instability coefficient is unavailable.

    Parameters
    ----------
    peptide : str
        Amino acid sequence of the peptide.

    Returns
    -------
    float

    Examples
    --------
    >>> # from https://rdrr.io/cran/Peptides/src/R/boman.R
    >>> instability_index('ACFEGM')
    81.566...
    >>> instability_index('FPP_mS_pA')
    Traceback (most recent call last):
        ...
    ValueError: Unknown amino acid(s) in peptide: {'P_m'}
    """
    tokenized_peptide = tokenizer.tokenize_peptide(peptide)
    aa_pairs = zip(tokenized_peptide, tokenized_peptide[1:])
    instabilities = [biology.instabilities[aa1][aa2] for aa1, aa2 in aa_pairs]
    return sum(instabilities) * 10 / length(peptide)


def isoelectric_point(peptide: str) -> float:
    """
    Calculate the isoelectric point (pH that the peptide carries no net charge) of the peptide.

    The isoelectric point is calculated using the peptide charge at different pH values.
    The method used is based on the Henderson-Hasselbach equation [Aronson, 1983](https://www.sciencedirect.com/science/article/pii/0307441283900468?via%3Dihub).

    Parameters
    ----------
    peptide : str
        Amino acid sequence of the peptide.

    Returns
    -------
    float
        Isoelectric point of the peptide.

    Examples
    --------
    >>> isoelectric_point('ADEFGHI')
    4.35...
    >>> isoelectric_point('AMSTV')
    5.5234375
    """
    test_ph = 7
    peptide_charge = charge(peptide, test_ph)
    if peptide_charge < 0:
        lower_limit, upper_limit = 0, 7
    else:
        lower_limit, upper_limit = 7, 14

    precision = 10**-4
    while (upper_limit - lower_limit) > precision and abs(peptide_charge) > precision:
        test_ph = (upper_limit + lower_limit) / 2
        peptide_charge = charge(peptide, test_ph)
        if peptide_charge < 0:
            upper_limit = test_ph
        else:
            lower_limit = test_ph

    return test_ph


def length(peptide: str) -> int:
    """
    Calculate the number of amino acids in the peptide.

    Parameters
    ----------
    peptide : str
        Amino acid sequence of the peptide.

    Returns
    -------
    int
        Number of amino acids in the peptide.

    Examples
    --------
    >>> length('ACD')
    3
    >>> length('C_mS_pADDWY')
    7
    >>> length('FP_mS_pA')
    Traceback (most recent call last):
        ...
    ValueError: Unknown amino acid(s) in peptide: {'P_m'}
    """
    return len(tokenizer.tokenize_peptide(peptide))


def molecular_formula(
    peptide: str,
) -> Dict[str, int]:
    """
    Determine the closed molecular formula of the amino acid sequence of the peptide.

    The peptide bonds between amino acids are included in the formula. Molecular formulas
    were retrieved from [PubChem](https://pubchem.ncbi.nlm.nih.gov/).

    Parameters
    ----------
    peptide : str
        Amino acid sequence of the peptide.

    Returns
    -------
    Dict[str, int]
        Count of each element in the peptide where the element symbols are keys and counts are values.
        The keys have the format `n_<element>` (`element` = element symbol).

    Examples
    --------
    >>> formula = molecular_formula('ADEF')
    >>> formula['n_H']
    28
    >>> formula['n_O']
    9
    >>> formula['n_S']
    0
    >>> formula = molecular_formula('PTHRAAPDES')
    >>> formula['n_H']
    69
    >>> formula['n_O']
    17
    >>> formula['n_S']
    0
    """
    tokenized_peptide = tokenizer.tokenize_peptide(peptide)
    aa_formulas = [Counter(biology.formulas[aa]) for aa in tokenized_peptide]
    elements = ["C", "H", "N", "O", "S", "P"]
    peptide_formula = dict()
    for element in elements:
        peptide_formula[f"n_{element}"] = sum(
            [formula[element] for formula in aa_formulas]
        )

    peptide_formula["n_H"] = peptide_formula["n_H"] - 2 * length(peptide) + 2
    peptide_formula["n_O"] = peptide_formula["n_O"] - length(peptide) + 1

    return peptide_formula


def molecular_weight(peptide: str) -> float:
    """
    Calculate the weight (g/mol) of the peptide without peptide bonds.

    The molecular weight of the peptide is calculated by summing the weights of the amino acids and
    subtracting the weight of water for each peptide bond. Molecular weight were retrieved from [PubChem](https://pubchem.ncbi.nlm.nih.gov/).


    Parameters
    ----------
    peptide : str
        Amino acid sequence of the peptide.

    Returns
    -------
    float
        Molecular weight of the peptide.

    Examples
    --------
    >>> molecular_weight('RMK_aS_pCD')
    860.885
    >>> molecular_weight('DEGHI')
    569.56
    """

    tokenized_peptide = tokenizer.tokenize_peptide(peptide)
    total_weight = sum([biology.weights[aa] for aa in tokenized_peptide])
    return total_weight - 18.015 * (length(peptide) - 1)


def n_h_acceptors(peptide: str) -> float:
    """
    Calculate the total number of hydrogen bond acceptors in the peptide.

    The number of hydrogen bond acceptors in the peptide is calculated by summing the number of hydrogen bond acceptors.
    Hydrogen bonds are important in protein-protein interactions [Hubbard & Haider, 2010](https://doi.org/10.1002/9780470015902.a0003011.pub2).
    The number of hydrogen bond acceptors for each amino acid were retrieved from [PubChem](https://pubchem.ncbi.nlm.nih.gov/).

    Parameters
    ----------
    peptide : str
        Amino acid sequence of the peptide.

    Returns
    -------
    float
        Total number of hydrogen bond acceptors in the peptide.

    Examples
    --------
    >>> n_h_acceptors('FSCA')
    14
    >>> n_h_acceptors('FXS_pGNM')
    Traceback (most recent call last):
        ...
    ValueError: Unknown amino acid(s) in peptide: {'X'}
    """
    tokenized_peptide = tokenizer.tokenize_peptide(peptide)
    return sum([biology.n_h_acceptors[aa] for aa in tokenized_peptide])


def n_h_donors(peptide: str) -> float:
    """
    Calculate the total number of hydrogen bond donors in the peptide.

    Hydrogen bonds are important in protein-protein interactions [Hubbard & Haider, 2010](https://doi.org/10.1002/9780470015902.a0003011.pub2).
    The number of hydrogen bond donors for each amino acid were retrieved from [PubChem](https://pubchem.ncbi.nlm.nih.gov/).

    Parameters
    ----------
    peptide : str
        Amino acid sequence of the peptide.

    Returns
    -------
    float
        Total number of hydrogen bond donors in the peptide.

    Examples
    --------
    >>> n_h_donors('VYP')
    7
    >>> n_h_donors('PGU')
    Traceback (most recent call last):
        ...
    ValueError: Unknown amino acid(s) in peptide: {'U'}
    """
    tokenized_peptide = tokenizer.tokenize_peptide(peptide)
    return sum([biology.n_h_donors[aa] for aa in tokenized_peptide])


def topological_polar_surface_area(
    peptide: str,
) -> float:
    """
    Calculate the total topological polar surface area of the peptide.

    The topological polar surface area relates to the Van der Waals forces [Adhav & Saikrishnan, 2023](https://pubs.acs.org/doi/10.1021/acsomega.3c00205).

    Parameters
    ----------
    peptide : str
        Amino acid sequence of the peptide.

    Returns
    ----------
    float
        Total topological polar surface area of the peptide.

    Examples
    ----------
    >>> topological_polar_surface_area('R_dPSRMNPAWE')
    853.19...
    >>> topological_polar_surface_area('AYZ')
    Traceback (most recent call last):
        ...
    ValueError: Unknown amino acid(s) in peptide: {'Z'}
    """
    tokenized_peptide = tokenizer.tokenize_peptide(peptide)
    total_surface = sum([biology.tpsas[aa] for aa in tokenized_peptide])
    return total_surface


def x_logp_energy(peptide: str) -> float:
    """
    Calculate the sum of xlogP index of the peptide divided by the length of the peptide.

    The xlogP index is a measure of the hydrophobicity of the peptide [Chen et al, 2007](https://pubs.acs.org/doi/10.1021/ci700257y)
    and the indices per amino acid was retrieved from [PubChem](https://pubchem.ncbi.nlm.nih.gov/).

    Parameters
    ----------
    peptide : str
        Amino acid sequence of the peptide.

    Returns
    ----------
    float
        xlogP index of the peptide.

    Examples
    ----------
    >>> x_logp_energy('R_dPSRMNPAWE')
    2.9
    >>> x_logp_energy('BCAF')
    Traceback (most recent call last):
        ...
    ValueError: Unknown amino acid(s) in peptide: {'B'}
    """
    tokenized_peptide = tokenizer.tokenize_peptide(peptide)
    energy = sum([-biology.x_logps[aa] for aa in tokenized_peptide])
    return energy / length(peptide)


__DESCRIPTOR_FNS = {
    "aliphatic_index": aliphatic_index,
    "aminoacid_frequencies": aminoacid_frequencies,
    "aromaticity": aromaticity,
    "average_number_rotatable_bonds": average_n_rotatable_bonds,
    "charge": charge,
    "charge_density": charge_density,
    "energy_based_on_logP": x_logp_energy,
    "hydrophobic_aa_ratio": hydrophobic_aa_ratio,
    "instability_index": instability_index,
    "isoelectric_point": isoelectric_point,
    "length": length,
    "molecular_formula": molecular_formula,
    "molecular_weight": molecular_weight,
    "n_h_donors": n_h_donors,
    "n_h_acceptors": n_h_acceptors,
    "topological_polar_surface_area": topological_polar_surface_area,
}

setattr(sys.modules[__name__], "descriptor_names", list(__DESCRIPTOR_FNS.keys()))


def compute_descriptors(
    peptide: str,
    descriptor_names: List[str] = None,
    pH: float = 7,
) -> Dict[str, Union[float, int]]:
    """Computes multiple descriptors of the peptide.

    Parameters
    ----------
    peptide : str
        Amino acid sequence of the peptide.
    descriptor_names : List[str], optional
        A List of descriptor names. If any of the descriptor names is invalid, ValueError is raised.
        Set to None by default, and computes all descriptors in this case.
    pH : float, optional
        pH value to compute charge and charge density (if requested), by default 7

    Returns
    ----------
    Dict[str, Union[float, int]]
        A dictionary that maps descriptor names to their values.

    Raises
    ----------
    ValueError
        ValueError is raised if any of the descriptor names is invalid.

    Examples
    ----------
    >>> compute_descriptors('ACD', ['charge', 'charge_density'], 13)
    {'charge': -2.99..., 'charge_density': -0.00976...}
    >>> compute_descriptors('STY', ['molecular_formula'])
    {'n_C': 16, 'n_H': 23, 'n_N': 3, 'n_O': 7, 'n_S': 0, 'n_P': 0}
    """
    if descriptor_names is None:
        descriptor_names = list(__DESCRIPTOR_FNS.keys())

    diff = set(descriptor_names) - set(__DESCRIPTOR_FNS.keys())
    if len(diff) > 0:
        raise ValueError(
            f"Invalid descriptor names: {diff}. Possible names are: {list(__DESCRIPTOR_FNS.keys())}"
        )

    name_to_descriptor = dict()
    for name in descriptor_names:
        if name in ["charge", "charge_density"]:
            name_to_descriptor[name] = __DESCRIPTOR_FNS[name](peptide, pH=pH)
        elif name in ["aminoacid_frequencies", "molecular_formula"]:
            name_to_descriptor = {
                **name_to_descriptor,
                **__DESCRIPTOR_FNS[name](peptide),
            }
        else:
            name_to_descriptor[name] = __DESCRIPTOR_FNS[name](peptide)
    return name_to_descriptor
