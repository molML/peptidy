import copy
from typing import List, Tuple, Union

from peptidy import biology, descriptors, tokenizer


def aminoacid_descriptor_encoding(
    peptide: str,
    descriptor_names: List[str] = None,
    padding_len: int = None,
    add_generative_tokens: bool = False,
    generative_token_value: int = -1,
    return_dimension_names: bool = False,
) -> Union[List[List[int]], Tuple[List[List[int]], List[str]]]:
    """
    Encode a peptide as sequence of vectors where each vector represents physicochemical properties of amino acids.

    Parameters
    ----------
    peptide : str
        The input peptide sequence.
    descriptor_names : List[str], optional
        The names of the amino acid descriptors to include in the encoding.
        If None, all available descriptors will be used. Defaults to None.
    padding_len :int, optional
        The length to which the encoded vector should be padded.
        If the vector is shorter than padding_len, it will be padded with generative_token_value.
        If it is longer, a ValueError will be raised. Defaults to None.
    add_generative_tokens : bool, optional
        Whether to add special tokens to the beginning and end (`"<beg>"`,`"<end>"`) of the encoded vector for generative applications.
        If True, the `generative_token_value` will be used for the special tokens. Defaults to False.
    generative_token_value : int, optional
        The value to use for the special tokens and padding. Defaults to -1.
    return_dimension_names : bool, optional
        Whether to return the list of descriptor names used in the encoding.
        If True, the function will return a tuple of (vectors, descriptor_names).
        If False, it will only return the encoded vectors. Defaults to False.

    Returns
    -------
    Union[List[List[int]], Tuple[List[List[int]], List[str]]]:
        The encoded vectors representing the peptide sequence.
        If return_dimension_names is True, a tuple of (vectors, descriptor_names) will be returned.

    Raises
    ------
        ValueError: If padding_len is less than the length of the encoded vector.

    Examples
    --------
    >>> aminoacid_descriptor_encoding('ACD', descriptor_names=['molecular_weight', 'average_number_rotatable_bonds'])
    [[89.09, 1.0], [121.16, 2.0], [133.1, 3.0]]
    >>> aminoacid_descriptor_encoding('ACD', descriptor_names=['molecular_weight', 'average_number_rotatable_bonds'], padding_len=6)  # doctest: +NORMALIZE_WHITESPACE
    [[89.09, 1.0], [121.16, 2.0], [133.1, 3.0], [-1, -1], [-1, -1], [-1, -1]]
    >>> aminoacid_descriptor_encoding('ACD', descriptor_names=['molecular_weight', 'average_number_rotatable_bonds'], padding_len=6, add_generative_tokens=True)  # doctest: +NORMALIZE_WHITESPACE
    [[-1, -1], [89.09, 1.0], [121.16, 2.0], [133.1, 3.0], [-1, -1], [-1, -1]]
    >>> aminoacid_descriptor_encoding('ACD', descriptor_names=['molecular_formula'], return_dimension_names=True)  # doctest: +NORMALIZE_WHITESPACE
    ([[3, 7, 1, 2, 0, 0], [3, 7, 1, 2, 1, 0], [4, 7, 1, 4, 0, 0]], ['n_C', 'n_H', 'n_N', 'n_O', 'n_S', 'n_P'])
    >>> aminoacid_descriptor_encoding('R_mKGFS_p', descriptor_names=['charge'], padding_len=2, add_generative_tokens=False, generative_token_value=-1, return_dimension_names=False)
    Traceback (most recent call last):
        ...
    ValueError: Padding length must be greater than or equal to input length
    """
    aa_descriptors = biology.descriptor_per_aas
    if descriptor_names is None:
        descriptor_names = list(aa_descriptors["A"].keys())
    if "molecular_formula" in descriptor_names:
        # replace molecular_formula with n_C, n_H, n_N, n_O, n_S, n_P
        descriptor_names.remove("molecular_formula")
        descriptor_names.extend(["n_C", "n_H", "n_N", "n_O", "n_S", "n_P"])
    tokenized_peptide = tokenizer.tokenize_peptide(peptide)
    vector_len = len(descriptor_names)
    vectors = list()
    if add_generative_tokens:
        vectors.append(vector_len * [generative_token_value])  # add beginning token
    vectors = vectors + [
        [aa_descriptors[token][descriptor] for descriptor in descriptor_names]
        for token in tokenized_peptide
    ]
    if add_generative_tokens:
        vectors.append(vector_len * [generative_token_value])  # add end token

    if padding_len is not None:
        if padding_len < len(vectors):
            raise ValueError(
                "Padding length must be greater than or equal to input length"
            )
        padding_vector = [generative_token_value] * vector_len
        n_padding = padding_len - len(vectors)
        for _ in range(n_padding):
            vectors.append(padding_vector.copy())

    if return_dimension_names:
        return vectors, descriptor_names
    return vectors


def blosum62_encoding(
    peptide: str,
    encode_post_translation: bool = True,
    padding_len: int = None,
    add_generative_tokens: bool = False,
    generative_token_value: int = 0,
    return_dimension_names: bool = False,
) -> Union[List[List[int]], Tuple[List[List[int]], List[str]]]:
    """
    Encodes a peptide sequence using BLOSUM62 encoding.

    Parameters
    ----------
    peptide : str
        The peptide sequence to be encoded.
    encode_post_translation : bool, optional
        Whether to encode post-translation information. Defaults to True.
    padding_len : int, optional
        The length to which the encoded vectors should be padded. Defaults to None.
    add_generative_tokens : bool, optional
        Whether to add special tokens to the encoded vectors. Defaults to False.
    generative_token_value : int, optional
        The value to be used for the special tokens. Defaults to 0 to assume no (dis)similarity.
    return_dimension_names : bool, optional
        Whether to return the names of the dimensions of the BLOSUM encodings. Defaults to False.

    Returns
    -------
    Union[List[List[int]], Tuple[List[List[int]], List[str]]]:
        The BLOSUM vectors representing the peptide sequence.
        The dimension names will also be returned if return_dimension_names is True.

    Raises
    ------
         ValueError: If the padding length is less than the input length.

    Examples
    --------
    >>> blosum62_encoding('A') # doctest: +NORMALIZE_WHITESPACE
    [[4, 0, -2, -1, -2, 0, -2, -1, -1, -1, -1, -2, -1, -1, -1, 1, 0, 0, -3, -2, 0]]
    >>> blosum62_encoding('A', encode_post_translation=False)  # doctest: +NORMALIZE_WHITESPACE
    [[4, 0, -2, -1, -2, 0, -2, -1, -1, -1, -1, -2, -1, -1, -1, 1, 0, 0, -3, -2]]
    >>> blosum62_encoding('ACD')  # doctest: +NORMALIZE_WHITESPACE
    [[4, 0, -2, -1, -2, 0, -2, -1, -1, -1, -1, -2, -1, -1, -1, 1, 0, 0, -3, -2, 0],
     [0, 9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2, 0],
     [-2, -3, 6, 2, -3, -1, -1, -3, -1, -4, -3, 1, -1, 0, -2, 0, -1, -3, -4, -3, 0]]
    >>> blosum62_encoding('ACDK_a', return_dimension_names=True)  # doctest: +NORMALIZE_WHITESPACE
    ([[4, 0, -2, -1, -2, 0, -2, -1, -1, -1, -1, -2, -1, -1, -1, 1, 0, 0, -3, -2, 0],
      [0, 9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2, 0],
      [-2, -3, 6, 2, -3, -1, -1, -3, -1, -4, -3, 1, -1, 0, -2, 0, -1, -3, -4, -3, 0],
      [-1, -3, -1, 1, -3, -2, -1, -3, 5, -2, -1, 0, -1, 1, 2, 0, -1, -2, -3, -2, 1]],
      ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'is_post_translated'])
    >>> blosum62_encoding('ACDK_a', padding_len=7, add_generative_tokens=True)  # doctest: +NORMALIZE_WHITESPACE
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 0, -2, -1, -2, 0, -2, -1, -1, -1, -1, -2, -1, -1, -1, 1, 0, 0, -3, -2, 0],
    [0, 9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2, 0],
    [-2, -3, 6, 2, -3, -1, -1, -3, -1, -4, -3, 1, -1, 0, -2, 0, -1, -3, -4, -3, 0],
    [-1, -3, -1, 1, -3, -2, -1, -3, 5, -2, -1, 0, -1, 1, 2, 0, -1, -2, -3, -2, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    """
    blosum62 = copy.deepcopy(biology.blosum62_scores)
    tokenized_peptide = tokenizer.tokenize_peptide(peptide)
    vector_len = len(blosum62["A"]) + int(encode_post_translation)
    vectors = list()
    if add_generative_tokens:
        vectors.append(vector_len * [generative_token_value])  # add beginning token
    for token in tokenized_peptide:
        aa = token[0]
        aa_vector = blosum62[aa].copy()
        if encode_post_translation:
            is_post_translated = len(token) > 1
            aa_vector.append(int(is_post_translated))
        vectors.append(aa_vector)

    if add_generative_tokens:
        vectors.append(vector_len * [generative_token_value])  # add end token

    if padding_len is not None:
        if padding_len < len(vectors):
            raise ValueError(
                "Padding length must be greater than or equal to input length"
            )
        padding_vector = [generative_token_value] * vector_len
        n_padding = padding_len - len(vectors)
        for _ in range(n_padding):
            vectors.append(padding_vector.copy())

    if return_dimension_names:
        aa_names = list(blosum62.keys())
        if encode_post_translation:
            aa_names.append("is_post_translated")
        return vectors, aa_names

    return vectors


def label_encoding(
    peptide: str,
    padding_len: int = None,
    add_generative_tokens: bool = False,
) -> List[int]:
    """
    Encodes a peptide sequence using label encoding.

    Parameters
    ----------
    peptide : str
        The input peptide sequence to be encoded.
    padding_len : int, optional
        The length to which the encoded sequence should be padded. Defaults to None.
    add_generative_tokens : bool, optional
        Whether to add special tokens at the beginning and end of the encoded sequence. Defaults to False.

    Returns
    -------
    List[int]
        The encoded peptide sequence as a list of integers.

    Raises
    ------
    ValueError
        If the padding length is less than the input length.

    Examples
    --------
    >>> label_encoding('ACD')
    [1, 2, 3]
    >>> label_encoding('ACDK_a', padding_len=7)  # doctest: +NORMALIZE_WHITESPACE
    [1, 2, 3, 28, 0, 0, 0]
    >>> label_encoding('ACD', padding_len=7, add_generative_tokens=True)
    [29, 1, 2, 3, 30, 0, 0]
    >>> label_encoding('AXR')
    Traceback (most recent call last):
        ...
    ValueError: Unknown amino acid(s) in peptide: {'X'}
    >>> label_encoding('A_C_D')
    Traceback (most recent call last):
        ...
    ValueError: Unknown amino acid(s) in peptide: {'A_C_D'}
    """
    token_to_label = biology.token_to_label.copy()
    labels = list()
    if add_generative_tokens:
        labels.append(len(token_to_label) + 1)  # add beginning token
    tokenized_peptide = tokenizer.tokenize_peptide(peptide)
    for token in tokenized_peptide:
        labels.append(token_to_label[token])

    if add_generative_tokens:
        labels.append(len(token_to_label) + 2)  # add end token

    if padding_len is not None:
        if padding_len < len(labels):
            raise ValueError(
                "Padding length must be greater than or equal to input length"
            )
        labels = labels + [0] * (padding_len - len(labels))

    return labels


def one_hot_encoding(
    peptide: str,
    padding_len: int = None,
    add_generative_tokens: bool = False,
) -> List[List[int]]:
    """
    Encodes a peptide sequence using one-hot encoding.

    Parameters
    ----------
    peptide : str
        The input peptide sequence to be encoded.
    padding_len : int, optional
        The length to which the encoded sequence should be padded. Defaults to None.
    add_generative_tokens : bool, optional
        Whether to add `<beg>` and `<end>` tokens at the start and end of the encoded sequence for language modeling applications.
        Defaults to False.


    Returns
    -------
    List[List[int]]:
        The encoded peptide sequence represented as a list of one-hot encoded vectors.

    Raises
    ------
        ValueError: If the padding length is less than the input length.

    Examples
    --------
    >>> one_hot_encoding('ACD')  # doctest: +NORMALIZE_WHITESPACE
    [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    >>> one_hot_encoding('ACD', padding_len=5)  # doctest: +NORMALIZE_WHITESPACE
    [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

     >>> one_hot_encoding('ACDK_a', padding_len=7, add_generative_tokens=True)  # doctest: +NORMALIZE_WHITESPACE
     [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    """

    token_to_label = biology.token_to_label.copy()
    vocab_size = len(token_to_label) + 1  # Add 1 for padding token
    vectors = list()
    if add_generative_tokens:
        vocab_size = vocab_size + 2  # Add 2 for start and end tokens
        beg_vector = [0] * vocab_size
        beg_vector[-2] = 1
        vectors.append(beg_vector)
    tokenized_peptide = tokenizer.tokenize_peptide(peptide)
    for token in tokenized_peptide:
        token_vector = [0] * vocab_size
        token_vector[token_to_label[token]] = 1
        vectors.append(token_vector)

    if add_generative_tokens:
        end_vector = [0] * vocab_size
        end_vector[-1] = 1
        vectors.append(end_vector)

    if padding_len is not None:
        if padding_len < len(vectors):
            raise ValueError(
                "Padding length must be greater than or equal to input length"
            )
        pad_vector = [0] * vocab_size
        pad_vector[0] = 1
        n_padding = padding_len - len(vectors)
        for _ in range(n_padding):
            vectors.append(pad_vector.copy())

    return vectors


def peptide_descriptor_encoding(
    peptide: str,
    descriptor_names: List[str] = None,
    pH: float = 7.0,
    return_dimension_names: bool = False,
) -> Union[List[float], Tuple[List[float], List[str]]]:
    """
    Encodes a peptide sequence using amino acid descriptors.

    Parameters
    ----------
    peptide : str
        The input peptide sequence to be encoded.
    descriptor_names : List[str], optional
        The names of the amino acid descriptors to include in the encoding.
        Defaults to None and all available descriptors are used.
    pH : float, optional
        The pH value to use for charge-based descriptors. Defaults to 7.0.
    return_dimension_names : bool, optional
        Whether to return the list of descriptor names used in the encoding.
        If True, the function will return a tuple of (descriptors : List[float], descriptor_names: List[str]).
        If False, it will only return the encoded vectors. Defaults to False.

    Returns
    -------
    Union[List[float], Tuple[List[float], List[str]]]:
        The descriptor vector representing the peptide sequence.
        If return_dimension_names is True, a tuple of (descriptors : List[float], descriptor_names: List[str]) will be returned.

    Examples
    --------
    >>> peptide_descriptor_encoding('ACD', descriptor_names=['molecular_weight', 'average_number_rotatable_bonds']) # doctest: +ELLIPSIS
    [307.32..., 2.0]
    >>> peptide_descriptor_encoding('AXR')
    Traceback (most recent call last):
        ...
    ValueError: Unknown amino acid(s) in peptide: {'X'}
    """
    all_descriptors = descriptors.compute_descriptors(peptide, descriptor_names, pH)

    if return_dimension_names:
        return list(all_descriptors.values()), list(all_descriptors.keys())

    return list(all_descriptors.values())
