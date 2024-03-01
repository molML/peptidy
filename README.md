# peptidy

Welcome to `peptidy` &mdash; a tiny and tidy python library to vectorize peptide sequences (and proteins) for machine learning applications!

`peptidy` bridges the gap between peptide sequences and machine learning applications by converting peptide sequences into numerical vectors with a *single function call*. `peptidy` is obsessed with the simplicity and tidiness &mdash; it is designed to be as accessible as possible for everyone.

Key features of `peptidy` include:
- **Simple**: The programming interface of `peptidy` is designed to be as simple as possible and well-documented to minimize entry barriers. Converting a peptide sequence into a numerical vector is just a single function call.
- **Tiny**: `peptidy` is written in pure Python (>=3.6) and free from external dependencies. It is designed to be light-weight.
- **Broad**: `peptidy` supports a wide range of encoding schemes for both discriminative and generative applications, *e.g.,* one-hot encoding, BLOSUM62, and physicochemical descriptors. Moreover, `peptidy` can handle peptides with post-translational modifications.

## Installation :hammer_and_wrench:
```bash
pip install peptidy
```
Simple :shrug:

## Quick Start :rocket:
The most important module in `peptidy` is `peptidy.encoding`. This module provides functions to convert peptide sequences into numerical vectors. Five encoding methods are implemented: `aminoacid_descriptor_encoding`, `blosum62_encoding`, `label_encoding`, `one_hot_encoding` and `peptide_descriptor_encoding`. All of those functions take a peptide sequence as input and return a numerical vector or matrix. [An extensive documentation is available for all encoding methods](https://molml.github.io/peptidy/api/encoding/) :book:

Here are some quick examples to start with:

```python
# Represent a peptide sequence with label encoding
import peptidy
peptide = "ACD"
padded_labels = peptidy.encoding.label_encoding(peptide, padding_len=5)  # [1, 2, 3, 0, 0]
```
There you go! The peptide "ACD" is converted into a list of labels, padded, and is ready to be fed into a deep learning model.


## Post-translational Modifications :sparkles:
`peptidy` can handle peptides with post-translational modifications, such as phosphorylation, acetylation, and glycosylation. You can find a full list of supported modifications in the [documentation](https://molml.github.io/peptidy/api/biology/).

```python
# Represent a peptide sequence with one-hot encoding and phosphorylation
peptide = "ACS_pD"  # S_p stands for phosphorylated Serine
padded_one_hot = peptidy.encoding.one_hot_encoding(peptide, padding_len=5)  # (5, 29)
```

## Computing Peptide Descriptors :desktop_computer:
Peptides are commonly represented by their physicochemical properties in machine learning models and `peptidy` provides functions to facilitate those models. In addition to providing `peptidy.encoding.peptide_descriptor_encoding` function, `peptidy` also exposes a module, `peptidy.descriptors` with many peptide descriptors, *e.g.,* charge, hydrophobicity, and instability index. The full list and descriptions of supported descriptors is available on the [documentation page](https://molml.github.io/peptidy/api/descriptors/).

```python
# Compute molecular weight of a peptide
peptide = "ACS_pD"
descriptors = peptidy.descriptors.molecular_weight(peptide)  # 474.375
```

## Concluding Remarks

This was a quick introduction to capabilities of `peptidy`. But `peptidy` can empower much more! We provide classification and generation [examples](https://github.com/molML/peptidy/examples) using xgboost, tensorflow, and keras, where peptidy smooths the computations. Feel free to check our [documentation](https://molml.github.io/peptidy/) for more information and examples and create issues or pull requests on our [GitHub repository](https://github.com/molML/peptidy/issues) for added functionalities or bug reports.

See you in the forums! :wave:
