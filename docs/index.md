# peptidy docs

The API documentation for `peptidy`.

Follow these steps to run an `mkdocs` server on the root of the repository:
```bash
conda create --name mkdocs
conda activate mkdocs
python -m pip install mkdocs
python -m pip install "mkdocstrings[python]"
python -m pip install mkdocs-material
mkdocs serve
```
