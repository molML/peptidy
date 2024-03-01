import pathlib

import setuptools

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")


setuptools.setup(
    name="peptidy",
    version="0.0.1",
    description="A Python library that converts peptide sequences to matrices for machine learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Rıza Özçelik",
    author_email="r.ozcelik@tue.nl",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "Intended Audience :: Science/Research",
    ],
    keywords="peptides, proteins, descriptors, machine learning, deep learning",
    python_requires=">=3.6, <4",
    install_requires=[],
    packages=setuptools.find_packages(exclude=["*tests*"]),
    package_data={
        "peptidy": ["./data/*"],
    },
    readme="README.md",
    license="MIT",
    url="https://github.com/molML/peptidy/",
    project_urls={
        "Bug Reports": "https://molml.github.io/peptidy/issues/",
        "Documentation": "https://molml.github.io/peptidy/",
        "Source": "https://github.com/molML/peptidy/",
    },
)
