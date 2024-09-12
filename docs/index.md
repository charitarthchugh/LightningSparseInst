# LightningSparseInst

<p align="center">
    <a href="https://github.com/charitarthchugh/lightningsparseinst/actions/workflows/test_suite.yml"><img alt="CI" src=https://img.shields.io/github/workflow/status/charitarthchugh/lightningsparseinst/Test%20Suite/main?label=main%20checks></a>
    <a href="https://charitarthchugh.github.io/lightningsparseinst"><img alt="Docs" src=https://img.shields.io/github/deployments/charitarthchugh/lightningsparseinst/github-pages?label=docs></a>
    <a href="https://github.com/grok-ai/nn-template"><img alt="NN Template" src="https://shields.io/badge/nn--template-0.4.0-emerald?style=flat&labelColor=gray"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.11-blue.svg"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

SparseInst (CVPR 2022) in PyTorch Lightning


## Installation

```bash
pip install git+ssh://git@github.com/charitarthchugh/lightningsparseinst.git
```


## Quickstart

[comment]: <> (> Fill me!)


## Development installation

Setup the development environment:

```bash
git clone git@github.com:charitarthchugh/lightningsparseinst.git
cd lightningsparseinst
conda env create -f env.yaml
conda activate lightningsparseinst
pre-commit install
```

Run the tests:

```bash
pre-commit run --all-files
pytest -v
```


### Update the dependencies

Re-install the project in edit mode:

```bash
pip install -e .[dev]
```
