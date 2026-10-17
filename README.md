# MOKIT

Model Optimization Kit for Inference and Training. Python >= 3.14, PyTorch >= 2.14.

## Installation

### Users

```bash
pip install git+https://github.com/CaoSY/MOKIT.git
```

To control the version of the installed PyTorch, use the instruction on [PyTorch](https://pytorch.org/) to install PyTorch first, then install MOKIT.

### Developers

```bash
git clone https://github.com/CaoSY/MOKIT.git
cd MOKIT
pip install --group dev -e .
pre-commit install
```

`pre-commit install` is needed once per clone: it writes the git hooks into
`.git/hooks/`, which git does not version-control. The hooks run from your active environment.
