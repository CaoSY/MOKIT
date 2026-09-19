# MOKIT — agent notes

Model Optimization Kit for Inference and Training. Python >= 3.14, PyTorch >= 2.14.
Flat layout: the package is `mokit/`, tests are in `tests/`.

This file is the source of truth for agent instructions. `CLAUDE.md` imports it
so Claude Code and other tools share one copy.

## What to keep in mind

Write these as you write the code, not as a cleanup pass.

### SPDX header — every new source file

```
# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause
```

Put it after any shebang and before a module docstring.

The year is the year **that file** was created. Do not copy it from an existing
file — a file added in 2030 says 2030, not 2026.

### Docstrings — every public module, class, and function/method

```python
def quantize(tensor, scale):
    """
    Quantize a tensor to int8.

    Args:
        tensor: The input tensor.
        scale: Per-channel scale factors.

    Returns:
        The quantized tensor.

    Raises:
        ValueError: If `scale` has the wrong shape.
    """
```

- Functions and methods: summary in the **imperative mood** — "Add two numbers.",
  not "Adds two numbers."
- Multi-line docstrings: summary on the **second** line, not the first
- Class docstrings: a **blank line** before them
- Document every parameter, return value, and raised exception

Modules and classes read as a noun phrase rather than an instruction: "Utilities
for tensor quantization."

### Type annotations — every function and method

```python
def quantize(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor: ...
```

Annotate every parameter and return type, and make them accurate — look a type up
rather than guessing, since a wrong annotation is worse than none.
