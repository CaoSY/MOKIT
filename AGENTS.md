# MOKIT — agent notes

Model Optimization Kit for Inference and Training. Python >= 3.14, PyTorch >= 2.14.
Flat layout: the package is `mokit/`, tests are in `tests/`.

This file is the source of truth for agent instructions. `CLAUDE.md` imports it
so Claude Code and other tools share one copy.

## Definition of done

A task is **not complete** until this passes. Verify it — do not assume it:

```
pre-commit run --all-files
```

If it modifies files, re-stage and run it again. Say in your summary that it
passed. Do not report a task finished while this is failing.

Two of the things it checks should be written *as you go*, not fixed up at the
end:

### 1. SPDX header — every new source file

```
# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause
```

`reuse lint` fails without it. Put the header after any shebang and before a
module docstring.

The year is the year **that file** was created. Do not copy it from an existing
file — a file added in 2030 says 2030, not 2026.

### 2. Docstrings — every public module, class, and function/method

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

Enforced by ruff's `D` rules. The conventions are not the defaults:

- Functions and methods: summary in the **imperative mood** — "Add two numbers.",
  not "Adds two numbers." (`D401`)
- Multi-line docstrings: summary on the **second** line, not the first (`D213`)
- Class docstrings: a **blank line** before them (`D203`)
- Document every parameter, return value, and raised exception

Modules and classes are not checked for imperative mood — a noun phrase like
"Utilities for tensor quantization." is correct there.
