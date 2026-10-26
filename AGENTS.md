# MOKIT — agent notes

Model Optimization Kit for Inference and Training. Python >= 3.14, PyTorch >= 2.14.
Flat layout: the package is `mokit/`, tests are in `tests/`.

This file is the source of truth for agent instructions. `CLAUDE.md` imports it
so Claude Code and other tools share one copy.

## Git

`main` is protected: direct pushes are rejected. Create a branch, push it, and
open a pull request. Do not attempt to push to `main`.

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

### How to organize unit tests

Tests live under `tests/`, mirroring the `mokit/` package layout.

Name each test file after the module it tests: `mokit/foo/bar.py` becomes
`tests/foo/bar_test.py`. For a package's `__init__.py`, use the package directory
name: `mokit/foo/__init__.py` becomes `tests/foo/foo_test.py`, not
`tests/foo/__init___test.py`. For example, `mokit/quantization/__init__.py` is tested
by `tests/quantization/quantization_test.py`.

Group tests by the class or function under test.

- Give each tested class or function its own section.
- Put tests for one of its methods, or for a helper function or class used only by
  that class's tests, in a subsection of that class's section.
- Keep class-level tests (for example, frozen or slotted dataclasses) in the class
  section.
- Use one combined section for a test parametrized across several related classes,
  for example `FloatFormat and FloatQuantConfig` or `MX format classes`.
- Use a `Serialization` section for cross-cutting weights-only loading tests.
- Use a section named for the group for module-level constants, for example
  `Dtype groups`.
- Do not add sections for classes or functions without tests.

Sections use `=` rules above and below the title; subsections use `-` rules:

```python
# ==============================================================================
# Class or function name
# ==============================================================================

# ------------------------------------------------------------------------------
# Method or helper name
# ------------------------------------------------------------------------------
```
