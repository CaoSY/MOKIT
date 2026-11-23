# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Shared pytest configuration for the MOKIT test suite."""

from collections.abc import Iterator

import pytest
import torch


def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Register the ``--device`` command-line option.

    Args:
        parser: The pytest argument parser.

    """
    parser.addoption(
        "--device",
        default=None,
        help="Device type for tests: 'cpu' or the available accelerator type; "
        "use the backend's visibility environment variable to choose devices.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """
    Register the ``accelerator`` marker.

    Args:
        config: The pytest configuration object.

    """
    config.addinivalue_line("markers", "requires_accelerator: requires an available accelerator.")


def _resolve_device(config: pytest.Config) -> torch.device:
    """
    Resolve the device type for the test session.

    Args:
        config: The pytest configuration object.

    Returns:
        The type-only device selected by ``--device``, the available accelerator
        when the option was not given, or CPU when no accelerator is available.

    Raises:
        pytest.UsageError: If an unavailable accelerator is requested.

    """
    requested = config.getoption("--device")
    accelerator = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else None
    if accelerator is not None:
        # Drop any device index; callers receive a type-only device.
        accelerator = torch.device(accelerator.type)
    if requested is None:
        return accelerator if accelerator is not None else torch.device("cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if accelerator is None or requested != accelerator.type:
        raise pytest.UsageError(f"--device={requested} was requested, but that accelerator is not available.")
    return accelerator


@pytest.fixture(scope="session")
def device(request: pytest.FixtureRequest) -> torch.device:
    """
    Return the device type for the test session.

    Args:
        request: The pytest fixture request.

    Returns:
        A type-only device: the available accelerator type or ``torch.device("cpu")``.

    """
    return _resolve_device(request.config)


@pytest.fixture(scope="session", autouse=True)
def _set_up_default_device_context(device: torch.device) -> None:
    """
    Run every test on the session device by default.

    Args:
        device: The session device.

    """
    if device.type == "cpu":
        # CPU is already the process default; avoid installing a TorchFunctionMode.
        return
    # The process exits with the session, so the override is not restored.
    torch.set_default_device(device)


@pytest.fixture
def cpu_default_device() -> Iterator[None]:
    """Temporarily enforce the default device to be CPU."""
    with torch.device("cpu"):
        yield


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """
    Skip ``requires_accelerator``-marked tests when the session device is CPU.

    Args:
        config: The pytest configuration object.
        items: The collected test items.

    """
    default_device_type = _resolve_device(config).type
    skip = pytest.mark.skip(reason="requires an available accelerator")
    for item in items:
        if item.get_closest_marker("requires_accelerator") is not None and default_device_type == "cpu":
            item.add_marker(skip)
