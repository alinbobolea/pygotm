from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings(
    "ignore",
    message=(
        r"'locale\.getdefaultlocale' is deprecated and slated for removal in "
        r"Python 3\.15\..*"
    ),
    category=DeprecationWarning,
    module=r"taichi\._lib\.utils",
)

import taichi as ti  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
TESTS_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))


@pytest.fixture(scope="session", autouse=True)
def taichi_session() -> object:
    """Initialise Taichi once per pytest session (CPU, f64)."""

    ti.init(arch=ti.cpu, default_fp=ti.f64)
    yield
    ti.reset()
