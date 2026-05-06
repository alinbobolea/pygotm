"""Translation inventory checks for AGENTS.md's GOTM core mapping."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORTRAN_ROOT = PROJECT_ROOT / "gotm-model" / "code" / "src"
PYTHON_ROOT = PROJECT_ROOT / "src" / "pygotm"
TEST_ROOT = PROJECT_ROOT / "tests"
EXPECTED_CORE_FORTRAN_FILES = 87


def _lower_path(path: Path) -> Path:
    return Path(*(part.lower() for part in path.parts))


def _expected_python_path(fortran_file: Path) -> Path:
    rel = fortran_file.relative_to(FORTRAN_ROOT).with_suffix(".py")
    return _lower_path(rel)


def _expected_test_path(fortran_file: Path) -> Path:
    python_path = _expected_python_path(fortran_file)
    parts = list(python_path.parts)
    parts[-1] = f"test_{parts[-1]}"
    return Path(*parts)


def test_core_fortran_files_have_python_translations() -> None:
    """Every core GOTM Fortran file must have its one-to-one Python module."""
    fortran_files = sorted(FORTRAN_ROOT.rglob("*.F90"))
    python_files = {
        _lower_path(path.relative_to(PYTHON_ROOT))
        for path in PYTHON_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    }

    missing = [
        f"{path.relative_to(FORTRAN_ROOT)} -> src/pygotm/{_expected_python_path(path)}"
        for path in fortran_files
        if _expected_python_path(path) not in python_files
    ]

    assert len(fortran_files) == EXPECTED_CORE_FORTRAN_FILES
    assert not missing, "\n".join(missing)


def test_core_fortran_files_have_corresponding_tests() -> None:
    """Each translated core module must keep its module-level test file."""
    fortran_files = sorted(FORTRAN_ROOT.rglob("*.F90"))
    test_files = {
        _lower_path(path.relative_to(TEST_ROOT))
        for path in TEST_ROOT.rglob("test_*.py")
        if "__pycache__" not in path.parts
    }

    missing = [
        f"{path.relative_to(FORTRAN_ROOT)} -> tests/{_expected_test_path(path)}"
        for path in fortran_files
        if _expected_test_path(path) not in test_files
    ]

    assert len(fortran_files) == EXPECTED_CORE_FORTRAN_FILES
    assert not missing, "\n".join(missing)
