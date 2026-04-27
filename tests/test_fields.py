from __future__ import annotations

from collections.abc import Generator

import pytest
import taichi as ti

from pygotm.fields import ColumnLayout, TaichiFieldCollection


@pytest.fixture(autouse=True)
def taichi_cpu() -> Generator[None, None, None]:
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64)
    yield
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64)


def test_single_column_layout_uses_fortran_style_storage_extent() -> None:
    layout = ColumnLayout(nlev=100)

    assert layout.storage_levels == 101
    assert layout.shape == (101,)
    assert layout.is_multi_column is False


def test_multi_column_collection_allocates_shared_shapes() -> None:
    collection = TaichiFieldCollection(ColumnLayout(nlev=50, n_cols=4))
    collection.allocate_many(["u", "v", "temperature"])

    assert collection.shape == (4, 51)
    assert collection.u.shape == (4, 51)
    assert collection.v.shape == (4, 51)
    assert collection.temperature.shape == (4, 51)
    assert collection.names() == ("u", "v", "temperature")


def test_duplicate_field_names_are_rejected() -> None:
    collection = TaichiFieldCollection(ColumnLayout(nlev=10))
    collection.allocate("u")

    with pytest.raises(ValueError, match="already allocated"):
        collection.allocate("u")
