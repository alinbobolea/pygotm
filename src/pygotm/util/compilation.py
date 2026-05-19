"""
Compilation metadata stub — translation of ``gotm_compilation.F90``.

In the original Fortran build, ``compiler``, ``compiler_id``, and
``compiler_version`` were injected by the CMake build system.  This Python
stub exposes the same names as empty strings; they carry no functional
meaning at runtime.

Public interface: :data:`compiler`, :data:`compiler_id`,
:data:`compiler_version`.
"""

__all__ = ["compiler", "compiler_id", "compiler_version"]

compiler: str = ""
compiler_id: str = ""
compiler_version: str = ""
