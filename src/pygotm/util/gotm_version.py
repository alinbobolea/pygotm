"""
GOTM version constants — translation of ``gotm_version.F90``.

Exposes the GOTM source version identifiers that were embedded at compile time
in the original Fortran build.  Values are fixed strings matching the GOTM
release this Python translation targets.

Public interface: :data:`git_commit_id`, :data:`git_branch_name`.
"""

__all__ = ["git_commit_id", "git_branch_name"]

git_commit_id: str = "4.1.0"
git_branch_name: str = "master"
