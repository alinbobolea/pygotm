# ruff: noqa: E501
r"""
!-------------------------------------------------------------------------
!BOP
!
! !ROUTINE: Printing GOTM library version
!
! !INTERFACE:
      subroutine gotm_lib_version(unit)
!
! !DESCRIPTION:
!  Simply prints the version number of the GOTM turbulence library to unit.
!
! !USES:
   use gotm_version
   IMPLICIT NONE
!
! !INPUT PARAMETERS:
   integer, intent(in)                 :: unit
!
! !REVISION HISTORY:
!  Original FORTRAN author(s): Karsten Bolding & Hans Burchard
!
!EOP
!-------------------------------------------------------------------------
!BOC
   write(unit,*) 'GOTM library version: ',git_commit_id
!
   return
   end
!EOC
!
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

from typing import TextIO

from pygotm.util.gotm_version import git_commit_id

__all__ = ["gotm_lib_version"]


def gotm_lib_version(unit: TextIO) -> None:
    """Write the translated GOTM turbulence-library version string to ``unit``."""

    unit.write(f"GOTM library version: {git_commit_id}\n")
