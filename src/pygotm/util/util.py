"""
util --- parameters and interfaces for utilities

This module is an encapsulation of a number of parameters used by different
routines found in the util directory. It should make it easier to read the
code, since finding a line like

    if (method == UPSTREAM) then ...

in a subroutine for advection methods tells you more than reading only

    if (method == 1) then ...

Original FORTRAN author(s): Lars Umlauf

Copyright by the GOTM-team under the GNU Public License - www.gnu.org
"""

__all__ = [
    # Advection scheme types
    "CENTRAL",
    "UPSTREAM",
    "P1",
    "P2",
    "Superbee",
    "MUSCL",
    "P2_PDM",
    "SPLMAX13",
    # Boundary condition types for diffusion
    "Dirichlet",
    "Neumann",
    # Boundary condition types for advection
    "flux",
    "value",
    "oneSided",
    "zeroDivergence",
]

# Advection scheme type constants
CENTRAL: int = -1
UPSTREAM: int = 1
P1: int = 2
P2: int = 3
Superbee: int = 4
MUSCL: int = 5
P2_PDM: int = 6
SPLMAX13: int = 13

# Boundary condition types for diffusion schemes
Dirichlet: int = 0
Neumann: int = 1

# Boundary condition types for advection schemes
flux: int = 1
value: int = 2
oneSided: int = 3
zeroDivergence: int = 4
