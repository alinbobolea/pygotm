r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: GOTM --- the main program  \label{sec:main}
!
! !INTERFACE:
!   program main
!
! !DESCRIPTION:
! This is the main program of GOTM.
!
! !REVISION HISTORY:
!  Original FORTRAN author(s): Karsten Bolding & Hans Burchard
!
!EOP
!-----------------------------------------------------------------------
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from pygotm.gotm.cmdline import format_help, parse_cmdline
from pygotm.gotm.gotm import finalize_gotm, initialize_gotm, integrate_gotm
from pygotm.gotm.print_version import print_version

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Phase 6 pyGOTM entry point."""

    options = parse_cmdline(list(argv or ()), exe="gotm")
    if options.show_help:
        print(format_help(options.exe))
        return 0
    if options.show_version:
        print(print_version())
        return 0

    config_only = bool(options.write_yaml_path or options.write_schema_path)
    run = initialize_gotm(
        options.yaml_file,
        write_yaml_path=options.write_yaml_path,
        write_yaml_detail=options.write_yaml_detail,
        write_schema_path=options.write_schema_path,
        output_id=options.output_id,
        list_fields=options.list_fields,
        ignore_unknown_config=options.ignore_unknown_config,
    )
    try:
        if options.list_fields:
            print("\n".join(run.registry.list()))
        elif not config_only:
            integrate_gotm(run)
    finally:
        finalize_gotm(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
