"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: Command line parsing
!
! !INTERFACE:
!   module cmdline
!
! !DESCRIPTION:
!
! !REVISION HISTORY:
!  Original author(s): Karsten Bolding & Hans Burchard
!
!EOP
!-----------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = ["CommandLineOptions", "format_help", "parse_cmdline"]


@dataclass(frozen=True)
class CommandLineOptions:
    exe: str = "gotm"
    yaml_file: str = "gotm.yaml"
    write_yaml_path: str = ""
    write_schema_path: str = ""
    output_id: str = ""
    write_yaml_detail: int = 1
    list_fields: bool = False
    ignore_unknown_config: bool = False
    show_version: bool = False
    show_help: bool = False


def format_help(exe: str = "gotm") -> str:
    return "\n".join(
        [
            f"Usage: {exe} [OPTIONS]",
            "",
            "Options:",
            "",
            "  -h, --help              print usage information and exit",
            "  -v, --version           print version information",
            (
                "  <yaml_file>             read configuration from file "
                "(default gotm.yaml)"
            ),
            (
                "  --ignore_unknown_config ignore unknown options encountered "
                "in configuration"
            ),
            "  -l, --list_variables    list all variables available for output",
            "  --output_id <string>    append to output file names - before extension",
            "  --write_yaml <file>     save yaml configuration to file",
            "  --detail <level>        settings to include in saved yaml file",
            "  --write_schema <file>   save configuration schema to file",
        ]
    )


def parse_cmdline(
    argv: Sequence[str] | None = None,
    *,
    exe: str = "gotm",
) -> CommandLineOptions:
    """Parse GOTM-style command-line arguments."""

    args = list(argv or ())
    options = CommandLineOptions(exe=exe)
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in ("-v", "--version"):
            return CommandLineOptions(**{**options.__dict__, "show_version": True})
        if arg in ("-h", "--help"):
            return CommandLineOptions(**{**options.__dict__, "show_help": True})
        if arg == "--write_yaml":
            index += 1
            if index >= len(args):
                raise ValueError("--write_yaml must be followed by a file path")
            options = CommandLineOptions(
                **{**options.__dict__, "write_yaml_path": args[index]}
            )
        elif arg == "--detail":
            index += 1
            if index >= len(args):
                raise ValueError("--detail must be followed by a detail level")
            detail_text = args[index]
            mapping = {
                "0": 0,
                "minimal": 0,
                "1": 1,
                "default": 1,
                "2": 2,
                "full": 2,
            }
            if detail_text not in mapping:
                raise ValueError(f"value {detail_text!r} for --detail not recognized")
            options = CommandLineOptions(
                **{**options.__dict__, "write_yaml_detail": mapping[detail_text]}
            )
        elif arg == "--write_schema":
            index += 1
            if index >= len(args):
                raise ValueError("--write_schema must be followed by a file path")
            options = CommandLineOptions(
                **{**options.__dict__, "write_schema_path": args[index]}
            )
        elif arg == "--output_id":
            index += 1
            if index >= len(args):
                raise ValueError("--output_id must be followed by a string")
            options = CommandLineOptions(
                **{**options.__dict__, "output_id": args[index]}
            )
        elif arg in ("-l", "--list_variables"):
            options = CommandLineOptions(**{**options.__dict__, "list_fields": True})
        elif arg == "--ignore_unknown_config":
            options = CommandLineOptions(
                **{**options.__dict__, "ignore_unknown_config": True}
            )
        else:
            if arg.startswith("--"):
                raise ValueError(f"command line option {arg} not recognized")
            yaml_path = Path(arg)
            if not yaml_path.exists():
                raise FileNotFoundError(
                    f"custom configuration file {arg} does not exist"
                )
            options = CommandLineOptions(**{**options.__dict__, "yaml_file": arg})
        index += 1
    return options
