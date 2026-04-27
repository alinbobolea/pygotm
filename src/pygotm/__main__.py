from __future__ import annotations

import click

from pygotm.validate import cli as validate_cli


@click.group()
def cli() -> None:
    """pyGOTM command-line interface."""


cli.add_command(validate_cli)


if __name__ == "__main__":
    cli()
