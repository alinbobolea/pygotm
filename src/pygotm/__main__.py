from __future__ import annotations

from pathlib import Path

import click

from pygotm.driver import GotmDriver
from pygotm.validation.run_validation import cli as validate_cli


@click.group()
def cli() -> None:
    """pyGOTM command-line interface."""


@click.command(name="run")
@click.argument(
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output",
    "output_path",
    "-o",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="NetCDF file to write.",
)
@click.option("--max-steps", default=None, type=int, help="Limit integration steps.")
def run_cli(
    config_path: Path,
    output_path: Path,
    max_steps: int | None,
) -> None:
    """Run one GOTM YAML configuration and write NetCDF output."""

    dataset = GotmDriver(config_path).run(max_steps=max_steps, output_path=output_path)
    try:
        click.echo(f"Wrote {output_path}")
    finally:
        dataset.close()


cli.add_command(run_cli)
cli.add_command(validate_cli, name="validate")


if __name__ == "__main__":
    cli()
