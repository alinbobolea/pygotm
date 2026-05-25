from __future__ import annotations

import json
from pathlib import Path

import click

from pygotm.citations.extractor import (
    BIB_PATH,
    citation_records,
    citations_for_config,
    citations_for_output,
)
from pygotm.driver import GotmDriver
from pygotm.errors import error_code_for_exception
from pygotm.gotm.print_version import collect_version_info, print_version
from pygotm.progress import ProgressReporter
from pygotm.schema import config_schema, netcdf_attrs_schema, output_schema
from pygotm.serve.daemon import serve_forever
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
@click.option(
    "--progress",
    "progress_mode",
    default="none",
    show_default=True,
    type=click.Choice(["none", "json", "plain"]),
    help="Emit run progress events to stderr.",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Show tracebacks instead of documented concise exit-code errors.",
)
def run_cli(
    config_path: Path,
    output_path: Path,
    max_steps: int | None,
    progress_mode: str,
    debug: bool,
) -> None:
    """Run one GOTM YAML configuration and write NetCDF output."""

    try:
        reporter = (
            None
            if progress_mode == "none"
            else ProgressReporter(
                stream=click.get_text_stream("stderr"), mode=progress_mode
            )
        )
        dataset = GotmDriver(config_path).run(
            max_steps=max_steps,
            output_path=output_path,
            progress=reporter,
        )
        try:
            click.echo(f"Wrote {output_path}")
        finally:
            dataset.close()
    except Exception as exc:
        if debug:
            raise
        code = error_code_for_exception(exc)
        click.echo(f"ERROR[{code}]: {exc}", err=True)
        raise SystemExit(code) from exc


@click.command(name="version")
@click.option("--json", "as_json", is_flag=True, help="Emit manifest-shaped JSON.")
def version_cli(as_json: bool) -> None:
    """Print pyGOTM runtime version information."""

    if as_json:
        click.echo(json.dumps(collect_version_info(), indent=2, sort_keys=True))
        return
    click.echo(print_version())


@click.group(name="schema")
def schema_cli() -> None:
    """Emit machine-readable config and output schemas."""


@schema_cli.command(name="config")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON Schema.")
def schema_config_cli(as_json: bool) -> None:
    """Emit the GOTM YAML configuration schema."""

    del as_json
    click.echo(json.dumps(config_schema(), indent=2, sort_keys=True))


@schema_cli.command(name="output")
@click.option(
    "--config", "config_path", type=click.Path(dir_okay=False, path_type=Path)
)
@click.option("--json", "as_json", is_flag=True, help="Emit JSON variable records.")
def schema_output_cli(config_path: Path | None, as_json: bool) -> None:
    """Emit output variable metadata."""

    del as_json
    click.echo(json.dumps(output_schema(config_path), indent=2, sort_keys=True))


@schema_cli.command(name="netcdf-attrs")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON attribute records.")
def schema_netcdf_attrs_cli(as_json: bool) -> None:
    """Emit the NetCDF global-attribute schema."""

    del as_json
    click.echo(json.dumps(netcdf_attrs_schema(), indent=2, sort_keys=True))


@click.command(name="cite")
@click.option("--all", "all_entries", is_flag=True, help="Emit all citations.")
@click.option(
    "--for-output",
    "output_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Emit citations for a NetCDF output.",
)
@click.option(
    "--for-config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Emit citations for a GOTM YAML config.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit JSON records.")
def cite_cli(
    all_entries: bool,
    output_path: Path | None,
    config_path: Path | None,
    as_json: bool,
) -> None:
    """Emit curated citations for pyGOTM or one active run."""

    selected = sum(value is not None or False for value in (output_path, config_path))
    selected += 1 if all_entries else 0
    if selected != 1:
        raise click.UsageError(
            "choose exactly one of --all, --for-output, --for-config"
        )

    if all_entries:
        if as_json:
            click.echo(json.dumps(citation_records(), indent=2, sort_keys=True))
        else:
            click.echo(BIB_PATH.read_text(encoding="utf-8"))
        return

    if output_path is not None:
        records = citations_for_output(output_path)
    else:
        assert config_path is not None
        records = citations_for_config(config_path)
    if as_json:
        click.echo(json.dumps(records, indent=2, sort_keys=True))
        return
    keys = records["citation_keys"]
    if isinstance(keys, list):
        click.echo("\n".join(str(key) for key in keys))


@click.command(name="serve")
@click.option(
    "--no-warmup",
    "skip_warmup",
    is_flag=True,
    help="Start without the self-contained Numba warmup.",
)
def serve_cli(skip_warmup: bool) -> None:
    """Serve stdin/stdout newline-delimited JSON-RPC requests."""

    raise SystemExit(serve_forever(do_warmup=not skip_warmup))


cli.add_command(run_cli)
cli.add_command(version_cli)
cli.add_command(schema_cli)
cli.add_command(cite_cli)
cli.add_command(serve_cli)
cli.add_command(validate_cli, name="validate")


if __name__ == "__main__":
    cli()
