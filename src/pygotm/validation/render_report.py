"""Re-render the HTML report from a saved results.json without re-running pyGOTM.

Usage
-----
    python validation/render_report.py
    python validation/render_report.py validation/results.json
    python validation/render_report.py results.json --output my_report.html
"""

from __future__ import annotations

from pathlib import Path

import click

from pygotm.validation.report import load_json, render_html


@click.command()
@click.argument(
    "results_json",
    default=None,
    required=False,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output", "output_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output HTML path (default: same dir as results.json, named report.html).",
)
def cli(results_json: Path | None, output_path: Path | None) -> None:
    """Render pyGOTM validation HTML report from results JSON."""
    if results_json is None:
        results_json = Path("validation/results.json")
        if not results_json.exists():
            raise click.ClickException(
                "No validation/results.json found. Run validation first."
            )

    report = load_json(results_json)
    html = render_html(report)

    if output_path is None:
        output_path = results_json.parent / "report.html"

    output_path.write_text(html, encoding="utf-8")
    click.echo(f"Report written to: {output_path}")


if __name__ == "__main__":
    cli()
