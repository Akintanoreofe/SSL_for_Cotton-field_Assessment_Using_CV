"""Console script for lejepa_cotton_field_assessment."""

import typer
from rich.console import Console

from lejepa_cotton_field_assessment import utils

app = typer.Typer()
console = Console()


@app.command()
def main() -> None:
    """Console script for lejepa_cotton_field_assessment."""
    console.print("Replace this message by putting your code into lejepa_cotton_field_assessment.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    utils.do_something_useful()


if __name__ == "__main__":
    app()
