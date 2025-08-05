"""Command-line interface."""

import typer


app: typer.Typer = typer.Typer()


@app.command(name="tlopo-toolkit")
def main() -> None:
    """TLOPO Toolkit."""


if __name__ == "__main__":
    app()  # pragma: no cover
