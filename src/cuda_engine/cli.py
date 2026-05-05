import typer

app = typer.Typer(help="CUDA synthesis engine CLI.")


@app.callback()
def main() -> None:
    """Command-line entry point placeholder for M0."""
