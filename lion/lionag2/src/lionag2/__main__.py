"""Command-line interface."""

import click


@click.command()
@click.version_option()
def main() -> None:
    """Lionag2."""


if __name__ == "__main__":
    main(prog_name="lionag2")  # pragma: no cover
