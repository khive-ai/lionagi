#!/usr/bin/env python
# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Command-line interface for lionagi utilities.
"""

import argparse
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("lionagi-cli")


def main():
    """
    Main entry point for the lionagi command-line interface.
    """
    parser = argparse.ArgumentParser(
        description="lionagi command-line utilities",
        prog="lionagi",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        help="Command to run",
        required=True,
    )

    # Add the build-registry command
    build_registry_parser = subparsers.add_parser(
        "build-registry",
        help="Build the adapter registry",
        description="Build a static mapping of adapter keys to module paths.",
    )
    build_registry_parser.add_argument(
        "--adapters-dir",
        type=str,
        default=None,
        help="Path to the adapters directory (default: auto-detect)",
    )
    build_registry_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the output JSON file (default: adapter_map.json in the package directory)",
    )
    build_registry_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.command == "build-registry":
        from .build_adapter_registry import main as build_registry_main

        sys.argv = [sys.argv[0]] + sys.argv[
            2:
        ]  # Remove the 'build-registry' command
        build_registry_main()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
