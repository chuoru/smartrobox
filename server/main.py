#!/usr/bin/env python3
##
# @file main.py
#
# @brief Provide entry point to run maintenance ui
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2025/07/21.
#
# Copyright (c) 2025 HACHIX.  All rights reserved.

# Standard library
import sys
import argparse

# External library
import uvicorn

# Internal library
from app.app import RestServer


if __name__ == "__main__":
    """! Main entry point to run the REST server."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "This is a meta argument parser for the train switching between"
            " different policies and environments. The actual arguments are"
            " handled by another internal argument parser."
        ),
        add_help=False,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the REST server on.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to run the REST server on.",
    )
    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        help="Show this help message and continue",
    )
    args = parser.parse_args()
    if args.help:
        parser.print_help()
        print("\n================================\n")
        sys.argv += ["--help"]
        sys.exit(0)
    rest_server = RestServer()
    uvicorn.run(rest_server.app, host=args.host, port=args.port)
