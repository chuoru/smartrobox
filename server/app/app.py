#!/usr/bin/env python3
##
# @file app.py
#
# @brief Provide the main application for the SmartOne Techman module.
#
# Copyright (c) 2025 HACHIX.  All rights reserved.

# Standard library
import os
import mimetypes

mimetypes.init()
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("image/svg+xml", ".svg")

# Internal library
from app.config import Config
from app.socket import websocket_endpoint


# External library
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


class RestServer:
    """! RESTful API server for the input_controller module."""

    current_folder = os.path.dirname(os.path.abspath(__file__))
    parent_folder = os.path.dirname(current_folder)
    dist_folder = os.path.join(parent_folder, "user_interface")
    _server_config = Config(os.path.join(parent_folder, "config.yaml"))
    project_folder = os.path.join(parent_folder, "projects", _server_config.get("projects"))

    def __init__(self):
        """! Initialize the REST server with the given configuration."""
        self.app = FastAPI()

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"], 
        )

        self.app.websocket("/ws")(websocket_endpoint)

        @self.app.get("/health")
        def health_check():
            return {}, 200
