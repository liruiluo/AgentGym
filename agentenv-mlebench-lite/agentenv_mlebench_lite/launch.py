from __future__ import annotations

import argparse
import ipaddress
from pathlib import Path

import uvicorn

from .config import build_manager, load_runtime_config
from .server import create_app


def launch() -> None:
    parser = argparse.ArgumentParser(description="Launch MLE-bench Lite adapter")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9017)
    arguments = parser.parse_args()
    try:
        loopback = (
            arguments.host == "localhost"
            or ipaddress.ip_address(arguments.host).is_loopback
        )
    except ValueError:
        loopback = False
    if not loopback:
        parser.error("--host must bind a loopback address")
    config = load_runtime_config(arguments.config)
    manager = build_manager(config)
    uvicorn.run(create_app(manager), host=arguments.host, port=arguments.port)


if __name__ == "__main__":
    launch()
