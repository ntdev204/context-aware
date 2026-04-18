#!/usr/bin/env python3
"""Generate Python Protobuf stubs from .proto files.

Usage (Windows PowerShell):
    python scripts/generate_proto.py

Output:
    src/communication/proto/messages_pb2.py
    src/communication/proto/messages_pb2_grpc.py
    src/communication/proto/training_service_pb2.py
    src/communication/proto/training_service_pb2_grpc.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import importlib.util

if importlib.util.find_spec("pkg_resources") is None:
    print("ERROR: 'pkg_resources' module is missing.")
    print("   This is required by grpcio-tools to generate protos.")
    print("   Please run: pip install setuptools")
    sys.exit(1)

ROOT = Path(__file__).parent.parent
PROTO_DIR = ROOT / "proto"
OUT_DIR = ROOT / "src" / "communication" / "proto"
PROTOS = ["messages.proto", "training_service.proto"]


def generate() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Create __init__.py so the package is importable
    init = OUT_DIR / "__init__.py"
    if not init.exists():
        init.write_text('"""Generated Protobuf stubs."""\n')

    for proto in PROTOS:
        proto_path = PROTO_DIR / proto
        if not proto_path.exists():
            print(f"  Not found: {proto_path}")
            continue

        cmd = [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{PROTO_DIR}",
            f"--python_out={OUT_DIR}",
            f"--grpc_python_out={OUT_DIR}",
            str(proto_path),
        ]
        print(f"Generating {proto}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR:\n{result.stderr}")
        else:
            print("  Done")

    print(f"\nStubs written to: {OUT_DIR}")
    print('Next: verify with  python -c "from src.communication.proto import messages_pb2"')


if __name__ == "__main__":
    generate()
