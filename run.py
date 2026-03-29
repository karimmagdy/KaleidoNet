#!/usr/bin/env python3
"""Launcher: adds project root to sys.path and runs the target script."""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Run the target
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py <script.py> [args...]")
        sys.exit(1)
    script = sys.argv[1]
    sys.argv = sys.argv[1:]  # shift argv so the script sees correct args
    with open(script) as f:
        code = compile(f.read(), script, "exec")
        exec(code, {"__name__": "__main__", "__file__": script})
