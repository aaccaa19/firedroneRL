"""Thin wrapper exposing the original script as a module entrypoint.
This file imports the original script's symbols and exposes a `main()` function.
"""
# Import the original top-level script
from pathlib import Path
import runpy
import os

script_path = Path(__file__).resolve().parent.parent / 'drl-based-drone' / 'UOAR_drone_rl.py'
# Run the script and import needed symbols into this module's namespace
_mod = runpy.run_path(str(script_path))

# re-export key functions/classes
DroneEnv = _mod.get('DroneEnv')
TD3Agent = _mod.get('TD3Agent')
main = _mod.get('main')

if DroneEnv is None or TD3Agent is None or main is None:
    raise ImportError('Failed to import core symbols from UOAR_drone_rl.py')
