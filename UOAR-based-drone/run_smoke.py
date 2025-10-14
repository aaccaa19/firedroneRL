# Helper to run UOAR_drone_rl.py with SMOKE_RUN env var set, avoiding shell quoting issues
import os
import runpy
import sys

# Force Matplotlib to use non-interactive backend to avoid Tk/Tcl errors
os.environ.setdefault('MPLBACKEND', 'Agg')

# Number of smoke episodes; adjust if needed
SMOKE_RUN = os.environ.get('SMOKE_RUN', '100')
os.environ['SMOKE_RUN'] = SMOKE_RUN

script_path = os.path.join(os.path.dirname(__file__), 'UOAR_drone_rl.py')
print(f"Running {script_path} with SMOKE_RUN={SMOKE_RUN}")
runpy.run_path(script_path, run_name='__main__')
