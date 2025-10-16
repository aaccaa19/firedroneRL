"""Console entry point for the package."""
import sys
import os
# Force non-interactive matplotlib backend to avoid Tcl/Tk dependency in venvs
os.environ.setdefault('MPLBACKEND', 'Agg')
from .app import main

def run():
    # simply call the script's main function
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print('Error running firedrone-rl:', e)
        sys.exit(2)
