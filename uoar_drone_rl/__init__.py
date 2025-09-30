"""UOAR Drone RL package
Lightweight lazy wrapper around the original script. We delay running the
large top-level script until the user actually calls into `main` or needs
classes. This avoids importing matplotlib backends (Tcl/Tk) at package
import time, which can fail in minimal virtualenvs on Windows.
"""
from pathlib import Path
import runpy
import threading

_MOD = None
_MOD_LOCK = threading.Lock()

def _load_mod():
	global _MOD
	if _MOD is None:
		with _MOD_LOCK:
			if _MOD is None:
				script_path = Path(__file__).resolve().parent.parent / 'UOAR-based-drone' / 'UOAR_drone_rl.py'
				_MOD = runpy.run_path(str(script_path))
	return _MOD

def _get_attr(name):
	mod = _load_mod()
	val = mod.get(name)
	if val is None:
		raise ImportError(f"'{name}' not found in UOAR_drone_rl.py")
	return val

def main(*args, **kwargs):
	return _get_attr('main')(*args, **kwargs)

class _LazyClass:
	def __init__(self, name):
		self._name = name
	def __call__(self, *args, **kwargs):
		cls = _get_attr(self._name)
		return cls(*args, **kwargs)
	def __getattr__(self, item):
		cls = _get_attr(self._name)
		return getattr(cls, item)

DroneEnv = _LazyClass('DroneEnv')
TD3Agent = _LazyClass('TD3Agent')

__all__ = ['main', 'DroneEnv', 'TD3Agent']
