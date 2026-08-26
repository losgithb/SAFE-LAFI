"""Safety-aware residual-field inference for ocular artifacts."""

from .apply import apply
from .fit import fit_fold
from .state import load_state

__all__ = ["apply", "fit_fold", "load_state"]
__version__ = "1.0.0"
