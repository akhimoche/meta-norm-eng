"""
Dynamic norm loader - automatically discovers and loads norms from the norms directory.

This module scans utils/norms/ for Python files, finds classes that inherit from Norm,
and provides a clean interface for loading them by name.
"""

import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import Dict, Type, Optional

from .norm import Norm


def _discover_norms() -> Dict[str, Type[Norm]]:
    """
    Auto-discover all Norm subclasses in the norms directory.
    
    Returns:
        Dictionary mapping norm names (file names) to their Norm classes
    """
    norms_dir = Path(__file__).parent
    norm_registry = {}
    
    # Scan all Python files in the norms directory
    for py_file in norms_dir.glob("*.py"):
        # Skip special files
        if py_file.name in ["__init__.py", "norm.py", "loader.py"]:
            continue
        
        # Import the module dynamically using file path (handles dots in filenames)
        try:
            # Use importlib.util to load from file path (works with any filename)
            spec = importlib.util.spec_from_file_location(
                f"norm_{py_file.stem}",  # Create a unique module name
                py_file
            )
            if spec is None or spec.loader is None:
                print(f"Warning: Could not create spec for {py_file.name}")
                continue
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find all classes in the module that inherit from Norm
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Check if it's a Norm subclass (but not Norm itself)
                if issubclass(obj, Norm) and obj is not Norm:
                    # Use the file name (stem) as the key
                    norm_registry[py_file.stem] = obj
                    break  # Only take the first Norm subclass per file
                    
        except Exception as e:
            print(f"Warning: Could not load norm from {py_file.name}: {e}")
    
    return norm_registry


# Build the registry at module load time
_NORM_REGISTRY = _discover_norms()


def list_available_norms() -> list[str]:
    """
    Get a list of all available norm names.
    
    Returns:
        List of norm names (file names without .py extension)
    """
    return sorted(_NORM_REGISTRY.keys())


def get_norm(norm_type: str, epsilon: float) -> Optional[Norm]:
    """
    Load and instantiate a norm by name.
    
    Args:
        norm_type: Name of the norm to load (matches the file name, e.g., "gpt5", "claude")
                   Use "None" or None for no norm
        epsilon: Epsilon value for norm compliance (0.0 = always obey, 1.0 = always ignore)
    
    Returns:
        Instantiated Norm object, or None if norm_type is "None"/None
        
    Raises:
        ValueError: If norm_type is invalid
    """
    # Handle no norm case
    if norm_type is None or norm_type == "None":
        return None
    
    # Check if the requested norm exists
    if norm_type not in _NORM_REGISTRY:
        available = ", ".join(list_available_norms())
        raise ValueError(
            f"Unknown norm type: '{norm_type}'\n"
            f"Available norms: {available}\n"
            f"Or use 'None' for no norm."
        )
    
    # Instantiate and return the norm
    NormClass = _NORM_REGISTRY[norm_type]
    return NormClass(epsilon=epsilon)


def print_available_norms():
    """Print all available norms in a user-friendly format."""
    available_norms = list_available_norms()
    
    print("\n" + "="*50)
    print("AVAILABLE NORMS:")
    print("="*50)
    
    for norm_name in available_norms:
        norm_class = _NORM_REGISTRY[norm_name]
        print(f"  • {norm_name:25} → {norm_class.__name__}")
    
    print(f"  • {'None':25} → No norm (baseline)")
    print("="*50 + "\n")


# Optional: Print available norms when this module is run directly
if __name__ == "__main__":
    print_available_norms()

