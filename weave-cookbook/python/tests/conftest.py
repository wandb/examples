import importlib.util
import sys
from pathlib import Path
from types import ModuleType

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))


def load_example(filename: str) -> ModuleType:
    """Import a numbered example file that cannot be imported by module name."""
    path = EXAMPLES_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
