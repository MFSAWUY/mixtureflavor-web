"""Independent web application using a local, unchanged model source snapshot."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor" / "flavorgnn"
sys.path.insert(0, str(VENDOR))
__version__ = "1.0.0"
