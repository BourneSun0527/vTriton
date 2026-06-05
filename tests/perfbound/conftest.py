"""pytest configuration for perfbound golden-number tests."""
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `from perfbound import ...` works
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
