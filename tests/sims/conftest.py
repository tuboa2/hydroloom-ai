from __future__ import annotations

import sys
from pathlib import Path

# Add project root and scripts directory to sys.path
root_dir = Path(__file__).resolve().parents[2]
scripts_dir = root_dir / "scripts"

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))
