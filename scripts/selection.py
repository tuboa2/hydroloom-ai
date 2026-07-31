from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.wqi_predictor.pipeline.selection import run_selection

if __name__ == "__main__":
    result = run_selection()
    print("Phase 3 complete.")
    print(result)
    