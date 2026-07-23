from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.wqi_predictor.pipeline.preprocess import run

if __name__ == "__main__":
    result = run()
    print("Phase 1 complete.")
    print(result)
