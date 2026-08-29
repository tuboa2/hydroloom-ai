from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .types import (
    GateViolationError,
    Hemisphere,
    WQIState,
)

ORDERED_STATES: tuple[WQIState, ...] = (
    WQIState.INITIALIZED,
    WQIState.INPUTS_VERIFIED,
    WQIState.CANDIDATES_FROZEN,
    WQIState.OOF_GENERATED,
    WQIState.ENSEMBLES_EVALUATED,
    WQIState.ENSEMBLE_FROZEN,
    WQIState.RESIDUAL_EVALUATED,
    WQIState.POSTPROCESSING_FROZEN,
    WQIState.FINAL_CONFIG_FROZEN,
    WQIState.REFIT_COMPLETE,
    WQIState.TEST_EVALUATED,
    WQIState.REPORTS_COMPLETE,
    WQIState.REGISTERED,
)


class Phase5StateMachine:
    """Strict forward-only state machine managing Phase 5 execution milestones."""

    def __init__(self, hemisphere: Hemisphere, state_file: Path) -> None:
        self.hemisphere = hemisphere
        self.state_file = state_file
        self.current_state = WQIState.INITIALIZED
        self.history: list[dict[str, Any]] = []
        self._load_or_init()

    def _load_or_init(self) -> None:
        if self.state_file.exists():
            try:
                with open(self.state_file, encoding="utf-8") as f:
                    data = json.load(f)
                self.current_state = WQIState(data["current_state"])
                self.history = data.get("history", [])
            except Exception as exc:
                raise GateViolationError(f"Corrupt state file at {self.state_file}: {exc}") from exc
        else:
            self._save()

    def _save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "hemisphere": self.hemisphere.value,
            "current_state": self.current_state.value,
            "history": self.history,
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def transition_to(self, next_state: WQIState, context: Mapping[str, Any] | None = None) -> None:
        """Validate and commit a forward-only transition."""
        cur_idx = ORDERED_STATES.index(self.current_state)
        next_idx = ORDERED_STATES.index(next_state)

        # Backward transition lock
        if next_idx <= cur_idx:
            raise GateViolationError(
                f"Illegal backward transition from {self.current_state.value} to {next_state.value}. "
                "Phase 5 is strictly forward-only."
            )

        # Permanent lock past TEST_EVALUATED
        if cur_idx >= ORDERED_STATES.index(WQIState.TEST_EVALUATED) and next_idx < cur_idx:
            raise GateViolationError(
                "TEST_EVALUATED milestone is permanently locked. No parameters may be retuned."
            )

        self.history.append(
            {
                "from_state": self.current_state.value,
                "to_state": next_state.value,
                "context": dict(context or {}),
            }
        )
        self.current_state = next_state
        self._save()
