from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..models.common import ensure_dir, write_json

logger = logging.getLogger(__name__)


def _flatten_dict(
    payload: Mapping[str, Any], parent_key: str = "", sep: str = "."
) -> dict[str, Any]:
    items: list[tuple[str, Any]] = []

    for key, value in payload.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)

        if isinstance(value, Mapping):
            items.extend(_flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))

    return dict(items)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"

    return str(value)


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines: list[str] = []

    lines.append("# Model Candidate Report Card")
    lines.append("")

    candidate = report.get("candidate", {})
    if candidate:
        lines.append("## Candidate")
        lines.append("")
        for key, value in candidate.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")

    baseline = report.get("baseline", {})
    if baseline:
        lines.append("## Baseline Comparison")
        lines.append("")
        for key, value in baseline.items():
            lines.append(f"- **{key}**: {_format_value(value)}")
        lines.append("")

    metrics = report.get("metrics", {})
    if metrics:
        lines.append("## Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        for key, value in _flatten_dict(metrics).items():
            lines.append(f"| {key} | {_format_value(value)} |")
        lines.append("")

    features = report.get("features", [])
    if features:
        lines.append("## Frozen Feature Contract")
        lines.append("")
        for feature in features:
            lines.append(f"- {feature}")
        lines.append("")

    lines.append("## Gate Status")
    lines.append("")
    lines.append(f"- passed_baseline_gate: {report.get('passed_baseline_gate', False)}")
    lines.append(f"- leakage_detected: {report.get('leakage_detected', False)}")
    lines.append("")

    return "\n".join(lines)


def write_report_card(
    output_dir: str | Path,
    candidate_meta: Mapping[str, Any],
    metrics: Mapping[str, Any],
    baseline: Mapping[str, Any],
    feature_columns: Sequence[str],
    passed_baseline_gate: bool,
    leakage_detected: bool = False,
) -> dict[str, Any]:
    resolved_dir = Path(ensure_dir(output_dir) or output_dir)

    features_list = list(feature_columns)

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "candidate": dict(candidate_meta or {}),
        "baseline": dict(baseline or {}),
        "metrics": dict(metrics or {}),
        "features": features_list,
        "feature_count": len(features_list),
        "passed_baseline_gate": bool(passed_baseline_gate),
        "leakage_detected": bool(leakage_detected),
    }

    json_path = resolved_dir / "report_card.json"
    md_path = resolved_dir / "report_card.md"

    write_json(json_path, report)
    md_path.write_text(_markdown_report(report), encoding="utf-8")

    logger.info("Report card written to %s and %s", json_path, md_path)

    return report
