"""
Aggregate experiment outputs into a one-page status report.

Run from fanp/ directory:
    python results_summary.py
"""

from __future__ import annotations

import json
import os
from glob import glob
from typing import Any


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _latest(paths: list[str]) -> str | None:
    if not paths:
        return None
    paths.sort(key=os.path.getmtime)
    return paths[-1]


def summarize_main_runs(paths: list[str]) -> dict:
    summary = {
        "count": len(paths),
        "latest": None,
        "best_dense": None,
        "best_fanp_90": None,
        "latest_table": [],
    }

    best_dense = None
    best_fanp_90 = None

    latest_path = _latest(paths)
    summary["latest"] = latest_path

    for path in paths:
        data = _read_json(path)
        meta = data.get("meta", {}) if isinstance(data, dict) else {}

        dense_acc = _safe_float(meta.get("dense_test_acc")) if isinstance(meta, dict) else None
        if dense_acc is not None:
            if best_dense is None or dense_acc > best_dense["dense_test_acc"]:
                best_dense = {"dense_test_acc": dense_acc, "path": path}

        fanp = data.get("fanp", {}) if isinstance(data, dict) else {}
        if isinstance(fanp, dict) and "0.9" in fanp and isinstance(fanp["0.9"], dict):
            fanp_90 = _safe_float(fanp["0.9"].get("test_acc"))
            if fanp_90 is not None:
                if best_fanp_90 is None or fanp_90 > best_fanp_90["fanp_90_acc"]:
                    best_fanp_90 = {"fanp_90_acc": fanp_90, "path": path}

    summary["best_dense"] = best_dense
    summary["best_fanp_90"] = best_fanp_90

    if latest_path is not None:
        latest = _read_json(latest_path)
        for sp_key in ["0.3", "0.5", "0.7", "0.9"]:
            row = {"sparsity": sp_key}
            for method in ["magnitude", "magnitude_ft", "fanp"]:
                block = latest.get(method, {}) if isinstance(latest, dict) else {}
                if isinstance(block, dict) and isinstance(block.get(sp_key), dict):
                    row[method] = _safe_float(block[sp_key].get("test_acc"))
                else:
                    row[method] = None
            summary["latest_table"].append(row)

    return summary


def _extract_ablation_rows(data: Any) -> list[dict]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return [row for row in data["results"] if isinstance(row, dict)]
    return []


def summarize_ablation_runs(paths: list[str]) -> dict:
    summary = {
        "count": len(paths),
        "latest": _latest(paths),
        "best_method_latest": None,
    }

    latest = summary["latest"]
    if latest is None:
        return summary

    rows = _extract_ablation_rows(_read_json(latest))
    best = None
    for row in rows:
        name = row.get("name")
        acc = _safe_float(row.get("test_acc"))
        if isinstance(name, str) and acc is not None:
            if best is None or acc > best["test_acc"]:
                best = {"name": name, "test_acc": acc}
    summary["best_method_latest"] = best
    return summary


def build_report(main_summary: dict, ablation_summary: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 74)
    lines.append("FANP RESULTS SUMMARY (AUTO-GENERATED)")
    lines.append("=" * 74)
    lines.append("")

    lines.append("[Main Experiment Runs]")
    lines.append(f"- Total main runs found: {main_summary['count']}")
    lines.append(f"- Latest main run: {os.path.basename(main_summary['latest']) if main_summary['latest'] else 'N/A'}")

    best_dense = main_summary.get("best_dense")
    if best_dense:
        lines.append(f"- Best dense test_acc: {best_dense['dense_test_acc']:.2f}% ({os.path.basename(best_dense['path'])})")

    best_fanp_90 = main_summary.get("best_fanp_90")
    if best_fanp_90:
        lines.append(f"- Best FANP @90% sparsity: {best_fanp_90['fanp_90_acc']:.2f}% ({os.path.basename(best_fanp_90['path'])})")

    lines.append("")
    lines.append("- Latest run comparison table:")
    lines.append("  Sparsity | Magnitude | Mag+FT | FANP")
    lines.append("  -------- | --------- | ------ | ----")
    for row in main_summary.get("latest_table", []):
        sp = row["sparsity"]
        mag = "N/A" if row.get("magnitude") is None else f"{row['magnitude']:.2f}%"
        mft = "N/A" if row.get("magnitude_ft") is None else f"{row['magnitude_ft']:.2f}%"
        fanp = "N/A" if row.get("fanp") is None else f"{row['fanp']:.2f}%"
        lines.append(f"  {sp:>8} | {mag:>9} | {mft:>6} | {fanp:>6}")

    lines.append("")
    lines.append("[Ablation Runs]")
    lines.append(f"- Total ablation runs found: {ablation_summary['count']}")
    lines.append(f"- Latest ablation run: {os.path.basename(ablation_summary['latest']) if ablation_summary['latest'] else 'N/A'}")

    best_method = ablation_summary.get("best_method_latest")
    if best_method:
        lines.append(
            f"- Best method in latest ablation: {best_method['name']} ({best_method['test_acc']:.2f}%)"
        )

    lines.append("")
    lines.append("[Project Readiness Snapshot]")
    lines.append("- Core research pipeline: complete")
    lines.append("- P0 production hardening: complete")
    lines.append("- P1 validation guards: in progress (schema + CI added)")
    lines.append("- Next: CI quick-run generation + report dashboards")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    main_paths = sorted(glob(os.path.join(RESULTS_DIR, "main_*.json")))
    ablation_paths = sorted(glob(os.path.join(RESULTS_DIR, "ablation_*.json")))

    main_summary = summarize_main_runs(main_paths)
    ablation_summary = summarize_ablation_runs(ablation_paths)

    report = build_report(main_summary, ablation_summary)

    report_path = os.path.join(RESULTS_DIR, "results_summary_latest.txt")
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report + "\n")

    print(report)
    print(f"\nSaved summary: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
