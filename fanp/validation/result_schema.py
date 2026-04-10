from __future__ import annotations

import json
import os
from glob import glob


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _require_keys(obj: dict, keys: list[str], context: str) -> list[str]:
    errors: list[str] = []
    for key in keys:
        if key not in obj:
            errors.append(f"{context}: missing key '{key}'")
    return errors


def _require_repro(meta: dict, context: str) -> list[str]:
    errors: list[str] = []
    repro = meta.get("repro")
    if not isinstance(repro, dict):
        return [f"{context}: missing or invalid 'repro'"]

    errors.extend(_require_keys(repro, ["python_version", "torch_version", "seed", "cuda", "git_commit"], context))
    cuda = repro.get("cuda")
    if not isinstance(cuda, dict):
        errors.append(f"{context}: repro.cuda must be a dict")
    else:
        errors.extend(_require_keys(cuda, ["available", "torch_cuda_version", "device_count", "devices", "cudnn_version"], context))
    return errors


def validate_main_results_file(path: str) -> list[str]:
    errors: list[str] = []
    data = _read_json(path)

    for key in ["magnitude", "magnitude_ft", "fanp", "meta"]:
        if key not in data:
            errors.append(f"main: missing top-level key '{key}'")

    meta = data.get("meta", {})
    if isinstance(meta, dict):
        errors.extend(_require_keys(meta, ["run_id", "run_timestamp", "checkpoint", "config", "repro"], "main.meta"))
        errors.extend(_require_repro(meta, "main.meta"))
        run_id = meta.get("run_id")
        if isinstance(run_id, str):
            basename = os.path.basename(path)
            if run_id not in basename:
                errors.append(f"main: run_id '{run_id}' not found in filename '{basename}'")
    else:
        errors.append("main: meta must be a dict")

    for method in ["magnitude", "magnitude_ft", "fanp"]:
        block = data.get(method, {})
        if not isinstance(block, dict):
            errors.append(f"main.{method}: must be a dict")
            continue
        for sparsity, payload in block.items():
            if not isinstance(payload, dict):
                errors.append(f"main.{method}.{sparsity}: must be a dict")
                continue
            for key in ["test_acc", "sparsity"]:
                if key not in payload:
                    errors.append(f"main.{method}.{sparsity}: missing '{key}'")

            if method == "fanp":
                model_save_path = payload.get("model_save_path")
                run_id = meta.get("run_id") if isinstance(meta, dict) else None
                if isinstance(model_save_path, str):
                    if not os.path.isfile(model_save_path):
                        errors.append(f"main.fanp.{sparsity}: model_save_path not found: {model_save_path}")
                    if isinstance(run_id, str) and run_id not in os.path.basename(model_save_path):
                        errors.append(
                            f"main.fanp.{sparsity}: run_id '{run_id}' not in model_save_path basename"
                        )
                else:
                    errors.append(f"main.fanp.{sparsity}: missing or invalid 'model_save_path'")

    return errors


def validate_ablation_results_file(path: str) -> list[str]:
    errors: list[str] = []
    data = _read_json(path)

    if "meta" not in data:
        errors.append("ablation: missing top-level key 'meta'")
    if "results" not in data:
        errors.append("ablation: missing top-level key 'results'")

    meta = data.get("meta", {})
    if isinstance(meta, dict):
        errors.extend(_require_keys(meta, ["run_id", "run_timestamp", "checkpoint", "config", "repro"], "ablation.meta"))
        errors.extend(_require_repro(meta, "ablation.meta"))
        run_id = meta.get("run_id")
        if isinstance(run_id, str):
            basename = os.path.basename(path)
            if run_id not in basename:
                errors.append(f"ablation: run_id '{run_id}' not found in filename '{basename}'")
    else:
        errors.append("ablation: meta must be a dict")

    results = data.get("results", [])
    if not isinstance(results, list):
        errors.append("ablation.results: must be a list")
    else:
        for index, row in enumerate(results):
            if not isinstance(row, dict):
                errors.append(f"ablation.results[{index}]: must be a dict")
                continue
            for key in ["name", "sparsity", "test_acc"]:
                if key not in row:
                    errors.append(f"ablation.results[{index}]: missing '{key}'")

    return errors


def find_latest_result(results_dir: str, pattern: str) -> str | None:
    paths = glob(os.path.join(results_dir, pattern))
    if not paths:
        return None
    paths.sort(key=os.path.getmtime)
    return paths[-1]


def validate_latest_outputs(results_dir: str) -> list[str]:
    errors: list[str] = []

    latest_main = find_latest_result(results_dir, "main_*.json")
    if latest_main is None:
        errors.append(f"No main_*.json found in {results_dir}")
    else:
        errors.extend(validate_main_results_file(latest_main))

    latest_ablation = find_latest_result(results_dir, "ablation_*.json")
    if latest_ablation is None:
        errors.append(f"No ablation_*.json found in {results_dir}")
    else:
        errors.extend(validate_ablation_results_file(latest_ablation))

    return errors
