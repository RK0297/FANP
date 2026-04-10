"""
Phase P1 smoke test.

Validates:
- run_id-safe naming in result artifacts
- reproducibility metadata presence
- expected schema for latest main and ablation outputs

Run from fanp/ directory:
    python test_phase_p1.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from validation.result_schema import validate_latest_outputs


def main() -> int:
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    errors = validate_latest_outputs(results_dir)

    print("=" * 60)
    print("P1 Validation: result schema + artifact naming")
    print("=" * 60)

    if errors:
        print(f"FAILED ({len(errors)} issues)")
        for issue in errors:
            print(f"  - {issue}")
        return 1

    print("PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
