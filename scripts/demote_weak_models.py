"""
demote_weak_models.py — One-shot hygiene script.
=================================================
Walks MODEL_DIR and FORCE-DEMOTES any legacy promoted metadata file whose
promoted accuracy falls BELOW the MIN_HARD_ACCURACY_FLOOR (default 0.60 = 60%).

Also reports every file that is already below-floor (whether promoted or not).

Rationale:
  The bot-side hard filter in model_status.py (get_best_timeframe_for_symbol)
  ALREADY refuses to use < 60% models as the "active" pick. But the files
  on disk still advertise promoted=True, which confuses audits and retraining
  baseline comparisons (promoted baseline should always represent a model
  we're actually willing to run in prod).

Usage:
  python scripts/demote_weak_models.py               # DRY RUN — show plan, don't write
  python scripts/demote_weak_models.py --apply       # ACTUALLY WRITE demoted=True + promoted=False
  python scripts/demote_weak_models.py --floor 0.63  # Use a custom floor (default 0.60)
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import MODEL_DIR, AI_CONFIG
from ai_engine.training.model_status import (
    _demote_model_metadata,
    MIN_HARD_ACCURACY_FLOOR,
)


def _read_json(p: Path) -> Dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _extract_accuracy(d: Dict, name: str) -> float:
    if not isinstance(d, dict):
        return 0.0
    if "accuracy" in d:
        try:
            return float(d["accuracy"])
        except Exception:
            pass
    if "validation" in d and isinstance(d["validation"], dict):
        v = d["validation"]
        for k in ("candidate_model_accuracy", "best_candidate_accuracy", "accuracy"):
            if k in v:
                try:
                    return float(v[k])
                except Exception:
                    continue
    return 0.0


def scan() -> Tuple[List[Tuple[Path, Dict, float]], List[Tuple[Path, Dict, float]]]:
    """Return (needs_demotion_list, already_below_list)."""
    need: List[Tuple[Path, Dict, float]] = []
    below: List[Tuple[Path, Dict, float]] = []
    if not MODEL_DIR.exists():
        return need, below
    for p in sorted(MODEL_DIR.glob("*.metadata.json")):
        d = _read_json(p)
        acc = _extract_accuracy(d, p.name)
        promoted = bool(d.get("promoted", False))
        if acc < MIN_HARD_ACCURACY_FLOOR:
            below.append((p, d, acc))
            if promoted:
                need.append((p, d, acc))
    return need, below


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demote legacy models whose promoted accuracy is below the 60% hard floor",
    )
    parser.add_argument("--apply", action="store_true", help="Actually write demotions (default: dry run)")
    parser.add_argument(
        "--floor",
        type=float,
        default=None,
        help=f"Override accuracy floor (default: {MIN_HARD_ACCURACY_FLOOR:.4f} from AI_CONFIG.min_model_accuracy)",
    )
    args = parser.parse_args()

    if args.floor is not None:
        # Override at module level so _demote_model_metadata logs match user expectation
        MIN_HARD_ACCURACY_FLOOR_override = float(args.floor)
    else:
        MIN_HARD_ACCURACY_FLOOR_override = MIN_HARD_ACCURACY_FLOOR

    print(f"[demote_weak_models] floor = {MIN_HARD_ACCURACY_FLOOR_override * 100:.1f}%")
    print(f"[demote_weak_models] MODEL_DIR = {MODEL_DIR}")
    print(f"[demote_weak_models] mode = {'APPLY (writing files)' if args.apply else 'DRY RUN (no writes)'}")
    print("-" * 90)

    need_demote, below_floor = scan()

    # Report ALL below-floor files first for audit visibility
    if below_floor:
        print(f"\nALL BELOW-FLOOR files (N={len(below_floor)}):")
        print(f"  {'Filename':60s}  {'Promoted':8s}  {'Accuracy':10s}")
        for p, d, acc in below_floor:
            print(f"  {p.name:60s}  {str(bool(d.get('promoted',False))):8s}  {acc*100:8.2f}%")
    else:
        print("\nNo below-floor metadata files found at all. Nice.")

    if not need_demote:
        print(f"\nDONE. No promoted+below-floor files to demote.")
        sys.exit(0)

    print(f"\nPROMOTED + BELOW-FLOOR -> will be demoted (N={len(need_demote)}):")
    print(f"  {'Filename':60s}  {'Accuracy':10s}")
    for p, d, acc in need_demote:
        print(f"  {p.name:60s}  {acc*100:8.2f}%  -> promoted=False, demoted=True")

    if not args.apply:
        print(f"\nDRY RUN complete. Re-run with --apply to write {len(need_demote)} demotions.")
        sys.exit(0)

    success = 0
    failed = 0
    for p, d, acc in need_demote:
        reason = f"promoted_accuracy_{int(acc*10000)/10000:.4f}_below_{MIN_HARD_ACCURACY_FLOOR_override:.4f}_hard_floor"
        try:
            if args.floor is not None:
                # Do it inline instead of through _demote_model_metadata to respect custom floor
                dd = dict(d)
                dd["promoted"] = False
                dd["demoted"] = True
                dd["demotion_reason"] = reason
                dd["demoted_at"] = __import__("datetime").datetime.utcnow().isoformat()
                dd["demotion_floor"] = MIN_HARD_ACCURACY_FLOOR_override
                p.write_text(json.dumps(dd, indent=2), encoding="utf-8")
                ok = True
            else:
                ok = _demote_model_metadata(p, reason)
            if ok:
                success += 1
                print(f"  ✓ demoted: {p.name}")
            else:
                failed += 1
                print(f"  ✗ FAILED: {p.name}")
        except Exception as e:
            failed += 1
            print(f"  ✗ EXCEPTION {p.name}: {e}")

    print(f"\nAPPLY complete. Success={success} Failed={failed} Total={len(need_demote)}.")
    if failed:
        sys.exit(2)
