"""Evaluate a trained checkpoint on one processed split.

Example:
    python scripts/evaluate.py --checkpoint experiments/checkpoints/lstm_direct_C_s0/best.pt --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.evaluator import evaluate_trajectories  # noqa: E402
from src.training.inference import predict_split  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate one checkpoint.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args(argv)

    pred = predict_split(args.checkpoint,
                         str(Path(args.processed_dir) / f"{args.split}.npz"),
                         device=args.device, batch_size=args.batch_size)
    res = evaluate_trajectories(pred["positions"], pred["targets"],
                                pred["sid"], pred["sample_id"])
    print(f"{args.checkpoint} [{args.split}] n={res['n_windows']} storms={res['n_storms']}")
    print("MTE:", {k: round(v, 1) for k, v in res["mte_km"].items()})
    print(f"ADE={res['ade_km']:.1f} km  FDE={res['fde_km']:.1f} km")
    if args.out:
        out = {"checkpoint": args.checkpoint, "split": args.split,
               "mte_km": res["mte_km"], "mte_full_km": res["mte_full_km"],
               "median_km": res["median_km"], "p90_km": res["p90_km"],
               "ade_km": res["ade_km"], "fde_km": res["fde_km"],
               "n_windows": res["n_windows"], "n_storms": res["n_storms"],
               "n_params": pred.get("n_params")}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
