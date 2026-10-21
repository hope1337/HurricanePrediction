"""Train one (architecture, strategy[, feature]) configuration.

Examples:
    python scripts/train.py --model lstm --strategy direct
    python scripts/train.py --model gru --strategy recursive --feature C --epochs 50 --seed 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.data import load_split_arrays  # noqa: E402
from src.training.seed import env_info, select_device, set_seed  # noqa: E402
from src.training.trainer import train_run  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train one TC forecasting model.")
    ap.add_argument("--model", required=True, choices=["rnn", "gru", "lstm"])
    ap.add_argument("--strategy", required=True, choices=["direct", "recursive"])
    ap.add_argument("--feature", default="C", choices=["A", "B", "C"])
    ap.add_argument("--config", default="configs/training.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    import logging
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    with open(args.config, encoding="utf-8") as fh:
        base = yaml.safe_load(fh)
    run_cfg = dict(base.get("model_defaults", {}))
    feature_sizes = {"A": 2, "B": 5, "C": 4}
    run_cfg.update({"architecture": args.model, "strategy": args.strategy,
                    "feature_config": args.feature,
                    "input_size": feature_sizes[args.feature],
                    "seed": base.get("seed", 0) if args.seed is None else args.seed,
                    "device": base.get("device", "auto") if args.device is None else args.device})
    if args.epochs is not None:
        run_cfg["max_epochs"] = args.epochs
    if args.batch_size is not None:
        run_cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        run_cfg["lr"] = args.lr

    set_seed(run_cfg["seed"])
    device = select_device(run_cfg["device"])
    proc = Path(base.get("processed_dir", "data/processed"))
    arrays = {"train": load_split_arrays(str(proc / "train.npz"), args.feature),
              "val": load_split_arrays(str(proc / "validation.npz"), args.feature)}
    scaler = joblib.load(proc / "scalers.joblib")
    tag = args.run_tag or f"{args.model}_{args.strategy}_{args.feature}_s{run_cfg['seed']}"
    run_dir = Path(base.get("experiments_dir", "experiments")) / "checkpoints" / tag
    summary = train_run(run_cfg, arrays, scaler, run_dir, device)
    summary["env"] = env_info()
    summary["run_tag"] = tag
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                          encoding="utf-8")
    print(f"done: {tag} best val ADE={summary['best_val_ade']:.2f} km "
          f"@epoch {summary['best_epoch']} ({summary['train_time_s']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
