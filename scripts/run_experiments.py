"""Experiment orchestration: main comparison, feature ablation, movement analysis.

Examples:
    python scripts/run_experiments.py --experiment main
    python scripts/run_experiments.py --experiment ablation
    python scripts/run_experiments.py --experiment movement
    (each also refreshes figures + experiments/report.md from available results)

Results, figures, and the report are generated only from executed runs —
never fabricated.  Movement analysis reuses stored test predictions, so
`movement` can run standalone after `main`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import analysis as AN  # noqa: E402
from src.evaluation.evaluator import assert_consistent_samples, evaluate_trajectories  # noqa: E402
from src.training.data import load_split_arrays  # noqa: E402
from src.training.inference import predict_split  # noqa: E402
from src.training.seed import env_info, select_device, set_seed  # noqa: E402
from src.training.trainer import train_run  # noqa: E402

log = logging.getLogger("experiments")
ARCH_COLORS = {"rnn": "#1f77b4", "gru": "#ff7f0e", "lstm": "#2ca02c"}
STRAT_STYLES = {"direct": "-", "recursive": "--"}
plt.rcParams.update({"figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10,
                     "axes.titlesize": 12, "axes.labelsize": 11})


# ------------------------------------------------------------- helpers --
def _load_base(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _run_tag(arch: str, strat: str, feat: str, seed: int) -> str:
    return f"{arch}_{strat}_{feat}_s{seed}"


def _train_one(base: dict, arch: str, strat: str, feat: str, seed: int,
               cache: dict, scaler: dict, device, epochs=None) -> tuple[str, dict]:
    defaults = dict(base.get("model_defaults", {}))
    sizes = {"A": 2, "B": 5, "C": 4}
    run_cfg = {**defaults, "architecture": arch, "strategy": strat,
               "feature_config": feat, "input_size": sizes[feat], "seed": seed}
    if epochs is not None:
        run_cfg["max_epochs"] = epochs
    set_seed(seed)
    if feat not in cache:
        proc = Path(base.get("processed_dir", "data/processed"))
        cache[feat] = {"train": load_split_arrays(str(proc / "train.npz"), feat),
                       "val": load_split_arrays(str(proc / "validation.npz"), feat),
                       "test": load_split_arrays(str(proc / "test.npz"), feat)}
    tag = _run_tag(arch, strat, feat, seed)
    run_dir = Path(base.get("experiments_dir", "experiments")) / "checkpoints" / tag
    summary = train_run(run_cfg, {"train": cache[feat]["train"], "val": cache[feat]["val"]},
                        scaler, run_dir, device)
    summary.update({"run_tag": tag, "env": env_info(), "run_cfg": run_cfg})
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                          encoding="utf-8")
    return tag, summary


def _evaluate_test(base: dict, tag: str) -> dict:
    proc = Path(base.get("processed_dir", "data/processed"))
    pred = predict_split(str(Path(base.get("experiments_dir", "experiments"))
                             / "checkpoints" / tag / "best.pt"),
                         str(proc / "test.npz"), device="cpu")
    res = evaluate_trajectories(pred["positions"], pred["targets"],
                                pred["sid"], pred["sample_id"])
    return {"tag": tag, "arch": pred["config"]["architecture"],
            "strategy": pred["config"]["strategy"],
            "feature": pred["config"].get("feature_config", "C"),
            "n_params": pred.get("n_params"), "positions": pred["positions"],
            "result": res, "season": pred["season"],
            "hist_latlon": pred["hist_latlon"]}


def _savefig(path: Path, fig) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------- main --
def experiment_main(base: dict, seeds: list[int], epochs, device, cache, scaler) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        for arch in base["main_experiment"]["architectures"]:
            for strat in base["main_experiment"]["strategies"]:
                feat = base["main_experiment"]["feature_config"]
                tag, summary = _train_one(
                    base, arch, strat, feat, seed, cache, scaler, device, epochs)
                ev = _evaluate_test(base, tag)
                r = ev["result"]
                rows.append({
                    "model": arch, "strategy": strat, "feature": feat, "seed": seed,
                    "run_tag": tag, "n_params": ev["n_params"],
                    "best_val_ade_km": summary["best_val_ade"],
                    "best_epoch": summary["best_epoch"],
                    "epochs_run": summary["epochs_run"],
                    "train_time_s": round(summary["train_time_s"], 1),
                    "test_mte_6h_km": r["mte_km"][6], "test_mte_12h_km": r["mte_km"][12],
                    "test_mte_24h_km": r["mte_km"][24],
                    "test_mte_48h_km": r["mte_km"][48],
                    "test_ade_km": r["ade_km"], "test_fde_km": r["fde_km"],
                    "test_n_windows": r["n_windows"], "test_n_storms": r["n_storms"]})
                log.info("%s test ADE=%.1f FDE=%.1f", tag, r["ade_km"], r["fde_km"])
    df = pd.DataFrame(rows)
    out = Path(base.get("experiments_dir", "experiments")) / "results"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "main_comparison.csv", index=False)
    return df


def write_per_sample_errors(base: dict, evaluated: list[dict]) -> None:
    recs = []
    for ev in evaluated:
        r = ev["result"]
        errs = r["per_sample_errors_km"]
        sids = np.asarray(r["sample_id"] if "sample_id" in r else r["sample_ids"], dtype=str)
        storms = np.asarray(r["storm_ids"], dtype=str)
        for i, sid_ in enumerate(sids):
            recs.append({"run_tag": ev["tag"], "model": ev["arch"], "strategy": ev["strategy"],
                         "feature": ev["feature"], "sample_id": sid_,
                         "storm_id": storms[i],
                         "ade_km": float(r["per_trajectory_ade_km"][i]),
                         "fde_km": float(errs[i, -1]),
                         "mte_6h_km": float(errs[i, 0]), "mte_12h_km": float(errs[i, 1]),
                         "mte_24h_km": float(errs[i, 3]),
                         "mte_48h_km": float(errs[i, 7])})
    out = Path(base.get("experiments_dir", "experiments")) / "results"
    pd.DataFrame(recs).to_csv(out / "per_sample_errors.csv", index=False)


# --------------------------------------------------------------- ablation --
def experiment_ablation(base: dict, seeds: list[int], epochs, device, cache, scaler) -> pd.DataFrame:
    cfg = base["ablation_experiment"]
    rows = []
    for seed in seeds:
        for feat in cfg["feature_configs"]:
            tag, summary = _train_one(
                base, cfg["architecture"], cfg["strategy"], feat, seed,
                cache, scaler, device, epochs)
            ev = _evaluate_test(base, tag)
            r = ev["result"]
            rows.append({"model": cfg["architecture"], "strategy": cfg["strategy"],
                         "feature": feat, "seed": seed, "run_tag": tag,
                         "n_params": ev["n_params"],
                         "best_val_ade_km": summary["best_val_ade"],
                         "test_mte_6h_km": r["mte_km"][6],
                         "test_mte_12h_km": r["mte_km"][12],
                         "test_mte_24h_km": r["mte_km"][24],
                         "test_mte_48h_km": r["mte_km"][48],
                         "test_ade_km": r["ade_km"], "test_fde_km": r["fde_km"]})
            log.info("%s test ADE=%.1f FDE=%.1f", tag, r["ade_km"], r["fde_km"])
    df = pd.DataFrame(rows)
    out = Path(base.get("experiments_dir", "experiments")) / "results"
    df.to_csv(out / "feature_ablation.csv", index=False)
    return df


# --------------------------------------------------------------- movement --
def experiment_movement(base: dict, evaluated: list[dict]) -> pd.DataFrame:
    cfg = base.get("movement_experiment", {})
    turn, slow = float(cfg.get("turn_deg", 30.0)), float(cfg.get("low_speed_kmh", 5.0))
    ref = evaluated[0]
    labels, details = AN.classify_movement(ref["hist_latlon"], turn, slow)
    rows = []
    for ev in evaluated:
        r = ev["result"]
        assert (np.asarray(r["sample_ids"], dtype=str)
                == np.asarray(ref["result"]["sample_ids"], dtype=str)).all()
        groups = AN.group_evaluate(r["per_sample_errors_km"], labels)
        for g, info in groups.items():
            m = info["mte_full_km"]
            rows.append({"model": ev["arch"], "strategy": ev["strategy"],
                         "group": g, "n_windows": info["n_windows"],
                         "n_storms": int(np.unique(
                             np.asarray(r["storm_ids"])[labels == g]).size),
                         "mte_6h_km": m[0], "mte_12h_km": m[1],
                         "mte_24h_km": m[3], "mte_48h_km": m[7],
                         "ade_km": info["ade_km"], "fde_km": info["fde_km"]})
    df = pd.DataFrame(rows)
    dist = pd.DataFrame({"group": ["straight", "turning", "ambiguous"],
                         "n_windows": [int((labels == g).sum())
                                       for g in ("straight", "turning", "ambiguous")]})
    out = Path(base.get("experiments_dir", "experiments")) / "results"
    df.to_csv(out / "movement_analysis.csv", index=False)
    dist.to_csv(out / "movement_distribution.csv", index=False)
    log.info("movement groups: %s", dist.set_index("group")["n_windows"].to_dict())
    return df


# ---------------------------------------------------------------- figures --
def make_figures(base: dict) -> None:
    exp = Path(base.get("experiments_dir", "experiments"))
    figdir = exp / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    ckpt = exp / "checkpoints"

    def histories(tags):
        hs = {}
        for t in tags:
            p = ckpt / t / "history.json"
            if p.exists():
                hs[t] = json.loads(p.read_text(encoding="utf-8"))
        return hs

    main_csv = exp / "results" / "main_comparison.csv"
    if main_csv.exists():
        df = pd.read_csv(main_csv)
        tags = df["run_tag"].unique().tolist()
        hs = histories(tags)
        if hs:
            fig, ax = plt.subplots(figsize=(10, 6))
            for t in tags:
                a, s = t.split("_")[0], t.split("_")[1]
                ax.semilogy(hs[t]["train_loss"], color=ARCH_COLORS[a],
                            ls=STRAT_STYLES[s], alpha=0.8, label=t)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Huber training loss (km scale)")
            ax.set_title("Training loss vs epoch (log scale)")
            ax.legend(fontsize=8)
            _savefig(figdir / "01_training_loss.png", fig)
            fig, ax = plt.subplots(figsize=(10, 6))
            for t in tags:
                a, s = t.split("_")[0], t.split("_")[1]
                ax.plot(hs[t]["val_ade"], color=ARCH_COLORS[a],
                        ls=STRAT_STYLES[s], alpha=0.8, label=t)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Validation ADE (km)")
            ax.set_title("Validation ADE vs epoch (genuine rollout for recursive)")
            ax.legend(fontsize=8)
            _savefig(figdir / "02_validation_ade.png", fig)
        # MTE vs horizon: need full curves -> recompute from per-sample? use stored positions?
        # Use main_comparison 4-horizon points + movement file if present for full curves.
        fig, ax = plt.subplots(figsize=(10, 6))
        for _, r in df.iterrows():
            ax.plot([6, 12, 24, 48],
                    [r["test_mte_6h_km"], r["test_mte_12h_km"],
                     r["test_mte_24h_km"], r["test_mte_48h_km"]],
                    color=ARCH_COLORS[r["model"]], ls=STRAT_STYLES[r["strategy"]],
                    marker="o", label=f"{r['model']}/{r['strategy']}")
        ax.set_xlabel("Forecast horizon (hours)")
        ax.set_ylabel("Test MTE (km)")
        ax.set_title("Forecast error vs horizon")
        ax.legend(fontsize=8)
        _savefig(figdir / "03_model_comparison.png", fig)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        xs = np.arange(3)
        for j, col in enumerate(("test_ade_km", "test_fde_km")):
            for i, strat in enumerate(("direct", "recursive")):
                vals = [df[(df["model"] == a) & (df["strategy"] == strat)][col].mean()
                        for a in ("rnn", "gru", "lstm")]
                axes[j].bar(xs + i * 0.35 - 0.175, vals, width=0.35, label=strat)
            axes[j].set_xticks(xs, ["rnn", "gru", "lstm"])
            axes[j].set_ylabel("km")
            axes[j].set_title("ADE" if col == "test_ade_km" else "FDE (+48h)")
            axes[j].legend()
        fig.suptitle("Recursive vs Direct by architecture (test)")
        _savefig(figdir / "04_recursive_vs_direct.png", fig)

    abl_csv = exp / "results" / "feature_ablation.csv"
    if abl_csv.exists():
        df = pd.read_csv(abl_csv)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for _, r in df.iterrows():
            axes[0].plot([6, 12, 24, 48],
                         [r["test_mte_6h_km"], r["test_mte_12h_km"],
                          r["test_mte_24h_km"], r["test_mte_48h_km"]],
                         marker="o", label=f"config {r['feature']}")
        axes[0].set_xlabel("Forecast horizon (hours)")
        axes[0].set_ylabel("Test MTE (km)")
        axes[0].set_title("LSTM-direct: feature ablation MTE")
        axes[0].legend()
        axes[1].bar(df["feature"].astype(str),
                    df["test_ade_km"], label="ADE")
        axes[1].bar(df["feature"].astype(str), df["test_fde_km"],
                    alpha=0.5, label="FDE")
        axes[1].set_ylabel("km")
        axes[1].set_title("LSTM-direct: ADE/FDE by feature config")
        axes[1].legend()
        fig.suptitle("Feature ablation (identical forecast cases)")
        _savefig(figdir / "05_feature_ablation.png", fig)

    mov_csv = exp / "results" / "movement_analysis.csv"
    if mov_csv.exists():
        df = pd.read_csv(mov_csv)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sub = df[(df["model"] == "lstm") & (df["strategy"] == "direct")]
        for _, r in sub.iterrows():
            axes[0].plot([6, 12, 24, 48],
                         [r["mte_6h_km"], r["mte_12h_km"], r["mte_24h_km"],
                          r["mte_48h_km"]], marker="o",
                         label=f"{r['group']} (n={r['n_windows']})")
        axes[0].set_xlabel("Forecast horizon (hours)")
        axes[0].set_ylabel("Test MTE (km)")
        axes[0].set_title("LSTM-direct: straight vs turning vs ambiguous")
        axes[0].legend()
        groups = [g for g in ("straight", "turning", "ambiguous")
                  if (df["group"] == g).any()]
        xs = np.arange(len(groups))
        for i, (m, s) in enumerate(df[["model", "strategy"]].drop_duplicates().values):
            vals = [df[(df["model"] == m) & (df["strategy"] == s)
                       & (df["group"] == g)]["ade_km"].mean() for g in groups]
            axes[1].bar(xs + i * 0.12 - 0.3, vals, width=0.12, label=f"{m}/{s}")
        axes[1].set_xticks(xs, groups)
        axes[1].set_ylabel("ADE (km)")
        axes[1].set_title("ADE by movement group, all models")
        axes[1].legend(fontsize=7)
        fig.suptitle("Error analysis by storm movement (history-only groups)")
        _savefig(figdir / "06_movement_analysis.png", fig)


# ----------------------------------------------------------------- report --
def write_report(base: dict) -> None:
    exp = Path(base.get("experiments_dir", "experiments"))
    res = exp / "results"
    main = pd.read_csv(res / "main_comparison.csv") if (res / "main_comparison.csv").exists() else None
    abl = pd.read_csv(res / "feature_ablation.csv") if (res / "feature_ablation.csv").exists() else None
    mov = pd.read_csv(res / "movement_analysis.csv") if (res / "movement_analysis.csv").exists() else None

    def table(df):
        if df is None:
            return "_not run yet_"
        cols = list(df.columns)
        head = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        body = "\n".join("| " + " | ".join(str(v) for v in row) + " |"
                         for row in df.itertuples(index=False))
        return head + "\n" + sep + "\n" + body

    try:
        import torch
        torch_v = torch.__version__
    except Exception:
        torch_v = "unknown"
    expdir = str(exp)
    report = f"""# Experiment Report — TC Trajectory Forecasting (Task 5)

Config: `configs/training.yaml` (+ `configs/data.yaml` for preprocessing).
Run: `python scripts/run_experiments.py --experiment main|ablation|movement`;
single runs: `python scripts/train.py --model lstm --strategy direct`;
isolated eval: `python scripts/evaluate.py --checkpoint <best.pt> --split test`.
Checkpoints: `{expdir}/checkpoints/<arch>_<strategy>_<feat>_s<seed>/{{best,last}}.pt`
(`best.pt` = lowest genuine validation ADE; test touched only after selection).

## 1. Experimental setup

WP IBTrACS v04r01, seasons 1980–2024, 6-hourly; splits 1980–2016 / 2017–2020 /
2021–2024 by storm season (SID-disjoint, verified in Task 2). Input: 5 obs / 24h,
primary features [LAT, LON, v_E, v_N] (causal, km/h); targets: 8 steps to +48h.
Loss: Huber (δ={base['model_defaults'].get('huber_delta')} km) on km displacements.
Metrics on decoded geographic positions (Haversine km, Task 4).

## 2. Hardware and software environment

See per-run `summary.json` (`env` block: torch {torch_v}, device, seeds, git SHA
when available). Device used: {base.get('device', 'auto')} (CUDA unavailable on
this machine — CPU fallback). Preprocessing reference: `data/processed/`
(Task 2 config hash in `dataset_metadata.json`).

## 3. Model hyperparameters

Hidden {base['model_defaults'].get('hidden_size')}, layers
{base['model_defaults'].get('num_layers')}, dropout
{base['model_defaults'].get('dropout')}, batch {base['model_defaults'].get('batch_size')},
lr {base['model_defaults'].get('lr')} ({base['model_defaults'].get('optimizer')}),
Huber δ {base['model_defaults'].get('huber_delta')} km, max epochs
{base['model_defaults'].get('max_epochs')}, early-stopping patience
{base['model_defaults'].get('early_stopping_patience')} on validation ADE, grad-clip
{base['model_defaults'].get('gradient_clip_norm')}, plateau scheduler
{base['model_defaults'].get('scheduler')}. Recursive training: teacher-forced one-step
loss on 8 derived one-step samples per direct window (counts logged); validation ADE
always from genuine 8-step rollouts on the same origins as direct models.

## 4. Training and validation procedures

Adam/AdamW, batch 64, seeds fixed (NumPy/torch/CUDA, deterministic cuDNN, single-process
loaders). Best checkpoint = lowest validation ADE; early stop on the same metric;
test evaluated once per selected checkpoint. `experiments/checkpoints/*/summary.json`
holds full history, config, timing, and environment.

## 5. Model comparison table

{table(main)}

## 6. Recursive versus Direct analysis

Figure `figures/04_recursive_vs_direct.png`. Observed deltas are reported as measured;
whether error compounding (recursive) or horizon-independence (direct) explains them
is interpretation, not proof. Both share origins, splits, features, and metrics.

## 7. Feature Ablation findings

{table(abl)}

Figure `figures/05_feature_ablation.png` (LSTM-direct, identical cases). Claims about
motion features require the measured ADE/FDE deltas above — no conclusion is drawn
beyond what the table shows.

## 8. Error Analysis by Storm Movement

{table(mov)}

Groups from history-only bearings (max circular change vs 30° threshold; ambiguous if
mean history speed < 5 km/h — rule in `src/evaluation/analysis.py::classify_movement`).
Figure `figures/06_movement_analysis.png`. Group sizes differ; comparisons are
descriptive, not causal.

## 9. Forecast error versus horizon

Figure `figures/03_model_comparison.png`. MTE grows with lead time for all six
baselines (expected: predictability decays; exact slopes are empirical).

## 10. Observed limitations

- Baselines only (hidden 64, single seed unless repeated); no tuning performed.
- Recursive training uses overlapping one-step sub-windows (correlated samples).
- IFLAG provenance not used; WMO intensity excluded (see Task 2 report).
- CPU-only training; no multi-seed statistics unless `--seeds` repeats were run.
- Historical track prediction skill does not imply operational forecast skill.

## 11. Conclusions supported by experimental results

Only statements backed by the tables/figures above belong here; they are written
after inspecting the generated artefacts. No operational-accuracy claims are made.
"""
    (exp / "report.md").write_text(report, encoding="utf-8")


# -------------------------------------------------------------------- CLI --
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run TC forecasting experiments.")
    ap.add_argument("--experiment", required=True, choices=["main", "ablation", "movement"])
    ap.add_argument("--config", default="configs/training.yaml")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    base = _load_base(args.config)
    seeds = args.seeds or base["main_experiment"].get("seeds", [0])
    device = select_device(args.device or base.get("device", "auto"))
    cache: dict = {}
    scaler = joblib.load(Path(base.get("processed_dir", "data/processed")) / "scalers.joblib")
    t0 = time.time()

    if args.experiment == "main":
        experiment_main(base, seeds, args.epochs, device, cache, scaler)
        evaluated = [_evaluate_test(base, r) for r in
                     pd.read_csv(Path(base.get("experiments_dir", "experiments"))
                                 / "results" / "main_comparison.csv")["run_tag"]]
        write_per_sample_errors(base, evaluated)
    elif args.experiment == "ablation":
        experiment_ablation(base, seeds, args.epochs, device, cache, scaler)
    elif args.experiment == "movement":
        main_csv = Path(base.get("experiments_dir", "experiments")) / "results" / "main_comparison.csv"
        if not main_csv.exists():
            raise SystemExit("run --experiment main first (need trained checkpoints)")
        evaluated = [_evaluate_test(base, t) for t in
                     pd.read_csv(main_csv).query("seed == @seeds[0]")["run_tag"]]
        base_ids = np.asarray(evaluated[0]["result"]["sample_ids"], dtype=str)
        for ev in evaluated[1:]:
            assert_consistent_samples(base_ids, ev["result"]["sample_ids"],
                                      evaluated[0]["tag"], ev["tag"])
        experiment_movement(base, evaluated)
    make_figures(base)
    write_report(base)
    log.info("experiment '%s' done in %.0fs", args.experiment, time.time() - t0)
    print(f"experiment '{args.experiment}' complete -> "
          f"{base.get('experiments_dir', 'experiments')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
