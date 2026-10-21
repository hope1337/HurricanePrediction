"""Training loops, validation, checkpointing, and early stopping.

Loss: Huber (configurable delta, default 50 km) on displacement targets in
kilometres — a documented physical representation.  Huber behaves like MSE
for typical steps and like MAE for extreme jumps, which suits heavy-tailed
translation errors.

* Direct models: loss over all 8 horizons of ``Y_direct`` per batch.
* Recursive models: teacher-forced one-step Huber loss on derived one-step
  samples (8 per direct window — counts are reported, not hidden).
* Validation (both strategies): Huber loss + genuine geographic ADE via the
  Task 4 evaluator on decoded positions.  Recursive validation ADE always
  uses full 8-step rollouts without teacher forcing.
* Selection: best validation ADE (primary, documented).  Early stopping on
  validation ADE with configurable patience; best + last checkpoints saved.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.evaluation.evaluator import evaluate_trajectories
from src.models.factory import build_model, count_parameters

log = logging.getLogger(__name__)


def build_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"optimizer must be 'adam' or 'adamw', got {name!r}")


def save_checkpoint(path: Path, model: nn.Module, optimizer, scheduler,
                    epoch: int, best_val_ade: float, config: dict,
                    scaler: dict, n_params: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch, "best_val_ade": best_val_ade,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "config": dict(config), "scaler": {k: np.asarray(v).tolist()
                                           if isinstance(v, np.ndarray) else v
                                           for k, v in scaler.items()},
        "n_params": n_params,
        "torch_version": torch.__version__,
    }, path)


def load_checkpoint(path: str | Path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    scaler = {k: (np.asarray(v, dtype=np.float64) if k in ("mean", "scale") else v)
              for k, v in ckpt["scaler"].items()}
    model = build_model(architecture=ckpt["config"]["architecture"],
                        strategy=ckpt["config"]["strategy"],
                        input_size=ckpt["config"]["input_size"],
                        hidden_size=ckpt["config"]["hidden_size"],
                        num_layers=ckpt["config"]["num_layers"],
                        dropout=ckpt["config"].get("dropout", 0.0),
                        forecast_steps=ckpt["config"].get("forecast_steps", 8),
                        scaler=scaler if ckpt["config"]["strategy"] == "recursive" else None,
                        feature_config=ckpt["config"].get("feature_config", "C"))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def _direct_val_metrics(model, loader, origins, Y_pos, sids, sample_ids, device, loss_fn):
    model.eval()
    tot_loss, n = 0.0, 0
    outs = []
    for Xb, Tb in loader:
        Xb, Tb = Xb.to(device), Tb.to(device)
        out = model(Xb)
        tot_loss += loss_fn(out, Tb).item() * len(Xb)
        n += len(Xb)
        outs.append(out.cpu().numpy())
    from src.models.forecasting import decode_direct
    pred_km = np.concatenate(outs, axis=0)
    pred_geo = decode_direct(pred_km, origins)
    res = evaluate_trajectories(pred_geo, Y_pos, sids, sample_ids)
    return tot_loss / n, res


def _recursive_rollout_ade(model, X_val, origins, Y_pos, sids, sample_ids, device,
                           batch_size: int = 512) -> dict:
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X_val), batch_size):
            outs.append(model.rollout(
                torch.as_tensor(X_val[i:i + batch_size]).to(device)).cpu().numpy())
    return evaluate_trajectories(np.concatenate(outs, axis=0), Y_pos, sids, sample_ids)


def train_run(run_cfg: dict, arrays: dict, scaler: dict, run_dir: Path,
              device: torch.device) -> dict:
    """Train one (architecture, strategy, feature) configuration.

    ``arrays``: {'train': {...}, 'val': {...}} from:data:`load_split_arrays`.
    ``run_cfg`` carries hyperparameters + identifiers.  Returns a summary dict
    and writes best.pt / last.pt / history.json / config.json into run_dir.
    """
    strategy = run_cfg["strategy"]
    model = build_model(architecture=run_cfg["architecture"], strategy=strategy,
                        input_size=run_cfg["input_size"], hidden_size=run_cfg["hidden_size"],
                        num_layers=run_cfg["num_layers"], dropout=run_cfg["dropout"],
                        forecast_steps=run_cfg.get("forecast_steps", 8),
                        scaler=scaler if strategy == "recursive" else None,
                        feature_config=run_cfg.get("feature_config", "C")).to(device)
    n_params, _ = count_parameters(model)
    opt = build_optimizer(model, run_cfg.get("optimizer", "adam"),
                          run_cfg["lr"], run_cfg.get("weight_decay", 0.0))
    loss_fn = nn.HuberLoss(delta=run_cfg.get("huber_delta", 50.0))
    sched_cfg = run_cfg.get("scheduler", {"enabled": True, "factor": 0.5, "patience": 5})
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=sched_cfg.get("factor", 0.5),
        patience=sched_cfg.get("patience", 5)) if sched_cfg.get("enabled", True) else None)

    from .data import derive_onestep_arrays, make_loader
    tr, va = arrays["train"], arrays["val"]
    if strategy == "direct":
        train_loader = make_loader(tr["X"], tr["Y_direct"], run_cfg["batch_size"],
                                   True, run_cfg["seed"])
        val_loader = make_loader(va["X"], va["Y_direct"], run_cfg["batch_size"], False)
    else:
        Xtr, Ttr = derive_onestep_arrays(tr, scaler, run_cfg.get("feature_config", "C"))
        Xva, Tva = derive_onestep_arrays(va, scaler, run_cfg.get("feature_config", "C"))
        log.info("one-step samples: train=%d (%.1f per direct window), val=%d",
                 len(Xtr), len(Xtr) / max(len(tr["X"]), 1), len(Xva))
        train_loader = make_loader(Xtr, Ttr, run_cfg["batch_size"], True, run_cfg["seed"])
        val_loader = make_loader(Xva, Tva, run_cfg["batch_size"], False)

    best_ade, best_epoch, bad = float("inf"), -1, 0
    history = {"train_loss": [], "val_loss": [], "val_ade": [], "lr": []}
    t_start = time.time()
    max_epochs = run_cfg.get("max_epochs", 100)
    patience = run_cfg.get("early_stopping_patience", 15)
    clip = run_cfg.get("gradient_clip_norm", 1.0)
    rollout_every = run_cfg.get("rollout_every", 1)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    for epoch in range(1, max_epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for Xb, Tb in train_loader:
            Xb, Tb = Xb.to(device), Tb.to(device)
            opt.zero_grad()
            out = model(Xb) if strategy == "direct" else model.forward_onestep(Xb)
            loss = loss_fn(out, Tb)
            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            tot += loss.item() * len(Xb)
            n += len(Xb)
        train_loss = tot / n
        if strategy == "direct":
            val_loss, val_res = _direct_val_metrics(
                model, val_loader, va["last_latlon"], va["Y"],
                va["sid"], va["sample_id"], device, loss_fn)
        else:
            model.eval()
            totv, nv = 0.0, 0
            with torch.no_grad():
                for Xb, Tb in val_loader:
                    Xb, Tb = Xb.to(device), Tb.to(device)
                    totv += loss_fn(model.forward_onestep(Xb), Tb).item() * len(Xb)
                    nv += len(Xb)
            val_loss = totv / nv
            if epoch % rollout_every == 0 or epoch == 1:
                val_res = _recursive_rollout_ade(
                    model, va["X"], va["last_latlon"], va["Y"],
                    va["sid"], va["sample_id"], device)
                last_rollout = val_res
            else:
                val_res = last_rollout
        val_ade = val_res["ade_km"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ade"].append(val_ade)
        history["lr"].append(opt.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step(val_ade)
        improved = val_ade < best_ade - 1e-9
        if improved:
            best_ade, best_epoch, bad = val_ade, epoch, 0
            save_checkpoint(run_dir / "best.pt", model, opt, scheduler, epoch,
                            best_ade, run_cfg, scaler, n_params)
        else:
            bad += 1
        log.info("epoch %3d train=%.4f val_loss=%.4f val_ADE=%.2f km %s", epoch,
                 train_loss, val_loss, val_ade, "(best)" if improved else f"(bad {bad})")
        if bad >= patience:
            log.info("early stopping at epoch %d (patience %d)", epoch, patience)
            break
    save_checkpoint(run_dir / "last.pt", model, opt, scheduler, epoch,
                    best_ade, run_cfg, scaler, n_params)
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    train_time = time.time() - t_start
    log.info("done: best val ADE %.2f km @ epoch %d, time %.1fs", best_ade, best_epoch, train_time)
    return {"best_val_ade": best_ade, "best_epoch": best_epoch,
            "epochs_run": epoch, "train_time_s": train_time,
            "n_params": n_params, "run_dir": str(run_dir), "history": history}
