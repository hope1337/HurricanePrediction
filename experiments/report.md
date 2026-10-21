# Experiment Report — TC Trajectory Forecasting (Task 5)

Config: `configs/training.yaml` (+ `configs/data.yaml` for preprocessing).
Run: `python scripts/run_experiments.py --experiment main|ablation|movement`;
single runs: `python scripts/train.py --model lstm --strategy direct`;
isolated eval: `python scripts/evaluate.py --checkpoint <best.pt> --split test`.
Checkpoints: `experiments/checkpoints/<arch>_<strategy>_<feat>_s<seed>/{best,last}.pt`
(`best.pt` = lowest genuine validation ADE; test touched only after selection).

## 1. Experimental setup

WP IBTrACS v04r01, seasons 1980–2024, 6-hourly; splits 1980–2016 / 2017–2020 /
2021–2024 by storm season (SID-disjoint, verified in Task 2). Input: 5 obs / 24h,
primary features [LAT, LON, v_E, v_N] (causal, km/h); targets: 8 steps to +48h.
Loss: Huber (δ=50.0 km) on km displacements.
Metrics on decoded geographic positions (Haversine km, Task 4).

## 2. Hardware and software environment

See per-run `summary.json` (`env` block: torch 2.14.1+cpu, device, seeds, git SHA
when available). Device used: auto (CUDA unavailable on
this machine — CPU fallback). Preprocessing reference: `data/processed/`
(Task 2 config hash in `dataset_metadata.json`).

## 3. Model hyperparameters

Hidden 64, layers
2, dropout
0.1, batch 64,
lr 0.001 (adam),
Huber δ 50.0 km, max epochs
100, early-stopping patience
15 on validation ADE, grad-clip
1.0, plateau scheduler
{'enabled': True, 'factor': 0.5, 'patience': 5}. Recursive training: teacher-forced one-step
loss on 8 derived one-step samples per direct window (counts logged); validation ADE
always from genuine 8-step rollouts on the same origins as direct models.

## 4. Training and validation procedures

Adam/AdamW, batch 64, seeds fixed (NumPy/torch/CUDA, deterministic cuDNN, single-process
loaders). Best checkpoint = lowest validation ADE; early stop on the same metric;
test evaluated once per selected checkpoint. `experiments/checkpoints/*/summary.json`
holds full history, config, timing, and environment.

## 5. Model comparison table

| model | strategy | feature | seed | run_tag | n_params | best_val_ade_km | best_epoch | epochs_run | train_time_s | test_mte_6h_km | test_mte_12h_km | test_mte_24h_km | test_mte_48h_km | test_ade_km | test_fde_km | test_n_windows | test_n_storms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rnn | recursive | C | 0 | rnn_recursive_C_s0 | 12930 | 238.55883335428013 | 19 | 34 | 251.2 | 38.28109181363434 | 82.61232659053005 | 195.5343892039728 | 493.4410803582073 | 244.6420616724473 | 493.4410803582088 | 2207 | 98 |
| rnn | direct | C | 0 | rnn_direct_C_s0 | 13840 | 241.99369140485544 | 99 | 100 | 93.9 | 43.50071463411442 | 90.04824563039666 | 202.4916283340677 | 479.0533318412228 | 245.46767579138157 | 479.05333184122304 | 2207 | 98 |
| gru | recursive | C | 0 | gru_recursive_C_s0 | 38530 | 242.6515048014515 | 6 | 21 | 258.3 | 38.31810295474441 | 82.71232008528065 | 197.25011230366917 | 504.92660759533055 | 248.45332744971824 | 504.92660759533067 | 2207 | 98 |
| gru | direct | C | 0 | gru_direct_C_s0 | 39440 | 245.0555139753727 | 97 | 100 | 155.2 | 45.37998604026877 | 93.11650806018042 | 205.1971129303022 | 482.11509601670735 | 248.1892729785193 | 482.1150960167077 | 2207 | 98 |
| lstm | recursive | C | 0 | lstm_recursive_C_s0 | 51330 | 243.25886454442852 | 6 | 21 | 295.9 | 38.419261895867486 | 83.09954617443172 | 197.5950292509187 | 497.2299599834452 | 246.5531505982054 | 497.2299599834456 | 2207 | 98 |
| lstm | direct | C | 0 | lstm_direct_C_s0 | 52240 | 246.5603911364084 | 82 | 97 | 178.0 | 45.621948591196784 | 93.70685730949756 | 206.2917258045661 | 487.4320701238526 | 250.2000804292406 | 487.4320701238517 | 2207 | 98 |

## 6. Recursive versus Direct analysis

Figure `figures/04_recursive_vs_direct.png`. Observed deltas are reported as measured;
whether error compounding (recursive) or horizon-independence (direct) explains them
is interpretation, not proof. Both share origins, splits, features, and metrics.

## 7. Feature Ablation findings

| model | strategy | feature | seed | run_tag | n_params | best_val_ade_km | test_mte_6h_km | test_mte_12h_km | test_mte_24h_km | test_mte_48h_km | test_ade_km | test_fde_km |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lstm | direct | A | 0 | lstm_direct_A_s0 | 51728 | 242.1278304072977 | 44.47629857632 | 91.99366435173172 | 202.9913835836443 | 472.7931384776023 | 244.10606533013265 | 472.79313847760267 |
| lstm | direct | B | 0 | lstm_direct_B_s0 | 52496 | 246.70427686644044 | 45.48524138987623 | 94.03731676040805 | 209.28150487666773 | 492.1405288560063 | 253.03360845434776 | 492.140528856006 |
| lstm | direct | C | 0 | lstm_direct_C_s0 | 52240 | 246.5603911364084 | 45.621948591196784 | 93.70685730949756 | 206.2917258045661 | 487.4320701238526 | 250.2000804292406 | 487.4320701238517 |

Figure `figures/05_feature_ablation.png` (LSTM-direct, identical cases). Claims about
motion features require the measured ADE/FDE deltas above — no conclusion is drawn
beyond what the table shows.

## 8. Error Analysis by Storm Movement

| model | strategy | group | n_windows | n_storms | mte_6h_km | mte_12h_km | mte_24h_km | mte_48h_km | ade_km | fde_km |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rnn | recursive | ambiguous | 25 | 13 | 26.621279892135394 | 63.9403698673777 | 176.63064332391008 | 472.5060269357988 | 226.72921036466195 | 472.5060269357988 |
| rnn | recursive | straight | 1475 | 92 | 38.67160683761539 | 83.12346168849855 | 196.22228088747352 | 492.62487762571385 | 244.8685291723169 | 492.6248776257145 |
| rnn | recursive | turning | 707 | 86 | 37.87866697299134 | 82.20620869601133 | 194.7677037921451 | 495.8841855434833 | 244.80299762773305 | 495.8841855434837 |
| rnn | direct | ambiguous | 25 | 13 | 30.561685006726343 | 68.8023508154031 | 178.59573825727304 | 440.4034617186745 | 218.53174935272057 | 440.4034617186746 |
| rnn | direct | straight | 1475 | 92 | 44.10908061653308 | 90.75720181336423 | 204.132231197311 | 485.1839013187996 | 247.84375103899808 | 485.1839013187992 |
| rnn | direct | turning | 707 | 86 | 42.68902569015036 | 89.32043374991238 | 199.91384619635463 | 467.62993265259314 | 241.4629900356985 | 467.6299326525936 |
| gru | recursive | ambiguous | 25 | 13 | 24.12365222033644 | 58.48678762889227 | 157.98671985051575 | 438.4016694999267 | 203.47197795495063 | 438.4016694999267 |
| gru | recursive | straight | 1475 | 92 | 38.96659435752586 | 83.91050729892498 | 199.71445246129656 | 507.0992468601933 | 250.5289019374929 | 507.0992468601929 |
| gru | recursive | turning | 707 | 86 | 37.46709368919659 | 81.06919727238716 | 193.4971887942316 | 502.746240603412 | 245.7136688470329 | 502.7462406034115 |
| gru | direct | ambiguous | 25 | 13 | 30.08177302166008 | 61.22628570556162 | 154.62374639470855 | 395.84752697355486 | 192.96449475712495 | 395.84752697355486 |
| gru | direct | straight | 1475 | 92 | 46.16254146234655 | 94.37029797873345 | 207.49098817422905 | 489.8081203533764 | 251.53685058814963 | 489.8081203533769 |
| gru | direct | turning | 707 | 86 | 44.28831146869931 | 91.62841107149536 | 202.19975533284455 | 469.1157725789311 | 243.1580742250967 | 469.1157725789308 |
| lstm | recursive | ambiguous | 25 | 13 | 23.89054983592364 | 57.1261092837447 | 153.97878688618212 | 417.96274213012856 | 196.80707555181328 | 417.96274213012856 |
| lstm | recursive | straight | 1475 | 92 | 39.08158495184785 | 84.26393645620148 | 200.0399396094028 | 502.4041531773847 | 249.34700230534511 | 502.4041531773841 |
| lstm | recursive | turning | 707 | 86 | 37.551215635510495 | 81.58874031397488 | 194.0365614720694 | 489.2380865538466 | 242.48344848806224 | 489.2380865538465 |
| lstm | direct | ambiguous | 25 | 13 | 30.35567619660133 | 64.90685247129726 | 166.1873899464967 | 405.9526486769024 | 200.4868723107305 | 405.9526486769024 |
| lstm | direct | straight | 1475 | 92 | 46.29631807289312 | 94.56536804143452 | 208.515085247252 | 498.4466548222112 | 254.3212238356306 | 498.4466548222117 |
| lstm | direct | turning | 707 | 86 | 44.75485074729702 | 92.93415121522298 | 203.07129188446703 | 467.3337293969675 | 243.3601139208071 | 467.33372939696727 |

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
