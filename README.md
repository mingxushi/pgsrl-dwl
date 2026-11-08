# Physics-Guided Structured Representation Learning for Doppler Wind Lidar Velocity Prediction

Source code and raw/processed data for the manuscript of the same title, prepared for IEEE Transactions on Geoscience and Remote Sensing. This repository contains the proposed method, not implementations of the comparison baselines. Repository: https://github.com/mingxushi/pgsrl-dwl. No publication DOI has been assigned.

## Method

**Stage 1 — physical representation learning.** Doppler spectra are encoded into peak-, width- and noise-related latent representations. A Doppler-based decoder estimates velocity with a learned correction; reconstruction, calibration and representation losses regularize the model.

**Stage 2 — dynamic forecasting.** The Stage-1 encoder and physics decoder are frozen. A DLinear direct anchor predicts wind profiles, residual latent dynamics evolves the physical states, and a causal reliability gate applies a bounded correction to the direct prediction. Training first warms up the dynamics and then fine-tunes the guidance network and transitions. The final output is the **guided wind prediction**.

Inference starts from 12 observed spectrum–wind pairs. Latent states and separate direct/guided wind histories are recursively updated afterward, without future observations. `rollout.py` includes the complete evolution, prediction, history-update and training-loss process.

## Files

```text
README.md
model.py                 spectral encoder, physics decoder and transition networks
rollout.py               recursive prediction and Stage-2 losses
refinement.py            optional refinement dependency; disabled in paper configs
preprocess.py            original binary parsing and chronological preprocessing
utils.py                 inverse wind normalization
scripts/
  train.py               complete three-phase training and test workflow
  train_stage2.py         dynamics warm-up or guidance training
  evaluate.py             validation-selected full-loop test evaluation
  predict.py              future prediction from 12 initial observations
  restore_data.py         verify and restore packaged data
configs/
  NR-I/                   warmup.json, guidance.json
  NR-II/                  warmup.json, guidance.json
  FR/                     warmup.json, guidance.json
data/
  raw/                    original binary observations in ZIP shards
  processed/              original processed files in reconstructable parts
  manifests/              ordered binary filenames and per-file SHA-256 hashes
  catalog.json            sizes, checksums and reconstruction order
  LICENSE                 separate data licensing status
requirements.txt
LICENSE                   MIT software license
CITATION.cff
```

Internal development labels have been removed from filenames. Older local V2 referred to the original rollout implementation, which continued evolving; V4 referred to dynamics warm-up, and V5 to reliability-guided fine-tuning. These were development labels, not three methods in the paper. The current names describe their actual roles.

## Installation

Use an isolated Python environment with a hardware-appropriate PyTorch installation:

```sh
python -m pip install -r requirements.txt
```

Core dependencies are PyTorch, NumPy and tqdm. Preparation checks used Python 3.13.9 and PyTorch 2.13.0+cpu; this is not claimed to be the original GPU training environment.

## Data

| Dataset | Period | Original binaries | Total steps | Train / validation / test |
|---|---|---:|---:|---|
| NR-I | 2025-08-25 to 2025-08-30 | 163,691 | 454 | 317 / 68 / 69 |
| NR-II | 2025-08-15 to 2025-08-22 | 183,410 | 509 | 356 / 76 / 77 |
| FR | 2025-07-19 to 2025-07-22 | 74,795 | 207 | 144 / 31 / 32 |

All sequences use 60 range gates and 100 spectral bins, with chronological 70/15/15 splitting and training-only normalization. Each sequence NPZ contains `spectra[T,60,100]`, `wind_speed[T,60]`, `step_indices`, `split_name` and acquisition metadata. The original near-range folder labels were reversed relative to the paper; this release uses acquisition dates/counts to label NR-I and NR-II correctly.

The payload includes 421,896 original `.bin` files and 21 original processed NPZ/JSON files, including record-level caches, split arrays, normalizers and preprocessing metadata. There are 425 binary artifacts totaling 21,631,534,594 bytes (about 21.63 GB), tracked through **Git LFS**. The processed parts are no larger than 32 MiB. Ordinary GitHub browser upload is not suitable for this payload.

Clone the repository and retrieve the data through Git LFS:

```sh
git lfs install
git clone https://github.com/mingxushi/pgsrl-dwl.git
cd pgsrl-dwl
git lfs pull
python scripts/restore_data.py --dataset all
```

Processed files are restored into `data/ready/NR-I`, `data/ready/NR-II` and `data/ready/FR`. Add `--raw` to also extract original binaries under `data/raw_bin/<dataset>`. Use `--verify-only` to check packaged artifacts without restoration; add `--raw` to include raw ZIPs in that check. Complete-file hashes are checked during reassembly, and mismatching existing files are not overwritten.

To recompute preprocessing into a fresh location:

```sh
python preprocess.py --input-dir data/raw_bin/NR-I --output-dir data/regenerated/NR-I --max-gates 60 --step-records 360
```

Byte-preserving restoration of released processed arrays and recomputation from raw observations are different operations. The parser groups 360 records and computes spectral wind estimates. Exact complete-sweep grouping and the manuscript's manufacturer-derived reference description still require author confirmation. First-record/cache comparisons match NR-I/NR-II exactly; FR has small differences within spectrum 2e-4 and wind 1e-5 absolute tolerances. No exhaustive raw-to-processed regeneration is claimed. Raw filenames preserve original timestamps and auxiliary/transition scan settings; nominal acquisition geometry does not describe every raw filename.

## Training and evaluation

Run from the repository root after restoring processed arrays:

```sh
python scripts/train.py --dataset NR-I --device cuda
python scripts/train.py --dataset NR-II --device cuda
python scripts/train.py --dataset FR --device cuda
```

Each command trains seeds 42, 123, 456, 789 and 1024 independently. Use `--seeds 42` for one seed, `--dry-run` to inspect the commands, and a new `--output-root` for a new run.

The workflow runs 10 epochs of Stage 1, 50 epochs of dynamics warm-up, then 40 epochs of guidance fine-tuning. Stage 1 uses Adam at .001 and batch 16; Stage 2 uses batch 8. Warm-up horizons are 4/8/16/32/56 for NR-I, 4/8/16/32/64 for NR-II and 4/8/16/19 for FR. Guidance horizons are 8/16/32/56, 8/16/32/64 and 8/16/19 respectively. Guidance correction scale is .25, with a hard reliability gate. Numeric hyperparameters are preserved in the two phase configurations per dataset.

Best checkpoints are selected on validation only, followed by a frozen full-loop test. Test trajectories contain 57/65/20 predictions after the initial 12 observations; these differ from the curriculum maxima. Results are saved under `results/paper/<dataset>/seed<seed>/frozen_test/`, including `metrics.json`, `test_predictions.npz`, `test_horizon_mae.csv` and `test_state_drift.csv`.

Scores are denormalized MAE/RMSE/R2 over the full trajectory, aggregated over the five seeds. Historical output keys `decode`/`refined` denote guided/final outputs in the paper configuration, whereas `raw_decode` is the auxiliary physics-decoded branch and `direct` the anchor. Optional optimization-based refinement is disabled. The rollout uses valid-sample-weighted horizon discounting; the manuscript's two discount formulas require clarification when valid counts differ.

## Future prediction

Prepare `initial.npz` with normalized `spectra[12,60,100]` and `wind_speed[12,60]` using the released training normalizer. Then, for example:

```sh
python scripts/predict.py --checkpoint results/paper/NR-II/seed42/selection/NR-II_seed42_h64_sel-val_long_loop_decode_mae.best.pt --stage1-checkpoint results/paper/NR-II/seed42/stage1/model.stage1.best.pt --initial-npz initial.npz --stats-npz data/ready/NR-II/normalization_stats.npz --horizon 65 --output results/nrii_predictions.npz --device cpu
```

The output contains denormalized `guided`, `direct` and `physics_decoded` wind profiles, binary gates and predicted latent states. Only load trusted checkpoints. No future label/spectrum is needed for prediction.

## Verification and release status

After cleanup and renaming, all 37 rollout/inference tests passed again, and all seven CLI entry points passed their help checks. All data artifacts and reconstructed data paths remain present; their hashes and all 21 restored processed files were verified during data preparation. Record-level caches were subsequently repackaged into 32 MiB shards for reliable transfer; each reconstructed cache retains its original full-file SHA-256 hash. AST comparisons confirm that the core numerical class/function definitions are preserved after import-name normalization; edits reorganize names, imports and paths. Full multi-seed GPU training was not repeated. Removing baseline implementations means this repository is scoped to the proposed method and data, not independent reproduction of every comparison row.

Before citing a final public release, resolve the reference-target source and acquisition screening/geometry wording above, confirm the manuscript's Latent Only/Physics Only naming if retaining those claims, and replace the placeholder availability link with a real public repository/version. Exact provenance of all reported statistical tables is not certified by packaging checks. The manuscript itself has not been edited.

## License and citation

Software is licensed under **MIT**. Data licensing is separate and remains pending in `data/LICENSE`; software MIT does not automatically apply to observations. Author/title metadata are in `CITATION.cff`; no publication date or DOI is asserted.

`.gitattributes` tracks raw ZIPs and processed parts with Git LFS. Restored duplicate arrays and generated outputs are ignored. Git LFS storage/bandwidth limits may affect retrieval of the full payload. GitHub large-file documentation: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
