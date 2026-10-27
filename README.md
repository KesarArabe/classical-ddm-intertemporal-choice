# Classical Drift Diffusion Model Analysis of Intertemporal-Choice Data

A local, reproducible analysis applying a **classical Drift Diffusion Model
(DDM)** to the ITC Database of intertemporal choice (Pongratz & Schoemann). The
model is fitted separately within each of 12 eligible studies, evaluated on
choice and response-time (RT) prediction, cross-validated over participants,
and synthesised across studies.

This project contains the completed, verified analysis. It was reorganised from
an earlier working directory; the fits, cross-validation folds, parameter
estimates, diagnostics, and figures were **not** recomputed during the
reorganisation. Every result file has a descriptive, unique filename, and the
manifest hashes were regenerated to match the new layout.

## Status

**Complete: 12/12 study fits converged with positive-definite Hessians, 12/12
studies completed participant-level 5-fold cross-validation (5/5 folds each),
20/20 output-verification checks pass, and 18/18 unit tests pass.**

The frozen primary sample is 12 studies, 553 participants, 86,466 trials, and
621 sessions.

Headline results:

- **In-sample choice prediction:** AUC 0.588–0.836 (median 0.764); log loss
  0.419–0.677 (median 0.543).
- **In-sample RT:** quantile MAE 0.100–0.723 s (median 0.285 s).
- **Held-out, participant-level 5-fold CV:** pooled choice log loss
  0.444–0.701 (median 0.550); pooled AUC 0.553–0.829 (median 0.739). Only 10
  of 86,466 held-out trials (0.0116%) had zero density under the training-fold
  parameters.
- **Clustering matters:** participant-clustered SEs differ from naive
  trial-level SEs by a median design effect of 15.1 (range 0.6–140.4), so
  trial-level SEs would be badly overconfident.
- **Between-study heterogeneity is large:** random-effects I² is 0.92–0.99 for
  every parameter. The meta-analytic means (`a` 3.44, `t0` 0.354 s, `v0`
  0.095, `beta_amount` 0.225, `beta_delay` −0.225) describe this sample, not a
  universal DDM parameter set.

Three estimates sit close to an optimisation bound, where Hessian and sandwich
SEs are **not** valid confidence intervals: `t0` for Knauth_2022 (0.005 s below
its upper bound) and Lempert_2015 (0.008 s below), and `a` for Zhou_2022 (4.790
against an upper bound of 5.0).

The classical model overpredicts the slow RT tail in two studies. Lempert_2015
is the clearest case: predicted q90 is 3.86 s against 2.96 s observed, the
conditional-PIT KS statistic is 0.097, q90 is overpredicted for all 30
participants, and the data show no deadline or pile-up. Zhou_2022 overpredicts
the SS q90 by 2.07 s. A diagnostic investigation of Lempert_2015 (profile
likelihood, tail/CCDF, participant deviations) finds the misfit structural
rather than a single mis-estimated parameter; the model was not altered to
improve it.

## Main outputs

Reports:

- `results/reports/classical_ddm_itc_full_analysis_report.md` — full 12-study
  analysis report.
- `results/reports/classical_ddm_itc_lempert_2015_diagnostic_report.md` —
  Lempert_2015 misfit investigation.

Key tables:

- `results/tables/classical_ddm_itc_12_study_fit_summary.csv` — one row per
  study (fit, prediction, and convergence metrics).
- `results/tables/classical_ddm_itc_clustered_uncertainty_summary.csv` — long
  parameter table with naive and clustered SEs.
- `results/tables/classical_ddm_itc_5_fold_cross_validation_summary.csv` —
  held-out CV metrics; `..._folds.csv` and `..._assignments.csv` hold fold
  membership.
- `results/tables/classical_ddm_itc_random_effects_meta_analysis.csv` and
  `classical_ddm_itc_meta_analysis_descriptive.csv` — cross-study synthesis.
- `results/tables/classical_ddm_itc_analysis_checks.json` — analysis-level
  pass/fail checks.
- `results/classical_ddm_itc_data_audit.json`, `..._validation.json`,
  `..._study_profile.csv`, `..._parameter_recovery_trials.csv`,
  `..._parameter_recovery_summary.json` — audit, numerical validation, sample
  profile, and synthetic recovery.

Figures (all in `results/figures/`):

- `classical_ddm_itc_study_parameter_forest_plot.png`
- `classical_ddm_itc_study_summary_metrics.png`
- `classical_ddm_itc_parameter_recovery.png`
- For each of the 12 studies:
  `classical_ddm_itc_<study>_reaction_time_quantiles.png`,
  `..._choice_calibration.png`, `..._choice_by_amount_delay.png`,
  `..._reaction_time_tail.png`, `..._pit_calibration.png`.

Per-study estimates and diagnostics are in `results/fits/`, the Lempert_2015
investigation in `results/lempert_2015/`, the earlier two-study pilot in
`results/pilot/`, and run logs in `logs/`. The logs are preserved from the
original run and therefore still contain the original absolute paths. The
manifest `results/classical_ddm_itc_manifest.json` records the SHA-256 hash and
size of every result file.

## Layout

```text
README.md                        this file
requirements.txt                 pinned Python dependencies
.gitignore                       excludes the raw CSV and caches
docs/
  methodology.md                 model, likelihood, evaluation, validation, CV
  decisions.md                   decision log
  dataset_audit.md               dataset structure, quality, frozen sample
  dataset_provenance.md          source, licence, download, local path
  limitations.md                 scope and known limitations
src/
  data/                          loading, provenance, cleaning, predictors
  models/                        classical DDM likelihood and simulation
  fitting/                       optimisation and clustered uncertainty
  evaluation/                    metrics, diagnostics, PIT/CDF, CV, meta-analysis
experiments/classical_ddm/
  config.json                    frozen analysis configuration
  run_full.py                    full analysis (supports --report-only)
  run_pilot.py                   two-study pilot
  investigate_lempert.py         Lempert_2015 diagnostic (supports --report-only)
  verify_outputs.py              integrity and consistency checks
tests/                           mathematical, numerical, analysis validation
results/                         all completed outputs (see above)
logs/                            run logs from the completed analysis
```

## Data

The analysis uses the ITC Database:

- Pongratz, H., & Schoemann, M. *A large-scale dataset of choice and
  response-time data in intertemporal choice* (preprint, manuscript v2.0).
- OSF project: <https://osf.io/3wsae/>
- Licence: **CC BY-NC-SA 4.0** (attribution, non-commercial, share-alike).
- Local file (outside this project): `../data/itc_all_data.csv`, i.e.
  `Default Project/data/itc_all_data.csv`.
- Verified SHA-256:
  `14e5bf2f6434d791a5f68d8f0e4cec41e4c6bb671ce6fd80e83879f9759e0517`
- Verified size: `484,996,376` bytes.

The raw CSV is **not** copied into this project and is excluded from version
control. To use a copy stored elsewhere, set the environment variable:

```powershell
$env:DDM_ITC_CSV = 'D:\data\itc_all_data.csv'
```

The loader checks the file size on every load and the SHA-256 on verification.
See `docs/dataset_provenance.md` for download and licence details.

## Design in one paragraph

The primary sample is a strict 12-study subset: simultaneous presentation, no
response deadline, no mouse/finger tracking, RTs recorded with sub-second
precision, RTs previously analysed by the original authors, and valid
amount/delay values. The classical DDM is fitted **separately within each
study** so that differences in reward scale and RT range are not forced into a
single pooled parameter set. The model estimates a boundary separation, a
non-decision time, a baseline drift, and amount/delay drift sensitivities.
Starting point and diffusion noise are fixed (midpoint and 1, respectively).
The model is evaluated on choice prediction, RT prediction, and goodness of
fit; "accuracy" is not defined for this preference task. See
`docs/methodology.md`.

## Reproduction

Run from this project directory (`classical-ddm-intertemporal-choice/`) using
the existing environment at `Default Project/.venv`. No new packages are
required.

```powershell
# 1. Unit tests (mathematical, numerical, analysis validation)
..\.venv\Scripts\python.exe -m unittest discover -s tests -v

# 2. Verify the saved results: manifest hashes, inventory, frozen sample,
#    bounds, clustered covariances, CV coverage, and stored likelihoods.
#    This does not refit anything.
..\.venv\Scripts\python.exe experiments\classical_ddm\verify_outputs.py

# 3. Rebuild the full report and manifest from saved artifacts (no refit)
..\.venv\Scripts\python.exe experiments\classical_ddm\run_full.py --report-only

# 4. Rebuild the Lempert_2015 report from its saved JSON (no refit)
..\.venv\Scripts\python.exe experiments\classical_ddm\investigate_lempert.py --report-only
```

A full recomputation is possible but not required to inspect the saved results:

```powershell
# Full 12-study analysis (refits everything; deterministic given config seeds)
..\.venv\Scripts\python.exe experiments\classical_ddm\run_full.py

# Optional: re-run the two-study pilot
..\.venv\Scripts\python.exe experiments\classical_ddm\run_pilot.py

# Optional: re-run only the Lempert_2015 investigation (refits that study)
..\.venv\Scripts\python.exe experiments\classical_ddm\investigate_lempert.py
```

`run_full.py` already runs the Lempert investigation; the standalone script
exists so that investigation can be re-run or inspected on its own. If the
pilot or the Lempert investigation is re-run, follow it with
`run_full.py --report-only` and then `verify_outputs.py` so the manifest picks
up every changed result file.

The full run is deterministic given the fixed seeds in
`experiments/classical_ddm/config.json`: multi-start fitting, PPC simulation,
CV fold assignment, and PIT grids all use fixed seeds. The manifest records the
environment (Python, platform, NumPy, pandas), settings, and file hashes.

## Scope and limitations

The main limitations are summarised in `docs/limitations.md`: study-level (not
hierarchical) fits, a lower RT truncation that bounds `t0`, coarse RT recording
in some studies, heavy-tailed RTs that the classical model does not always
capture, three boundary-adjacent estimates whose SEs are not valid CIs, and
imprecise between-study variance with only 12 studies. Cross-validation tests
generalisation to new participants within a study, not to a new study or task
design.

## Licence

The source data are CC BY-NC-SA 4.0. Any redistribution of the data or adapted
material must comply with that licence. The analysis code is provided for
research use; add your own code licence before public release if you want one.
