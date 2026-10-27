# Limitations and known issues

This document records the scope of the completed analysis and the limitations
that should travel with any use of its results. It is a description of the
frozen analysis, not a list of planned changes.

## 1. Scope

- The analysis applies **one** model: a classical DDM with trial-varying drift,
  a constant boundary, a midpoint start point, unit diffusion noise, and no
  across-trial parameter variability. No alternative or extended DDM is fitted
  here.
- The primary sample is the strict 12-study subset defined in
  `dataset_audit.md`. The broader ~60-study sample in the database is not used.
- Fits are **study-level**. A hierarchical or participant-level parameter model
  is not part of this analysis.
- The task has no correct answer, so "accuracy" is undefined. Choice
  prediction (log loss, Brier score, AUC) and RT prediction (quantile error,
  log score, PIT calibration) are the outcome measures.
- Cross-validation is participant-level and within-study. It does not test
  transfer to a new study, a new task, or a different RT-screening convention.

## 2. Data limitations

- **Lower RT truncation.** Trials with `rt < 0.2 s` are removed. This bounds
  `t0` from below and means the analysis cannot identify very short
  non-decision times. A different lower bound is a sensitivity analysis, not
  part of this run.
- **Upper RT truncation.** Trials with `rt > 20 s` are removed. This removes a
  small heavy tail and can slightly affect tail diagnostics.
- **Coarse RT recording.** Studies recording RTs only to whole seconds are
  excluded, but the retained studies still differ in timing precision.
- **Heterogeneous designs.** The 12 studies differ in reward scale, delay
  range, trial counts, session structure, and instructions. Study-level
  parameters are therefore not directly comparable in absolute units.
- **Repeated sessions.** `Bruder_2021` contributes 34 participants with three
  sessions each (621 participant-sessions in total). Sessions are treated as
  exchangeable trials, and uncertainty is clustered by participant.
- **Missing trials.** The database records relative missingness, but the
  analysis does not model it. If missingness is related to RT or choice, the
  estimates could be biased.
- **Secondary data.** The dataset is a compilation of 100 studies, not a single
  pre-registered experiment. Cleaning rules were fixed before the full run and
  are documented, but the sample remains opportunistic.

## 3. Model limitations

- **Fixed start point and noise.** The midpoint start and unit noise are
  identifying restrictions, not estimates. They rule out the choice-RT
  association that a non-midpoint start could produce within a trial.
- **No across-trial variability.** Drift, start point, and non-decision time
  are fixed within a study (drift varies only through the two predictors).
  This is the classical model and is expected to underpredict slow-tail
  dispersion.
- **Linear drift in standardised predictors.** The log amount ratio and log
  delay ratio enter linearly. Calibration plots show departures from this
  specification, especially in Lempert_2015.
- **Boundary constraints.** The fastest observed RT is a hard upper bound on
  `t0`. In several studies the estimate sits close to that bound, so `t0` is
  not freely identified.
- **Single boundary pair.** The model assumes one boundary separation per
  study; it does not allow boundary changes with amount, delay, or time.

## 4. Estimation and inference limitations

- **Three boundary-adjacent estimates.** Hessian and clustered sandwich SEs are
  **not valid confidence intervals** when a parameter sits at or near a bound:
  `t0` for `Knauth_2022` (0.005 s below the upper bound), `t0` for
  `Lempert_2015` (0.008 s below), and `a` for `Zhou_2022` (4.790 against an
  upper bound of 5.0). These are reported as flags, not corrected post hoc.
- **Clustered SEs address clustering only.** The participant-clustered sandwich
  estimator accounts for within-participant correlation; it does not account
  for the boundary constraint or for model misspecification.
- **Design effects are large.** The median ratio of clustered to naive SE is
  15.1 (range 0.6–140.4). Naive trial-level SEs would be badly overconfident.
- **Meta-analysis with 12 studies.** The DerSimonian–Laird between-study
  variance is imprecise with 12 units. I² is 0.92–0.99 for every parameter, so
  the meta-analytic mean is a description of this heterogeneous sample, not a
  universal DDM parameter set.
- **Multi-start optimisation is local.** Five starts (two for CV folds) reduce
  but do not eliminate the risk of local optima. Start-objective spread,
  convergence status, and Hessian conditioning are reported for every fit.

## 5. Fit problems found (reported, not hidden)

- **Lempert_2015 slow-tail misfit.** The model predicts q90 = 3.86 s against
  2.96 s observed. The conditional-PIT KS statistic is 0.097, q90 is
  overpredicted for all 30 participants, and there is no deadline or pile-up
  evidence. The profile likelihood over `a` (95% interval [2.714, 2.762],
  `a_hat` = 2.738) indicates that changing the boundary alone does not remove
  the misfit. The investigation is diagnostic only; the model was not altered.
- **Zhou_2022 SS tail.** The model overpredicts the smaller-sooner q90 by
  2.07 s. This is a second instance of the same structural slow-tail
  overprediction.
- **Zero-density held-out trials.** Ten of 86,466 held-out trials (0.0116%)
  fell in a region with zero density under the training-fold parameters. They
  are counted and reported rather than silently dropped; their contribution to
  pooled log loss is handled explicitly in
  `classical_ddm_itc_5_fold_cross_validation_folds.csv`.

## 6. Reproducibility caveats

- The raw dataset is not included. Reproduction requires the exact file
  described in `dataset_provenance.md` (size and SHA-256 are checked).
- Results were produced with Python 3.10.11, NumPy 2.2.6, SciPy 1.15.3,
  pandas 2.3.3, Matplotlib 3.10.9, tabulate 0.10.0, and PyDDM 0.9.0 on
  Windows. Different BLAS/LAPACK builds or optimiser versions can change
  numerical results at the margin.
- The saved results in this project were **not** refitted during
  reorganisation. `verify_outputs.py` recomputes every stored log-likelihood
  and checks all manifest hashes, which validates the saved numbers but is not
  the same as a fresh full refit.
- The pilot in `results/pilot/` is an earlier two-study run and is superseded
  by the full analysis. It is retained for provenance only.
