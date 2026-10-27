# Decision log

## D001 — Ignore the previous custom model

The previous exponential-decay DDM project, its hypotheses, custom equation,
and results are not used. Only the raw ITC CSV is reused, and it was verified
independently.

## D002 — Use the existing CSV by reference

The 485 MB CSV remains at `../data/itc_all_data.csv` (outside the project
directory). It is not copied into this project. The loader verifies its SHA-256
on demand and checks its size on every load. A `DDM_ITC_CSV` environment
variable can point to an equivalent copy elsewhere. This avoids duplicating
licensed data and keeps the new project small.

## D003 — Primary sample

Use the strict 12-study sample:

- no response deadline,
- simultaneous presentation,
- no mouse/finger tracking,
- RTs not rounded to whole seconds,
- RTs previously analysed by the original authors,
- valid positive amounts and `ss_time < ll_time`,
- RT between 0.2 and 20 s,
- compiler subject/trial exclusions applied.

This gives 12 studies, 553 participants, and 86,466 trials. The broader
60-study sample is reserved for a later sensitivity/generalisation check.

## D004 — Study-level fits first

The classical DDM is fitted separately within each study. This respects
differences in reward scale, delay range, and RT scale, and treats the studies
as replications. A pooled or hierarchical model is a later extension, not part
of the first experiment.

## D005 — Fixed starting point and noise

The classical model fixes the starting point at the midpoint and diffusion
noise at 1. This avoids weak identification between drift, starting point, and
boundary and keeps the first model simple. Across-trial variability in starting
point is part of a later "full DDM" experiment.

## D006 — Trial-varying drift

Drift is a linear function of within-study standardised log amount ratio and
log delay ratio. No discount function or custom exponential-decay equation is
used. This keeps the drift specification model-agnostic and directly tied to
the displayed stimuli.

## D007 — Sessions

All eligible sessions are retained. The 34 participants in `Bruder_2021` with
three sessions contribute all three sessions. Uncertainty will be clustered by
participant, and a first-session-only analysis is reserved as a sensitivity
check. This decision avoids discarding 6,520 valid trials in the pilot.

## D008 — Validation before full fit

The full 12-study fit will not start until:

- the density matches PyDDM's analytical solver,
- numerical integration and choice probabilities agree,
- synthetic parameter recovery succeeds,
- the two-study pilot converges and gives plausible parameters,
- runtime is measured and reported.

## D009 — GitHub is manual

No Git repository, commit, push, or GitHub operation is performed by this
project. All work stays local.

## D010 — Hybrid first-passage density

The reflection series (Anderson/Navarro-Fuss) is exact but numerically
unstable for decision times much larger than the squared boundary. During the
pilot it summed to zero for slow trials, which made the likelihood jagged and
caused L-BFGS-B to stop at a spurious local optimum (Knauth_2022, `a` = 1.54
instead of 2.05).

The implementation now uses the reflection series for `t < a^2/2` and the
sine/eigenfunction series for `t >= a^2/2`, with the latter evaluated in log
space. Both branches were checked against 140-digit arithmetic. A regression
test covers the long-time tail.

## D011 — Simulation step

Euler-Maruyama with `dt = 0.001` biased the upper RT quantiles upward by about
0.1 s at the pilot parameters, which inflated the posterior predictive
coverage. All simulation-based checks now use `dt = 0.0001` (bias about
0.02 s), documented in `config.json`.

## D012 — Pilot outcome

The likelihood, recovery, and optimiser checks pass. Knauth_2022 fits well.
Lempert_2015 shows a genuine misfit in the slow RT tail: the model predicts a
90th percentile near 3.8 s against an observed 2.95 s. The fit is stable and
the coverage check is only marginally inside its ceiling, so this is treated
as a substantive result and as motivation for the next model (across-trial
variability in drift and non-decision time), not as a reason to weaken the
check.

In both studies `t0` sits within 1-2% of its upper bound (the fastest observed
RT). Its Hessian-based standard error is therefore not a valid confidence
interval. This is reported as a limitation, not corrected by post-hoc
adjustment.

## D013 — Full analysis is study-specific and separately fitted

The full 12-study analysis keeps the frozen sample and cleaning rules. Each
study is fitted separately with the same five-parameter classical model.
Cross-study synthesis uses study-level estimates and their standard errors; it
is not a pooled re-fit. No parameter is shared across studies during
estimation.

## D014 — Participant-clustered uncertainty is the primary SE

The Hessian-based SE is retained as a naive benchmark. The reported
uncertainty is a participant-clustered sandwich estimator with per-participant
score sums and the numerical Hessian, with a \(G/(G-1)\) correction. This
addresses within-participant correlation from repeated trials and sessions. It
does not address the `t0` boundary or model misspecification, and the report
says so explicitly.

## D015 — Participant-level 5-fold CV

Cross-validation folds are defined over `subj_ident`, balanced greedily by
trial count, with a fixed seed. The predictor scaler is fitted on training
participants only. This prevents trials or predictor-scaling information from
the same participant leaking into the held-out set. Held-out performance is
scored with proper scoring rules plus PIT calibration.

## D016 — PIT calibration and model-implied quantiles

In addition to simulation-based posterior predictive checks, the analysis
computes conditional PIT values by numerically integrating the validated
density. Model-implied quantiles are obtained by integrating and inverting the
CDF rather than by simulation, which removes Monte Carlo noise from the RT
diagnostics. The integration uses 12 series terms; this agrees with 100 terms
to better than \(10^{-8}\) in integrated mass over the fitted range.

## D017 — Lempert_2015 investigation is diagnostic only

The Lempert_2015 investigation examines tail probabilities, deadline/pile-up
evidence, participant-level deviations, and a profile likelihood over `a`. No
model term, cleaning rule, or fitted parameter was changed to improve the fit.
The investigation is reported as evidence about the structural limitations of
the classical model.

## D018 — Verification by manifest and recomputation

The full run writes `results/classical_ddm_itc_manifest.json` with SHA-256
hashes for every output file. `verify_outputs.py` rechecks the hashes, the
expected inventory, the frozen sample size, parameter bounds,
clustered-covariance validity, CV fold coverage, and recomputes every stored
log-likelihood from the saved parameters. A run is not reported as complete
unless these checks pass.

## D019 — Reorganisation without refitting

The completed analysis was reorganised into `classical-ddm-intertemporal-choice`
with descriptive output filenames (`classical_ddm_itc_*`). No model was
refitted and no cross-validation was rerun. Code paths, report links, README
commands, and docs were updated, and the manifest hashes were regenerated from
the renamed files with `run_full.py --report-only`. The original working
directory was kept until the reorganised project passed its tests and output
verification. The raw CSV remains outside the project.

