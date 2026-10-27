# Methodology: classical DDM

## 1. Model

For trial \(i\), the evidence process is a Wiener diffusion process with
constant boundaries and trial-specific drift:

\[
dX_i(t) = v_i\,dt + dW_i(t),
\]

with absorbing boundaries at \(0\) and \(a\), starting point \(a/2\), and
unit diffusion noise. The observed response time is

\[
RT_i = t_0 + T_i,
\]

where \(T_i\) is the first-passage time and \(t_0\) is non-decision time.
The upper boundary corresponds to the larger-later (LL) choice; the lower
boundary corresponds to the smaller-sooner (SS) choice.

Drift varies across trials with the displayed amounts and delays:

\[
v_i = v_0 + \beta_{\text{amount}}\,x_{\text{amount},i}
      + \beta_{\text{delay}}\,x_{\text{delay},i}.
\]

The predictors are:

\[
x_{\text{amount},i} = \operatorname{standardise}
\left(\log\frac{LL_i}{SS_i}\right),
\qquad
x_{\text{delay},i} = \operatorname{standardise}
\left(\log\frac{LL_i+1}{SS_i+1}\right).
\]

The \(+1\) in the delay ratio handles immediate options (`ss_time = 0`) while
preserving the ordering. Standardisation is done within study (within training
data when cross-validating).

This is a classical DDM with trial-varying drift. It deliberately fixes:

- starting point at the midpoint (`z = 0.5`),
- diffusion noise at 1,
- constant boundaries within a trial,
- no across-trial variability in drift, starting point, or non-decision time.

Those restrictions define the classical model. Later experiments will relax
them in separate, established formulations.

## 2. Parameters

| Parameter | Meaning | Bounds |
|---|---|---|
| `a` | boundary separation | 0.2 to 5.0 |
| `t0` | non-decision time (s) | 0.001 to min(RT) - 0.005 |
| `v0` | baseline drift | -20 to 20 |
| `beta_amount` | effect of amount ratio on drift | -20 to 20 |
| `beta_delay` | effect of delay ratio on drift | -20 to 20 |

Positive `beta_amount` means a larger LL/SS amount ratio increases drift toward
LL. Negative `beta_delay` means a longer LL delay decreases drift toward LL.

## 3. Likelihood

For each trial, the likelihood is the joint density of the observed choice and
RT:

\[
L_i = f_{c_i}(RT_i - t_0 \mid v_i, a),
\]

where \(f_{\text{upper}}\) and \(f_{\text{lower}}\) are the first-passage
densities at the upper and lower boundaries. Both densities are evaluated with
a **hybrid expansion**:

- for short decision times (\(t < a^2/2\)) the reflection series
  (Anderson, 1960; Navarro & Fuss, 2009) is used;
- for long decision times (\(t \ge a^2/2\)) the sine/eigenfunction series is
  used.

The split is necessary. The reflection series is exact but suffers
catastrophic cancellation when \(t \gg a^2\): in double precision it can sum
to zero or a negative number even though the true density is positive. The
eigenfunction series has the opposite behaviour. Both branches were checked
against 140-digit arithmetic in the overlap and in the long-time tail; see
`tests/test_likelihood.py`. This bug was found during the pilot and fixed
before the reported fits.

The log-likelihood is summed over trials within a study. Parameters are
estimated by maximum likelihood using L-BFGS-B with five starting values.
Standard errors are obtained from a numerical Hessian at the optimum, and the
spread of the final objective across starts is reported as a local-optimum
check.

### 3.1 Participant-clustered uncertainty

Trials from the same participant are not independent. The Hessian-based
standard error treats them as if they were, so it is reported only as a naive
benchmark. The primary uncertainty estimate is a participant-clustered
sandwich covariance:

\[
\widehat{\mathrm{Cov}}(\hat\theta)
= \frac{G}{G-1}\,H^{-1}
\left(\sum_{g=1}^{G} s_g s_g^\top\right) H^{-1},
\]

where \(H\) is the numerical Hessian of the summed negative log-likelihood,
\(s_g\) is the sum of the per-trial score contributions for participant \(g\),
and \(G\) is the number of participants. The factor \(G/(G-1)\) is a
small-sample correction. Per-trial scores are central differences of the
validated per-trial log-likelihood at the fitted parameters.

The sandwich estimator accounts for within-participant correlation but **not**
for the `t0` boundary constraint or for model misspecification. When `t0` sits
within 5% of its upper bound, both the naive and clustered `t0` standard errors
are reported as invalid confidence intervals.

## 4. Why study-level fits first

The 12 primary studies differ in reward scale, delay range, and RT scale. A
single pooled parameter set would confound those differences with the model
parameters. Fitting separately within each study:

- respects the study designs,
- makes the 12 fits independent replications,
- keeps the model simple enough to validate,
- allows study-level parameter estimates to be combined later.

Participant-level fits and a pooled hierarchical model are later extensions.

## 5. Evaluation

### Choice prediction

- log loss
- Brier score
- AUC for LL choices

Choice probabilities are computed from the analytic constant-drift expression
\(P(\text{LL}) = 1 / (1 + \exp(-v a))\), which follows from the same model.

### RT prediction

- held-out log predictive density (conditional on choice and marginal)
- mean absolute error of RT quantiles
- coverage of central prediction intervals
- observed versus predicted RT quantiles (0.1, 0.3, 0.5, 0.7, 0.9)
- probability integral transform (PIT) calibration
- model-implied quantiles from numerical integration, which avoid Monte Carlo
  noise

The PIT is \(P(T \le RT \mid \text{choice})\) under the fitted model. If the
model is correct it is uniform on \((0,1)\). It is reported overall and
separately for SS and LL choices, together with a Kolmogorov-Smirnov statistic
and binned deviations. With thousands of trials the KS test is sensitive to
small departures, so the mean PIT and maximum bin deviation are interpreted
alongside the p-value.

Model-implied quantiles are computed by integrating the joint density on a
log-spaced decision-time grid, normalising by the analytic choice probability
for the conditional distributions, and inverting the resulting CDF. The
integration uses 12 series terms; comparisons against 100 terms agree to better
than \(10^{-8}\) in integrated mass over the fitted parameter range.

Within a trial, constant drift and a midpoint start make choice and RT
independent. Across trials, an SS/LL RT difference can arise only through
variation in drift. The model is therefore expected to capture only the part
of the choice-RT association that the two predictors explain.

### Goodness of fit

- posterior predictive checks using simulated datasets,
- choice proportions by amount/delay bins,
- RT distributions by choice,
- parameter plausibility and convergence diagnostics.

### What is not evaluated

"Accuracy" is not defined: both options are valid preferences, not correct and
incorrect responses. The report will distinguish choice prediction from
accuracy and will not describe one as the other.

## 6. Validation plan

1. **Analytic comparison.** Compare the implemented density with PyDDM's
   analytical solver over a grid of drift, boundary, and RT values.
2. **High-precision comparison.** Check both expansions against 140-digit
   decimal arithmetic, including the long-time regime where the reflection
   series cancels in double precision.
3. **Normalisation.** Numerically integrate the two choice densities and check
   that they sum to 1 and that the upper integral matches the analytic choice
   probability.
4. **Simulation.** Simulate choices and RTs from known parameters and compare
   empirical choice proportions and RT quantiles with the analytic model. The
   Euler-Maruyama step is 0.0001 s; at 0.001 s the upper RT quantiles are
   biased upward by about 0.1 s.
5. **Parameter recovery.** Fit simulated datasets with known parameters and
   check bias and relative error.
6. **Optimiser checks.** Fit the same data from five starting values and
   check agreement, convergence, and Hessian conditioning.
7. **Pilot.** Fit two real studies, inspect parameter plausibility, and check
   runtime before the full 12-study run.

## 7. Cross-validation

The full experiment uses 5-fold cross-validation by participant within each
study: fit on 80% of participants and predict the held-out 20%. Folds are
assigned over participants, never over trials, and are greedily balanced by
trial count. No participant can appear in both training and test data in the
same fold.

The amount and delay predictors are standardised using the training
participants only; those means and SDs are then applied unchanged to the
held-out participants. The model is refitted from scratch in each fold.

Held-out evaluation uses:

- choice log loss, Brier score, and AUC;
- the conditional RT log score \(\log f_c(RT \mid \text{choice}) - \log P(c)\);
- the marginal RT log score \(\log(f_{\text{upper}} + f_{\text{lower}})\);
- conditional PIT calibration;
- model-implied quantile MAE.

Fold-level means and SDs are reported, together with metrics pooled over all
held-out trials. Cross-validation tests generalisation to new participants
within the same study; it does not test transfer to a new study or task design.

## 8. Cross-study synthesis

Study-level estimates are combined descriptively and with a
DerSimonian-Laird random-effects meta-analysis. The synthesis uses the
participant-clustered standard errors. It is a description of heterogeneity
across separately fitted studies, not a pooled re-fit, and it does not
propagate participant-level clustering beyond the study-level standard errors.

## 8a. Diagnostic investigation of Lempert_2015

The investigation is descriptive and makes no change to the model, the
cleaning rules, or the fitted parameters. It reports:

- observed versus posterior-predictive RT survival curves overall and by
  choice (CCDF), plus proportions above fixed thresholds;
- deadline and pile-up checks (maximum RT, proportion above 5 s, most common
  0.01-s RT bin);
- observed versus model-implied q50/q90/q95 for every participant;
- a profile likelihood over the boundary separation `a`.

The profile grid is scaled to the Hessian standard error rather than fixed a
priori, because a fixed span is either too coarse to resolve a 95% interval or
needlessly wide. The grid spans `a_hat +/- 8 SE_a` with 15 points (spacing
about 1.1 SE), always includes `a_hat` and the two grid edges, and refits the
remaining four parameters at each value of `a`. The 95% interval is obtained by
linear interpolation of the profile deviance where it crosses the
chi-squared(1) threshold of 3.841. If the crossing falls at a grid edge the
interval is reported as grid-bounded, which indicates weak identification
rather than a precise estimate.

## 9. Limitations to report

The consolidated list, including known fit problems and inference caveats, is
in `docs/limitations.md`. The main points are:

- No correct answer, so no accuracy.
- Lower RT truncation at 0.2 s bounds `t0` from below.
- Upper RT truncation at 20 s removes a small tail.
- Study-level pooling ignores participant-level parameter variation.
- Repeated sessions are treated as exchangeable trials, with participant
  clustering in uncertainty estimates.
- The classical DDM may misfit long, heavy-tailed intertemporal-choice RTs.
  This is itself a result, not a reason to hide the misfit.
- The fastest observed RT is a hard upper bound on `t0`. In several studies
  the estimate sits close to that bound, so `t0` is not freely identified. The
  natural relaxation is across-trial variability in `t0`.
- The linear drift in standardised log amount and log delay ratios is a
  modelling choice, not a law. The calibration plots show departures from it,
  especially in Lempert_2015.
- Random-effects meta-analysis with 12 studies gives an imprecise estimate of
  between-study variance. The meta-analytic mean is not a universal DDM
  parameter.
- Participant-level cross-validation uses study-specific models. It says
  nothing about out-of-study transfer.
- The PIT and posterior predictive checks are in-sample diagnostics; the
  held-out log scores and PIT are the predictive checks.
