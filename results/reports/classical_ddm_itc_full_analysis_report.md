# Classical DDM: full 12-study analysis

This report covers the complete frozen primary sample. Every study was fitted separately; no parameters were pooled across studies during estimation. Cross-validation folds are defined over participants, and uncertainty is reported both naively and with a participant-clustered sandwich estimator.

## 1. Inputs and frozen sample

- Dataset: `C:\Users\ARAVE\OneDrive\Documents\Default Project\data\itc_all_data.csv`
- Size: 484,996,376 bytes
- SHA-256 verified: **True**
- Licence: CC BY-NC-SA 4.0
- Primary sample: **12 studies, 553 participants, 86,466 trials, 621 sessions**
- Cleaning rules and the 12-study sample are unchanged from the frozen plan (see `docs/dataset_audit.md` and `docs/decisions.md`).

## 2. Validation and recovery

- PyDDM density-grid maximum absolute error: 1.776e-15.
- Normalisation checks: v=0.0, a=1.0 total error -4.4e-16; v=0.5, a=1.2 total error -4.4e-16; v=2.0, a=2.0 total error -6.7e-16; v=-1.0, a=0.8 total error -4.4e-16.
- Simulation check: choice-proportion difference +0.006; maximum absolute RT-quantile difference 0.064 s.

Synthetic recovery (5 datasets, 3,000 trials each) is summarised below. All parameters recover well; the largest relative RMSE is for `v0`, which is expected because baseline drift is partly traded off against the amount and delay slopes.

| parameter   |   true |   mean_estimate |   bias |   rmse |   relative_rmse |
|:------------|-------:|----------------:|-------:|-------:|----------------:|
| a           |    1.2 |           1.212 |  0.012 |  0.018 |           0.015 |
| t0          |    0.3 |           0.301 |  0.001 |  0.002 |           0.007 |
| v0          |    0.2 |           0.214 |  0.014 |  0.03  |           0.148 |
| beta_amount |    0.8 |           0.8   | -0     |  0.013 |           0.016 |
| beta_delay  |   -0.6 |          -0.578 |  0.022 |  0.038 |           0.063 |

## 3. Convergence and boundary diagnostics

| paper               |   trials |   subjects | converged   | hessian_positive_definite   |   hessian_condition |   start_objective_spread |   runtime_seconds |     t0 |   t0_margin_seconds | t0_at_upper_5pct   | params_at_5pct   |
|:--------------------|---------:|-----------:|:------------|:----------------------------|--------------------:|-------------------------:|------------------:|-------:|--------------------:|:-------------------|:-----------------|
| Bixter_2015_Exp1    |     9333 |         56 | True        | True                        |             14.5461 |                        0 |           20.7341 | 0.0956 |              0.0999 | False              |                  |
| Bixter_2015_Exp2    |     4652 |         28 | True        | True                        |             18.072  |                        0 |            9.7181 | 0.0233 |              0.173  | False              |                  |
| Bruder_2021         |     9784 |         34 | True        | True                        |             43.0952 |                        0 |           25.0073 | 0.329  |              0.025  | False              |                  |
| Gluth_2017_Exp1     |     1236 |         21 | True        | True                        |             19.24   |                        0 |            3.2475 | 0.8471 |              0.1729 | False              |                  |
| Gluth_2017_Exp2     |     1763 |         31 | True        | True                        |             21.8994 |                        0 |            4.676  | 0.8227 |              0.1183 | False              |                  |
| Herting_2010        |     3169 |         29 | True        | True                        |             12.2402 |                        0 |            8.5512 | 1.131  |              0.101  | False              |                  |
| Knauth_2022         |     8689 |         35 | True        | True                        |            233.829  |                        0 |           24.8586 | 0.1965 |              0.0051 | True               | t0               |
| Konstantinidis_2020 |    14759 |         40 | True        | True                        |             14.7344 |                        0 |           25.0451 | 0.022  |              0.173  | False              |                  |
| Lempert_2015        |    10589 |         30 | True        | True                        |            155.366  |                        0 |           25.3164 | 0.4621 |              0.0079 | True               | t0               |
| Robinson_2015       |     8520 |         36 | True        | True                        |             17.5029 |                        0 |           15.7958 | 0.1341 |              0.0616 | False              |                  |
| Zhou_2022           |     2604 |         53 | True        | True                        |             38.2876 |                        0 |            6.5906 | 0.1024 |              0.1096 | False              | a                |
| Zilker_2021         |    11368 |        160 | True        | True                        |             20.2922 |                        0 |           24.2864 | 0.1479 |              0.0983 | False              |                  |

- Converged fits: **12/12**.
- Positive-definite Hessians: **12/12**.
- Studies with `t0` within 5% of its upper bound: **2/12**.
- Studies with any parameter within 5% of a bound: **3** (see `params_at_5pct` and `results/fits/*_parameters.csv`).
- Failed fits: none.

Where `t0` is at or near the bound, its Hessian-based SE is **not** a valid confidence interval. The clustered SE has the same caveat; it accounts for participant clustering, not for the boundary constraint.

## 4. Parameter estimates

| paper               |     a |    t0 |     v0 |   beta_amount |   beta_delay |   clustered_se_a |   clustered_se_t0 |   clustered_se_v0 |   clustered_se_beta_amount |   clustered_se_beta_delay |
|:--------------------|------:|------:|-------:|--------------:|-------------:|-----------------:|------------------:|------------------:|---------------------------:|--------------------------:|
| Bixter_2015_Exp1    | 2.863 | 0.096 |  0.241 |         0.139 |       -0.306 |            0.073 |             0.025 |             0.063 |                      0.018 |                     0.03  |
| Bixter_2015_Exp2    | 3.067 | 0.023 |  0.538 |         0.127 |       -0.248 |            0.12  |             0.038 |             0.079 |                      0.021 |                     0.035 |
| Bruder_2021         | 3.294 | 0.329 |  0.24  |         0.361 |       -0.34  |            0.117 |             0.02  |             0.067 |                      0.028 |                     0.028 |
| Gluth_2017_Exp1     | 4.337 | 0.847 |  0.033 |         0.083 |       -0.1   |            0.181 |             0.049 |             0.034 |                      0.03  |                     0.028 |
| Gluth_2017_Exp2     | 4.622 | 0.823 |  0.017 |         0.067 |       -0.046 |            0.166 |             0.067 |             0.033 |                      0.028 |                     0.029 |
| Herting_2010        | 3.793 | 1.131 |  0.126 |         0.3   |       -0.228 |            0.164 |             0.048 |             0.048 |                      0.026 |                     0.022 |
| Knauth_2022         | 2.047 | 0.196 |  0.258 |         0.413 |       -0.171 |            0.061 |             0.005 |             0.085 |                      0.102 |                     0.039 |
| Konstantinidis_2020 | 2.798 | 0.022 | -0.122 |         0.264 |       -0.393 |            0.123 |             0.026 |             0.074 |                      0.019 |                     0.031 |
| Lempert_2015        | 2.738 | 0.462 |  0.03  |         0.484 |       -0.313 |            0.063 |             0.007 |             0.087 |                      0.029 |                     0.026 |
| Robinson_2015       | 2.663 | 0.134 |  0.15  |         0.09  |       -0.32  |            0.079 |             0.034 |             0.036 |                      0.03  |                     0.026 |
| Zhou_2022           | 4.79  | 0.102 | -0.274 |         0.225 |       -0.238 |            0.173 |             0.038 |             0.045 |                      0.02  |                     0.024 |
| Zilker_2021         | 4.428 | 0.148 | -0.012 |         0.235 |       -0.008 |            0.084 |             0.056 |             0.022 |                      0.012 |                     0.003 |

Point estimates are maximum likelihood. The `clustered_se_*` columns are participant-clustered sandwich standard errors; the unclustered SEs and design effects are in `results/tables/classical_ddm_itc_clustered_uncertainty_summary.csv`. Design effects well above 1 show that trial-level SEs would be badly overconfident.

## 5. Choice prediction

| paper               |   choice_log_loss |   brier |    auc |   observed_ll |   predicted_ll |
|:--------------------|------------------:|--------:|-------:|--------------:|---------------:|
| Bixter_2015_Exp1    |            0.5659 |  0.1937 | 0.7431 |        0.6386 |         0.6398 |
| Bixter_2015_Exp2    |            0.4191 |  0.1331 | 0.7658 |        0.8123 |         0.8105 |
| Bruder_2021         |            0.4807 |  0.1581 | 0.8349 |        0.6205 |         0.63   |
| Gluth_2017_Exp1     |            0.6606 |  0.2349 | 0.6191 |        0.5485 |         0.5397 |
| Gluth_2017_Exp2     |            0.6768 |  0.2422 | 0.588  |        0.5269 |         0.5231 |
| Herting_2010        |            0.495  |  0.1644 | 0.8357 |        0.5879 |         0.5762 |
| Knauth_2022         |            0.6248 |  0.2157 | 0.7001 |        0.5947 |         0.612  |
| Konstantinidis_2020 |            0.5139 |  0.1707 | 0.8154 |        0.4135 |         0.4305 |
| Lempert_2015        |            0.5196 |  0.1749 | 0.8179 |        0.5097 |         0.506  |
| Robinson_2015       |            0.6202 |  0.2156 | 0.7069 |        0.5764 |         0.5888 |
| Zhou_2022           |            0.5152 |  0.1667 | 0.7614 |        0.2865 |         0.2662 |
| Zilker_2021         |            0.57   |  0.1955 | 0.8133 |        0.4931 |         0.5205 |

The ITC task has no correct answer, so accuracy is undefined. Log loss, Brier score, and AUC describe how well the drift model ranks and calibrates LL choices.

## 6. RT diagnostics and posterior predictive checks

| paper               |   rt_quantile_mae |   coverage_SS |   coverage_LL |   pit_ks_conditional |   pit_mean_conditional |   simulation_failure_rate |
|:--------------------|------------------:|--------------:|--------------:|---------------------:|-----------------------:|--------------------------:|
| Bixter_2015_Exp1    |            0.2065 |        0.861  |        0.8431 |               0.0908 |                 0.4758 |                    0      |
| Bixter_2015_Exp2    |            0.1354 |        0.7743 |        0.8627 |               0.0879 |                 0.4742 |                    0      |
| Bruder_2021         |            0.3832 |        0.93   |        0.8954 |               0.0895 |                 0.4986 |                    0      |
| Gluth_2017_Exp1     |            0.3931 |        0.8746 |        0.8407 |               0.046  |                 0.5082 |                    0.0004 |
| Gluth_2017_Exp2     |            0.4487 |        0.8717 |        0.873  |               0.0566 |                 0.5067 |                    0.0011 |
| Herting_2010        |            0.2752 |        0.866  |        0.8368 |               0.0367 |                 0.4957 |                    0      |
| Knauth_2022         |            0.0997 |        0.8322 |        0.8519 |               0.1019 |                 0.463  |                    0      |
| Konstantinidis_2020 |            0.234  |        0.7324 |        0.8414 |               0.0613 |                 0.4716 |                    0      |
| Lempert_2015        |            0.2955 |        0.949  |        0.9294 |               0.0966 |                 0.5031 |                    0      |
| Robinson_2015       |            0.143  |        0.8626 |        0.8516 |               0.0492 |                 0.4957 |                    0      |
| Zhou_2022           |            0.7225 |        0.909  |        0.8794 |               0.089  |                 0.5088 |                    0.0008 |
| Zilker_2021         |            0.5194 |        0.8914 |        0.8446 |               0.0504 |                 0.5018 |                    0.0005 |

RT quantile MAE is computed from the posterior predictive simulation (20,000 simulated trials at dt=1e-4). Central-80 coverage is reported separately for SS and LL. The conditional PIT KS statistic tests uniformity; with thousands of trials it is sensitive to small deviations, so effect size (mean PIT and bin deviations) matters more than the p-value.

**Structural note.** With constant drift and a midpoint start, choice and RT are independent within a trial. Across trials, the only route to an SS/LL RT difference is variation in drift, so the model can capture only the part of the association that is explained by the two predictors. In Lempert_2015 it predicts a small SS-minus-LL q90 difference with the opposite sign to the observed difference. This is a structural limitation to keep in mind when interpreting the RT diagnostics.

## 7. Participant-level cross-validation

| paper               |   n_folds_ok |   test_choice_log_loss_mean |   pooled_test_choice_log_loss |   test_auc_mean |   pooled_test_auc |   test_conditional_log_score_mean |   test_rt_quantile_mae_mean |   pooled_test_pit_ks_conditional |   cv_n_zero_density |   cv_n_below_t0 |
|:--------------------|-------------:|----------------------------:|------------------------------:|----------------:|------------------:|----------------------------------:|----------------------------:|---------------------------------:|--------------------:|----------------:|
| Bixter_2015_Exp1    |            5 |                      0.5723 |                        0.5726 |          0.7459 |            0.7338 |                           -1.4678 |                      0.3162 |                           0.0876 |                   0 |               0 |
| Bixter_2015_Exp2    |            5 |                      0.438  |                        0.4442 |          0.7925 |            0.7076 |                           -1.4633 |                      0.3232 |                           0.0863 |                   0 |               0 |
| Bruder_2021         |            5 |                      0.4948 |                        0.4959 |          0.8514 |            0.8228 |                           -1.5405 |                      0.3699 |                           0.0858 |                   5 |               5 |
| Gluth_2017_Exp1     |            5 |                      0.6687 |                        0.6683 |          0.6146 |            0.6018 |                           -2.3427 |                      0.7505 |                           0.0436 |                   0 |               0 |
| Gluth_2017_Exp2     |            5 |                      0.7005 |                        0.7007 |          0.614  |            0.5525 |                           -2.4734 |                      0.5215 |                           0.0512 |                   2 |               2 |
| Herting_2010        |            5 |                      0.5012 |                        0.5022 |          0.8414 |            0.8289 |                           -2.0567 |                      0.8913 |                           0.0224 |                   0 |               0 |
| Knauth_2022         |            5 |                      0.6378 |                        0.6389 |          0.7021 |            0.6803 |                           -0.7751 |                      0.2252 |                           0.0957 |                   1 |               1 |
| Konstantinidis_2020 |            5 |                      0.5292 |                        0.5276 |          0.8288 |            0.8012 |                           -1.4349 |                      0.2455 |                           0.0594 |                   0 |               0 |
| Lempert_2015        |            5 |                      0.5289 |                        0.5289 |          0.8246 |            0.8082 |                           -1.1367 |                      0.3365 |                           0.0925 |                   1 |               1 |
| Robinson_2015       |            5 |                      0.6238 |                        0.6235 |          0.7111 |            0.6939 |                           -1.4088 |                      0.2035 |                           0.0478 |                   0 |               0 |
| Zhou_2022           |            5 |                      0.5258 |                        0.5275 |          0.7734 |            0.7452 |                           -2.2711 |                      0.8336 |                           0.0848 |                   0 |               0 |
| Zilker_2021         |            5 |                      0.5714 |                        0.5714 |          0.8145 |            0.8089 |                           -2.3734 |                      0.6597 |                           0.0491 |                   1 |               1 |

- Held-out trials with zero model density (excluded from the RT log score): **10 of 86,466** (0.0116%).
- Held-out trials faster than the training-fold `t0`: **10**.

Both counts are reported rather than silently dropped: they measure how often a study-level `t0` is too slow for new participants.

Folds are over participants and are balanced by trial count. The predictor scaler is fitted on training participants only. Held-out metrics are proper scoring rules: choice log loss/Brier/AUC and the conditional RT log score; PIT and quantile MAE are calibration diagnostics. CV failures: none.

## 8. Cross-study synthesis

Study-level estimates were combined with DerSimonian-Laird random-effects meta-analysis. This is a descriptive synthesis of separately fitted models, not a pooled re-fit.

| parameter   |   k |   estimate |     se |   ci_lower |   ci_upper |   tau2 |     i2 |   q_pvalue |
|:------------|----:|-----------:|-------:|-----------:|-----------:|-------:|-------:|-----------:|
| a           |  12 |     3.4441 | 0.2482 |     2.9576 |     3.9305 | 0.7239 | 0.9873 |          0 |
| t0          |  12 |     0.3543 | 0.0602 |     0.2363 |     0.4722 | 0.042  | 0.994  |          0 |
| v0          |  12 |     0.0947 | 0.0465 |     0.0036 |     0.1858 | 0.0225 | 0.9197 |          0 |
| beta_amount |  12 |     0.2249 | 0.0309 |     0.1643 |     0.2854 | 0.0104 | 0.9539 |          0 |
| beta_delay  |  12 |    -0.2255 | 0.0518 |    -0.327  |    -0.124  | 0.0314 | 0.9869 |          0 |

Unweighted descriptive summaries are in `results/tables/classical_ddm_itc_meta_analysis_descriptive.csv`. Heterogeneity is substantial for several parameters, which is expected given different task designs, amount ranges, and delay ranges. The meta-analytic means should not be read as universal DDM parameters.

## 9. Lempert_2015 investigation

- Predicted q90 = 3.86 s versus observed 2.96 s (+0.89 s).
- Conditional PIT KS = 0.097; mean PIT = 0.503.
- q90 overpredicted for 30 of 30 participants; mean deviation +1.02 s.
- Max observed RT 5.99 s; 21 trials above 5.5 s, so there is no evidence of an unrecorded deadline or pile-up.
- Profile 95% interval for `a`: [2.714, 2.762] (a_hat = 2.738; bounded by grid: True).

The full diagnostic is in `results/reports/classical_ddm_itc_lempert_2015_diagnostic_report.md`. The conclusion is that the misfit is structural: the model overpredicts the slow tail and predicts an SS/LL RT difference with the opposite sign to the observed one. No model change was made to improve the result.

## 10. What completed and what did not

- Dataset hash verification and frozen cleaning audit.
- Mathematical validation (PyDDM grid, normalisation, simulation).
- Synthetic parameter recovery.
- Separate MLE for 12/12 studies.
- Participant-clustered sandwich uncertainty for all successful fits.
- Participant-level 5-fold CV for all studies with successful fits.
- Posterior predictive checks and PIT calibration for all successful fits.
- Cross-study random-effects synthesis.
- Lempert_2015 diagnostic investigation (no model change).
- Manifest and completeness verification script.

**Failed analyses:** none.

## 11. Limitations

- The model is a classical DDM with constant drift within a trial, fixed midpoint start, and unit noise. It omits across-trial variability in drift, start point, and non-decision time.
- Constant drift with a midpoint start makes choice and RT independent within a trial. Across trials, only drift variation can generate an SS/LL RT difference, so the model is likely to understate the choice-RT association.
- `t0` is bounded above by the fastest observed RT. Several fits sit close to that bound, so their Hessian and sandwich SEs are not valid confidence intervals for the unconstrained parameter.
- Participant-clustered SEs account for within-participant correlation but not for the boundary constraint or for model misspecification.
- Cross-validation evaluates a study-specific model fitted to other participants in the same study. It does not test transfer to new studies or new task designs.
- The random-effects synthesis uses study-level clustered SEs and assumes approximate normality; with 12 studies the meta-analytic uncertainty is itself imprecise.
- Posterior predictive checks use the observed predictor distribution and 20,000 simulations at dt=1e-4. Simulation has a small residual discretisation bias in the extreme tail.
- The ITC task has no accuracy criterion, so choice quality is judged only by log loss, Brier score, and AUC.

## 12. Next steps

1. Add across-trial variability in drift and non-decision time, and compare on the same participant-level CV folds.
2. Allow a non-midpoint start point and test whether it captures the observed choice-RT association.
3. Consider a contaminant/deadline component only if the deadline diagnostics justify it in other studies; Lempert_2015 does not.
4. Report hierarchical or participant-level estimates alongside study-level MLEs before making individual-difference claims.
5. Pre-specify the model comparison metric (held-out log score) and the set of models before looking at the remaining studies.
6. Re-run `verify_outputs.py` after any change and keep the manifest with the results.

## 13. Reproduction

From the project root:

```text
..\.venv\Scripts\python.exe -m unittest discover -s tests -v
..\.venv\Scripts\python.exe experiments\classical_ddm\run_full.py
..\.venv\Scripts\python.exe experiments\classical_ddm\verify_outputs.py
```

All paths are relative to `classical-ddm-intertemporal-choice/`. The dataset is read from `../data/itc_all_data.csv` (or `DDM_ITC_CSV`) and is never modified.

## 14. Output inventory

- `results/fits/classical_ddm_itc_{study}_fit.json` - parameters, likelihood, Hessian, starts.
- `results/fits/classical_ddm_itc_{study}_clustered_uncertainty.json` - naive and clustered covariance, participant scores.
- `results/fits/classical_ddm_itc_{study}_diagnostics.json` - PPC, PIT, model quantiles, bounds.
- `results/figures/classical_ddm_itc_{study}_*.png` - RT quantiles, calibration, choice bins, PPC tail survival, PIT histogram.
- `results/tables/classical_ddm_itc_12_study_fit_summary.csv` - one row per study.
- `results/tables/classical_ddm_itc_clustered_uncertainty_summary.csv` - long parameter table.
- `results/tables/classical_ddm_itc_5_fold_cross_validation_*.csv` - CV results, fold membership, and participant assignments.
- `results/tables/classical_ddm_itc_random_effects_meta_analysis.csv` and `classical_ddm_itc_meta_analysis_descriptive.csv`.
- `results/lempert_2015/` - investigation JSON, profile likelihood, participant deviations, figures; report in `results/reports/`.
- `results/classical_ddm_itc_manifest.json` - hashes and environment for verification.

## 15. Analysis checks

| check                                | passed   |
|:-------------------------------------|:---------|
| dataset_hash_verified                | True     |
| expected_sample_size                 | True     |
| all_12_studies_fitted                | True     |
| all_fits_converged                   | True     |
| all_hessians_positive_definite       | True     |
| cross_validation_complete            | True     |
| lempert_investigation_written        | True     |
| pyddm_max_abs_error_below_1e_6       | True     |
| normalisation_total_error_below_1e_4 | True     |

