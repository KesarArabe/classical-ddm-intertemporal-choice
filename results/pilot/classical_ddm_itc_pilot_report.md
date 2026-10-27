# Classical DDM pilot report

This is a **pilot report**. The full 12-study fit has not been run. The purpose is to validate the likelihood, check recovery, and decide whether the full experiment is ready.

## 1. Dataset verification

- Path: `C:\Users\ARAVE\OneDrive\Documents\Default Project\data\itc_all_data.csv`
- Size: 484,996,376 bytes
- SHA-256 matches the OSF file: **True**
- Licence: CC BY-NC-SA 4.0

## 2. Cleaning audit

| step                       |    rows |
|:---------------------------|--------:|
| raw_rows                   | 1172644 |
| after_subject_exclusion    | 1132944 |
| after_trial_exclusion      | 1025890 |
| after_rt_range             | 1015811 |
| after_valid_stimuli_choice |  997324 |
| after_design_filters       |   86466 |
| studies                    |      12 |
| subjects                   |     553 |
| trials                     |   86466 |
| sessions                   |     621 |
| min_trials_per_subject     |       7 |
| max_trials_per_subject     |     380 |
| subjects_ge_150_trials     |     253 |
| subjects_ge_300_trials     |      70 |

## 3. Pilot sample

| paper        |   trials |   subjects |   median_trials_per_subject |   rt_median |   choice_ll |   time_pressure |   rt_in_full_seconds |   rt_data_analyzed | mouse_or_finger   | presentation   |
|:-------------|---------:|-----------:|----------------------------:|------------:|------------:|----------------:|---------------------:|-------------------:|:------------------|:---------------|
| Knauth_2022  |     8689 |         35 |                       278   |       0.822 |       0.595 |               0 |                    0 |                  1 | False             | Simultaneous   |
| Lempert_2015 |    10589 |         30 |                       355.5 |       1.699 |       0.51  |               0 |                    0 |                  1 | False             | Simultaneous   |

## 4. Mathematical validation

| check                         | metric                        |     value |
|:------------------------------|:------------------------------|----------:|
| PyDDM density grid            | max absolute error            |  0        |
| PyDDM density grid            | max relative error            |  0        |
| normalisation (v=0.0, a=1.0)  | total integral error          | -0        |
| normalisation (v=0.0, a=1.0)  | upper-bound probability error | -0        |
| normalisation (v=0.5, a=1.2)  | total integral error          | -0        |
| normalisation (v=0.5, a=1.2)  | upper-bound probability error | -0        |
| normalisation (v=2.0, a=2.0)  | total integral error          | -0        |
| normalisation (v=2.0, a=2.0)  | upper-bound probability error | -0        |
| normalisation (v=-1.0, a=0.8) | total integral error          | -0        |
| normalisation (v=-1.0, a=0.8) | upper-bound probability error | -0        |
| simulation                    | choice-proportion difference  |  0.005878 |
| simulation                    | max RT-quantile difference    |  0.064399 |

## 5. Synthetic parameter recovery

| parameter   |   true |   mean_estimate |   sd_estimate |   bias |   rmse |   relative_rmse |
|:------------|-------:|----------------:|--------------:|-------:|-------:|----------------:|
| a           |    1.2 |           1.212 |         0.015 |  0.012 |  0.018 |           0.015 |
| t0          |    0.3 |           0.301 |         0.002 |  0.001 |  0.002 |           0.007 |
| v0          |    0.2 |           0.214 |         0.029 |  0.014 |  0.03  |           0.148 |
| beta_amount |    0.8 |           0.8   |         0.015 | -0     |  0.013 |           0.016 |
| beta_delay  |   -0.6 |          -0.578 |         0.034 |  0.022 |  0.038 |           0.063 |

## 6. Pilot fits

| paper        |   trials |   subjects |     a |   se_a |    t0 |   se_t0 |    v0 |   se_v0 |   beta_amount |   se_beta_amount |   beta_delay |   se_beta_delay |   log_likelihood |   log_likelihood_per_trial | converged   | hessian_positive_definite   |   hessian_condition |   start_objective_spread |   runtime_seconds |   choice_log_loss |   brier |   auc |   rt_quantile_mae |   coverage_SS |   coverage_LL |   observed_ll |   predicted_ll |   simulation_failure_rate |   t0_margin_seconds |
|:-------------|---------:|-----------:|------:|-------:|------:|--------:|------:|--------:|--------------:|-----------------:|-------------:|----------------:|-----------------:|---------------------------:|:------------|:----------------------------|--------------------:|-------------------------:|------------------:|------------------:|--------:|------:|------------------:|--------------:|--------------:|--------------:|---------------:|--------------------------:|--------------------:|
| Lempert_2015 |    10589 |         30 | 2.738 |  0.013 | 0.462 |   0.001 | 0.03  |   0.008 |         0.484 |            0.009 |       -0.313 |           0.008 |         -17591.7 |                     -1.661 | True        | True                        |             155.366 |                        0 |            22.355 |             0.52  |   0.175 | 0.818 |             0.296 |         0.949 |         0.929 |         0.51  |          0.506 |                         0 |               0.013 |
| Knauth_2022  |     8689 |         35 | 2.047 |  0.009 | 0.196 |   0.001 | 0.258 |   0.011 |         0.413 |            0.012 |       -0.171 |           0.012 |         -12099.8 |                     -1.393 | True        | True                        |             233.829 |                        0 |            17.416 |             0.625 |   0.216 | 0.7   |             0.1   |         0.832 |         0.852 |         0.595 |          0.612 |                         0 |               0.01  |

## 7. Diagnostics

| paper        |   choice_log_loss |   brier |   auc |   rt_quantile_mae |   coverage_SS |   coverage_LL |   observed_ll |   predicted_ll |   simulation_failure_rate |   t0_margin_seconds |
|:-------------|------------------:|--------:|------:|------------------:|--------------:|--------------:|--------------:|---------------:|--------------------------:|--------------------:|
| Lempert_2015 |             0.52  |   0.175 | 0.818 |             0.296 |         0.949 |         0.929 |         0.51  |          0.506 |                         0 |               0.013 |
| Knauth_2022  |             0.625 |   0.216 | 0.7   |             0.1   |         0.832 |         0.852 |         0.595 |          0.612 |                         0 |               0.01  |

## 8. Pilot checks

| check                              | passed   |
|:-----------------------------------|:---------|
| all_fits_converged                 | True     |
| all_hessians_positive_definite     | True     |
| all_optimizer_starts_agree         | True     |
| all_t0_below_min_rt                | True     |
| all_auc_above_0_50                 | True     |
| all_coverage_between_0_60_and_0_95 | True     |
| recovery_relative_rmse_below_0_30  | True     |

## 9. Interpretation

- All pilot fits converged: **True**.
- All Hessians were positive definite: **True**.
- All multi-start optimisations reached the same objective: **True**.
- Recovery relative RMSE was below 0.30 for all parameters: **True**.

### Choice prediction versus accuracy

The ITC task has no correct answer, so there is no accuracy to report. The relevant metrics are log loss, Brier score, and AUC for LL choices. The model is comfortably better than chance in both studies (AUC 0.818 and 0.700).

### RT fit

- **Lempert_2015**: RT quantile MAE = 0.296 s. The largest discrepancy is the SS q90: observed 2.95 s versus predicted 3.85 s (+0.90 s). Central-80 coverage is 0.949 (SS) and 0.929 (LL).

- **Knauth_2022**: RT quantile MAE = 0.100 s. The largest discrepancy is the LL q70: observed 1.07 s versus predicted 1.34 s (+0.27 s). Central-80 coverage is 0.832 (SS) and 0.852 (LL).

The Knauth fit is good on both choices and RTs. The Lempert fit predicts too much mass in the slow tail: coverage is only just inside the 0.95 ceiling, so the coverage check alone would hide the misfit. Both studies show the same shape of discrepancy, with observed RTs more concentrated than predicted. This is consistent with the single-boundary, fixed-parameter model absorbing a misspecified drift: the boundary is pushed up to match the choice curve, which widens the predicted RT distribution. This is a substantive result, not a numerical artefact.

### Non-decision time

- **Lempert_2015**: t0 = 0.462 s, minimum observed RT = 0.475 s, margin = 0.013 s.

- **Knauth_2022**: t0 = 0.196 s, minimum observed RT = 0.207 s, margin = 0.010 s.

Both t0 estimates sit close to the fastest observed response, which is a hard upper bound in this model. The classical DDM therefore cannot use the non-decision time it would prefer; across-trial variability in t0 is the natural next relaxation. Because t0 is at (or within 1-2% of) the constraint boundary, its Hessian-based standard error is not a valid confidence interval and should not be interpreted as one.

### Decision

The likelihood is numerically validated, recovery is good, and the optimiser is stable. The pilot is acceptable as an engineering gate for the full 12-study classical fit, with one caveat: the classical model already shows a clear RT-tail misfit in Lempert_2015. The full fit should therefore be reported as a baseline, and the next model (across-trial variability in drift and non-decision time) should be specified in advance as the substantive comparison.

The pilot is a review gate, not a final result. If any check failed, the cause must be addressed before the full fit. The full fit should also add participant-clustered uncertainty, 5-fold cross-validation by participant, and the remaining 10 primary studies.

## 10. What has not been done

- No full 12-study fit.
- No held-out cross-validation yet.
- No participant-level or hierarchical model yet.
- No comparison with later DDM formulations yet.

