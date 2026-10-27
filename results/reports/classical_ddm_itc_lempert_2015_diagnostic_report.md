# Lempert_2015: diagnostic investigation

This investigation does **not** modify the model, the cleaning rules, or the fitted parameters. It documents where the classical DDM fails for this study.

## Fitted parameters

| Parameter | Estimate |
|---|---:|
| a | 2.7379 |
| t0 | 0.4621 |
| v0 | 0.0295 |
| beta_amount | 0.4837 |
| beta_delay | -0.3127 |

## Tail diagnostics

| Threshold / quantile   |   Observed |   Predicted |   Difference |
|:-----------------------|-----------:|------------:|-------------:|
| >1.0 s                 |     0.9427 |      0.848  |      -0.0947 |
| >2.0 s                 |     0.3499 |      0.3978 |       0.0479 |
| >3.0 s                 |     0.0949 |      0.1884 |       0.0935 |
| >4.0 s                 |     0.0251 |      0.0898 |       0.0647 |
| >5.0 s                 |     0.0053 |      0.0441 |       0.0388 |
| >6.0 s                 |     0      |      0.0208 |       0.0208 |
| q50                    |     1.699  |      1.698  |      -0.001  |
| q90                    |     2.9632 |      3.8551 |       0.8919 |
| q95                    |     3.5    |      4.805  |       1.305  |
| q99                    |     4.6561 |      7.0413 |       2.3852 |

Positive differences mean the model predicts more slow responses than were observed.

## Deadline and pile-up checks

- Trials: 10,589; min RT 0.475 s; max RT 5.994 s.
- Trials above 5 s: 56 (0.5289%); above 5.5 s: 21; above 6 s: 0.
- Most common 0.01-s RT bin: 1.28 s with 96 trials (0.907%), versus 22.8 expected if RTs were spread evenly across distinct bins.

There is no evidence of an unrecorded deadline or a response pile-up. The misfit is therefore not explained by truncated RTs.

## Participant-level deviations

- q90 is overpredicted for 30 of 30 participants and underpredicted for 0.
- Mean q90 deviation: +1.016 s; median +1.011 s; RMSE 1.127 s.
- Worst overprediction: +2.193 s; worst underprediction: +0.128 s.
- Observed-versus-predicted q90 correlation: 0.204; slope 0.007.

A slope far below 1 means the model compresses participant differences: predicted q90 varies much less across participants than observed q90. With a single boundary and non-decision time per study, the model has no participant-level parameter to explain stable individual differences in RT scale. The per-participant CSV gives the exact pattern.

## PIT calibration

Conditional PIT mean = 0.5031, KS = 0.0966 (p = 2.2e-86), maximum bin deviation = 0.0690.

## Profile likelihood over boundary separation

- Hessian (trial-level) SE for a: 0.0127.
- Participant-clustered SE for a: 0.0631.
- Profile grid: a_hat +/- 0.1015, 15 points from 2.6364 to 2.8395.
- Full-fit log-likelihood: -17591.732.
- Maximum profile log-likelihood: -17591.732.
- 95% profile interval for a: [2.7142, 2.7620].
- Interval bounded inside the grid: True.

The profile interval is likelihood-based and treats trials as independent, so it inherits the naive precision rather than the participant-clustered one: it is about 0.19 times the width of a Wald interval built from the clustered SE. The profile is reported here as a shape diagnostic (is the likelihood single-peaked near the estimate?), not as the study's uncertainty interval; the clustered SE is the interval to use.

The profile is single-peaked and sharply curved around the estimate, so the boundary separation is well identified. Moving a away from the estimate makes the overall fit worse, so the tail misfit is not a consequence of a suboptimal boundary estimate.

## Honest conclusion

The Lempert_2015 misfit is a genuine lack of fit of the classical DDM, not a numerical artefact. The model overpredicts the slow tail, shows PIT miscalibration, and predicts a small SS/LL RT difference with the opposite sign to the observed difference. Because constant drift makes choice and RT independent within a trial, the model can generate an SS/LL difference only through drift variation, which is too weak here. No deadline or pile-up explains the pattern. The next step should be a model with across-trial variability (drift, non-decision time, and/or start point), specified in advance and compared on held-out participants; it should not be a post hoc adjustment to this fit.

