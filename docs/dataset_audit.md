# Dataset audit

## 1. Source and verification

- Dataset: ITC Database, Pongratz & Schoemann (preprint, manuscript v2.0).
- OSF project: <https://osf.io/3wsae/>.
- Local file: `../data/itc_all_data.csv` (outside this project), or the path in
  the `DDM_ITC_CSV` environment variable.
- Licence: CC BY-NC-SA 4.0, verified from the OSF API.
- SHA-256: `14e5bf2f6434d791a5f68d8f0e4cec41e4c6bb671ce6fd80e83879f9759e0517`.
- Size: 484,996,376 bytes, matching the OSF `data/all_data.csv` file exactly.
- The raw file is read by reference and is never copied into this project or
  modified.

## 2. Structure

| Quantity | Value |
|---|---|
| Trial rows | 1,172,644 |
| Studies (`paper`) | 100 |
| Unique subjects (`subj_ident`) | 11,852 |
| Columns | 42 |
| Missing values in `ss_value`, `ss_time`, `ll_value`, `ll_time`, `choice`, `rt` | 0 |

Key variables:

| Variable | Meaning |
|---|---|
| `paper` | study key |
| `subj_ident` | unique subject identifier across the database |
| `sess_ident` | unique session identifier |
| `ss_value`, `ss_time` | smaller-sooner amount; delay in days |
| `ll_value`, `ll_time` | larger-later amount; delay in days |
| `choice` | 0 = SS chosen, 1 = LL chosen |
| `rt` | response time in seconds |
| `subj_excl`, `trial_excl` | compiler exclusion flags |
| `time_pressure` | response deadline present |
| `rt_data_analyzed` | RTs analysed in the original publication |
| `rt_in_full_seconds` | RTs recorded only to whole seconds |
| `additional_methods` | e.g. mouse tracking, fMRI, eye tracking |

## 3. Data-quality findings

| Issue | Count |
|---|---|
| `trial_excl == True` | 107,072 |
| Rows belonging to `subj_excl == True` subjects | 39,196 |
| `rt <= 0` | 318 |
| `rt < 0.2` | 8,098 |
| `rt > 20` | 3,895 |
| `ss_value >= ll_value` | 33,487 |
| `ss_time >= ll_time` | 13,921 |
| Exact duplicate-looking rows | 20,129 |

The duplicate count is misleading. Many rows are legitimate trials with
coarse RT recording. For example, `Alvarez_2022` has 22,116 rows but only 5,412
unique rows because RTs are recorded in whole seconds. Exact-row
deduplication would therefore delete valid trials and is **not** used.

Other design features:

- 27/100 studies have a response deadline.
- 4/100 studies record RTs only to whole seconds.
- 26/100 studies have RTs previously analysed by the original authors.
- Some studies use mouse or finger tracking, where RT includes movement time.
- The median participant contributes 39 trials; 1,639 participants have at
  least 200 trials.
- 34 participants in `Bruder_2021` contribute three sessions of 96 trials each.

## 4. Frozen cleaning rules

A trial is eligible if all of the following hold:

1. The subject is not flagged by `subj_excl`.
2. The trial is not flagged by `trial_excl`.
3. `0.2 <= rt <= 20` seconds.
4. `ss_value > 0` and `ll_value > 0`.
5. `ss_time < ll_time` (a genuine delay difference).
6. `choice` is 0 or 1.
7. No response deadline (`time_pressure == 0`).
8. Simultaneous presentation (`presentation_of_information == "Simultaneous"`).
9. No mouse or finger tracking in `additional_methods`.
10. RT not recorded only to whole seconds (`rt_in_full_seconds == 0`).
11. RTs previously analysed by the original authors (`rt_data_analyzed == 1`).

Rules 3-6 are trial-level; rules 7-11 are study-level design filters. The
lower RT bound avoids contaminant fast guesses but means that `t0` cannot be
identified below 0.2 s. This is reported as a limitation; a sensitivity
analysis with a different lower bound is reserved for later work.

All sessions are retained. Because 34 participants contribute three sessions,
uncertainty is clustered by participant in the full analysis; a
first-session-only sample is reserved as a sensitivity analysis.

## 5. Primary sample

Applying the rules above yields:

| Quantity | Value |
|---|---|
| Studies | 12 |
| Participants | 553 |
| Trials | 86,466 |
| Median trials per participant | 109 |
| Participants with >= 150 trials | 253 |
| Participant-sessions | 621 |

The 12 studies are:

`Bixter_2015_Exp1`, `Bixter_2015_Exp2`, `Bruder_2021`,
`Gluth_2017_Exp1`, `Gluth_2017_Exp2`, `Herting_2010`,
`Knauth_2022`, `Konstantinidis_2020`, `Lempert_2015`,
`Robinson_2015`, `Zhou_2022`, `Zilker_2021`.

This is the primary sample. A broader sample of about 60 studies and 6,020
participants is available for a later sensitivity/generalisation check, but it
mixes RT-screening conventions and is not the primary analysis.

## 6. Suitability and limitations

The dataset is suitable for a DDM comparison because it provides trial-level
choices, RTs, amount pairs, and delay pairs. It is not a classic perceptual
two-alternative forced-choice task: there is no correct answer, so "accuracy"
is undefined. Choice prediction, RT prediction, and goodness of fit are the
appropriate outcomes.

The main limitations are heterogeneity across studies, coarse RT recording in
some studies, response deadlines in others, long and heavy-tailed RTs, and the
small number of trials per participant in much of the database.
