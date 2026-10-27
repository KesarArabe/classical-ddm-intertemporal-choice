"""Verify completeness, internal consistency, and file integrity of the full run.

Run from the project root:

    ..\.venv\Scripts\python.exe experiments\classical_ddm\verify_outputs.py

The script checks the manifest hashes, the expected file inventory, the frozen
sample size, the stored likelihoods (by recomputing them at the stored
parameters), parameter bounds, clustered-covariance dimensions, CV fold
coverage, and the presence of the Lempert investigation.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.clean import (  # noqa: E402
    add_predictors,
    clean_trials,
    primary_audit,
    select_primary,
)
from src.data.load import (  # noqa: E402
    load_trials,
    resolve_configured_path,
    verify_dataset,
)
from src.models.classical_ddm import (  # noqa: E402
    DDMParameters,
    PARAMETER_NAMES,
    log_likelihood,
)

OUT_DIR = ROOT / "results"
PREFIX = "classical_ddm_itc"
MANIFEST_PATH = OUT_DIR / f"{PREFIX}_manifest.json"
FIT_DIR = OUT_DIR / "fits"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
LEMPERT_DIR = OUT_DIR / "lempert_2015"
REPORT_PATH = OUT_DIR / "reports" / f"{PREFIX}_full_analysis_report.md"
LEMPERT_REPORT_PATH = (
    OUT_DIR / "reports" / f"{PREFIX}_lempert_2015_diagnostic_report.md"
)
CONFIG_PATH = Path(__file__).with_name("config.json")


def _study_stem(paper: str) -> str:
    """Return the shared filename stem for one study's result files."""
    return f"{PREFIX}_{paper}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append((name, bool(passed), detail))

    manifest_path = MANIFEST_PATH
    check("manifest exists", manifest_path.exists(), str(manifest_path))
    if not manifest_path.exists():
        print("Manifest missing; cannot verify.")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    missing = []
    hash_mismatch = []
    size_mismatch = []
    for entry in manifest["files"]:
        path = OUT_DIR / entry["relative_path"]
        if not path.exists():
            missing.append(entry["relative_path"])
            continue
        if path.stat().st_size != entry["size_bytes"]:
            size_mismatch.append(entry["relative_path"])
        if _sha256(path) != entry["sha256"]:
            hash_mismatch.append(entry["relative_path"])
    check("all manifest files exist", not missing, f"missing={missing[:5]}")
    check("all file sizes match manifest", not size_mismatch, f"{size_mismatch[:5]}")
    check("all file hashes match manifest", not hash_mismatch, f"{hash_mismatch[:5]}")

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    studies = cfg["primary_studies"]
    fit_dir = FIT_DIR
    fig_dir = FIG_DIR
    summary_dir = TABLE_DIR
    lempert_dir = LEMPERT_DIR

    missing_fits = []
    for paper in studies:
        stem = _study_stem(paper)
        for suffix in (
            "_fit.json",
            "_clustered_uncertainty.json",
            "_diagnostics.json",
            "_parameters.csv",
        ):
            if not (fit_dir / f"{stem}{suffix}").exists():
                missing_fits.append(f"{stem}{suffix}")
    check("all per-study fit files present", not missing_fits, f"{missing_fits[:5]}")

    missing_figures = []
    for paper in studies:
        stem = _study_stem(paper)
        for suffix in (
            "_reaction_time_quantiles.png",
            "_choice_calibration.png",
            "_choice_by_amount_delay.png",
            "_reaction_time_tail.png",
            "_pit_calibration.png",
        ):
            if not (fig_dir / f"{stem}{suffix}").exists():
                missing_figures.append(f"{stem}{suffix}")
    for name in (
        f"{PREFIX}_study_parameter_forest_plot.png",
        f"{PREFIX}_study_summary_metrics.png",
        f"{PREFIX}_parameter_recovery.png",
    ):
        if not (fig_dir / name).exists():
            missing_figures.append(name)
    check("all figures present", not missing_figures, f"{missing_figures[:5]}")

    required_summary = [
        f"{PREFIX}_12_study_fit_summary.csv",
        f"{PREFIX}_12_study_fit_summary.json",
        f"{PREFIX}_clustered_uncertainty_summary.csv",
        f"{PREFIX}_parameter_estimates_long.csv",
        f"{PREFIX}_random_effects_meta_analysis.csv",
        f"{PREFIX}_meta_analysis_descriptive.csv",
        f"{PREFIX}_random_effects_meta_analysis.json",
        f"{PREFIX}_5_fold_cross_validation_summary.csv",
        f"{PREFIX}_5_fold_cross_validation_folds.csv",
        f"{PREFIX}_5_fold_cross_validation_assignments.csv",
        f"{PREFIX}_analysis_checks.json",
    ]
    missing_summary = [
        name for name in required_summary if not (summary_dir / name).exists()
    ]
    check("all summary files present", not missing_summary, f"{missing_summary[:5]}")
    check(
        "report present",
        REPORT_PATH.exists(),
        str(REPORT_PATH),
    )
    check(
        "Lempert investigation present",
        LEMPERT_REPORT_PATH.exists()
        and (lempert_dir / f"{PREFIX}_lempert_2015_investigation.json").exists()
        and (lempert_dir / f"{PREFIX}_lempert_2015_profile_likelihood_a.csv").exists(),
        str(lempert_dir),
    )

    # Frozen sample.
    data_path = resolve_configured_path(cfg["data"]["path"], ROOT)
    verification = verify_dataset(data_path, check_hash=True)
    check(
        "dataset hash verified",
        bool(verification["sha256_ok"]),
        verification["sha256"],
    )
    raw = load_trials(data_path)
    cleaned, _ = clean_trials(raw, cfg["cleaning"])
    primary = select_primary(cleaned)
    audit = primary_audit(primary)
    check(
        "frozen sample size",
        int(audit["studies"]) == int(cfg["expected_primary"]["studies"])
        and int(audit["subjects"]) == int(cfg["expected_primary"]["subjects"])
        and int(audit["trials"]) == int(cfg["expected_primary"]["trials"]),
        f"{audit['studies']} studies, {audit['subjects']} subjects, {audit['trials']} trials",
    )

    # Stored likelihoods and bounds.
    likelihood_failures = []
    bound_failures = []
    for paper in studies:
        fit_path = fit_dir / f"{_study_stem(paper)}_fit.json"
        if not fit_path.exists():
            continue
        payload = json.loads(fit_path.read_text(encoding="utf-8"))
        df = primary[primary["paper"] == paper].copy()
        df, _ = add_predictors(df)
        params = DDMParameters.from_array(
            [payload["params"][name] for name in PARAMETER_NAMES]
        )
        recomputed = log_likelihood(
            params,
            df["rt"].to_numpy(dtype=float),
            df["choice"].to_numpy(dtype=int),
            df["x_amt"].to_numpy(dtype=float),
            df["x_delay"].to_numpy(dtype=float),
            n_terms=int(cfg["model"]["n_terms"]),
        )
        if not np.isfinite(recomputed) or abs(recomputed - payload["log_likelihood"]) > 1e-6 * max(
            1.0, abs(payload["log_likelihood"])
        ):
            likelihood_failures.append(
                f"{paper}: stored {payload['log_likelihood']}, recomputed {recomputed}"
            )
        min_rt = float(df["rt"].min())
        if not (params.t0 < min_rt):
            bound_failures.append(f"{paper}: t0={params.t0:.4f} >= min_rt={min_rt:.4f}")
        for name, (low, high) in zip(PARAMETER_NAMES, payload["bounds"]):
            value = float(payload["params"][name])
            if not (float(low) <= value <= float(high)):
                bound_failures.append(
                    f"{paper}: {name}={value} outside [{low}, {high}]"
                )
    check(
        "stored likelihoods reproduce",
        not likelihood_failures,
        f"{likelihood_failures[:3]}",
    )
    check("all estimates respect bounds", not bound_failures, f"{bound_failures[:3]}")

    # Clustered covariance dimensions and positivity.
    clustered_failures = []
    for paper in studies:
        path = fit_dir / f"{_study_stem(paper)}_clustered_uncertainty.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        cov = np.asarray(payload["clustered_covariance"], dtype=float)
        if cov.shape != (5, 5):
            clustered_failures.append(f"{paper}: shape {cov.shape}")
        elif not np.all(np.diag(cov) > 0):
            clustered_failures.append(f"{paper}: non-positive diagonal")
        elif not np.all(np.isfinite(cov)):
            clustered_failures.append(f"{paper}: non-finite entries")
        elif np.min(np.linalg.eigvalsh(cov)) < -1e-8:
            clustered_failures.append(f"{paper}: covariance not positive semidefinite")
    check(
        "clustered covariance matrices valid",
        not clustered_failures,
        f"{clustered_failures[:3]}",
    )

    # CV fold coverage and isolation information.
    assignments = pd.read_csv(
        summary_dir / f"{PREFIX}_5_fold_cross_validation_assignments.csv"
    )
    cv_failures = []
    for paper in studies:
        subset = assignments[assignments["paper"] == paper]
        expected_subjects = int(primary[primary["paper"] == paper]["subj_ident"].nunique())
        if subset.shape[0] != expected_subjects:
            cv_failures.append(
                f"{paper}: {subset.shape[0]} assignments for {expected_subjects} subjects"
            )
        if sorted(subset["fold"].unique().tolist()) != list(range(int(cfg["cross_validation"]["n_folds"]))):
            cv_failures.append(f"{paper}: fold labels {sorted(subset['fold'].unique())}")
        if subset["subj_ident"].duplicated().any():
            cv_failures.append(f"{paper}: duplicate subject assignment")
    check("CV assignments cover all participants", not cv_failures, f"{cv_failures[:3]}")

    cv_summary = pd.read_csv(
        summary_dir / f"{PREFIX}_5_fold_cross_validation_summary.csv"
    )
    check(
        "CV completed for all studies",
        len(cv_summary) == len(studies)
        and bool((cv_summary["n_folds_ok"] == cfg["cross_validation"]["n_folds"]).all()),
        f"{len(cv_summary)} rows",
    )

    fits_summary = pd.read_csv(
        summary_dir / f"{PREFIX}_12_study_fit_summary.csv"
    )
    ok = fits_summary[fits_summary["status"] == "ok"]
    check(
        "all 12 studies fitted",
        len(ok) == len(studies),
        f"{len(ok)} successful fits",
    )
    check(
        "all successful fits converged",
        bool(ok["converged"].all()) if len(ok) else False,
        f"{int(ok['converged'].sum()) if len(ok) else 0}/{len(ok)}",
    )
    check(
        "all Hessians positive definite",
        bool(ok["hessian_positive_definite"].all()) if len(ok) else False,
        f"{int(ok['hessian_positive_definite'].sum()) if len(ok) else 0}/{len(ok)}",
    )

    report_text = REPORT_PATH.read_text(encoding="utf-8")
    check(
        "report mentions all studies",
        all(paper in report_text or paper.split("_")[0] in report_text for paper in studies),
        "study names searched in report",
    )

    failures = [item for item in checks if not item[1]]
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}" + (f" -- {detail}" if detail and not passed else ""))
    print(f"\n{len(checks) - len(failures)}/{len(checks)} checks passed.")
    if failures:
        print("Verification failed.")
        return 1
    print("Verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
