"""Data loading and provenance checks for the ITC Database."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable

import pandas as pd

# src/data/load.py -> project root is parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT.parent / "data" / "itc_all_data.csv"

EXPECTED_SIZE = 484_996_376
EXPECTED_SHA256 = (
    "14e5bf2f6434d791a5f68d8f0e4cec41e4c6bb671ce6fd80e83879f9759e0517"
)

REQUIRED_COLUMNS = [
    "paper",
    "subj_ident",
    "sess_ident",
    "trial_idx",
    "ss_value",
    "ss_time",
    "ll_value",
    "ll_time",
    "choice",
    "rt",
    "subj_excl",
    "trial_excl",
    "time_pressure",
    "rt_data_analyzed",
    "rt_in_full_seconds",
    "additional_methods",
    "presentation_of_information",
]

LOAD_COLUMNS = REQUIRED_COLUMNS + [
    "procedure",
    "incentivization",
    "online_study",
    "missing_trials_relative",
]


def resolve_data_path(path: str | os.PathLike | None = None) -> Path:
    """Return the data path, allowing an environment-variable override."""
    if path is not None:
        return Path(path).expanduser().resolve()
    env_path = os.environ.get("DDM_ITC_CSV")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return DEFAULT_DATA_PATH


def resolve_configured_path(
    config_path: str | os.PathLike,
    root: str | os.PathLike,
) -> Path:
    """Resolve the dataset path from the config, with ``DDM_ITC_CSV`` override.

    ``DDM_ITC_CSV`` takes precedence so the project can be reproduced against
    a copy stored elsewhere without editing the frozen configuration.
    """
    env_path = os.environ.get("DDM_ITC_CSV")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return (Path(root) / Path(config_path)).expanduser().resolve()


def sha256_file(path: str | os.PathLike, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset(
    path: str | os.PathLike | None = None,
    check_hash: bool = True,
) -> dict:
    """Verify existence, size, and optionally the SHA-256 of the CSV."""
    resolved = resolve_data_path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Dataset not found: {resolved}. See docs/dataset_provenance.md "
            "for download instructions or set DDM_ITC_CSV."
        )
    size = resolved.stat().st_size
    if size != EXPECTED_SIZE:
        raise ValueError(
            f"Unexpected dataset size: {size} bytes; expected {EXPECTED_SIZE}. "
            "The file may be incomplete or a different version."
        )
    result = {
        "path": str(resolved),
        "size": size,
        "size_ok": True,
        "sha256": None,
        "sha256_ok": None,
    }
    if check_hash:
        digest = sha256_file(resolved).lower()
        result["sha256"] = digest
        result["sha256_ok"] = digest == EXPECTED_SHA256.lower()
        if not result["sha256_ok"]:
            raise ValueError(
                "SHA-256 does not match the verified OSF file. "
                f"Got {digest}; expected {EXPECTED_SHA256}."
            )
    return result


def load_raw(
    path: str | os.PathLike | None = None,
    usecols: Iterable[str] | None = None,
    check_hash: bool = False,
) -> pd.DataFrame:
    """Load the raw CSV.

    The size is checked on every load. The hash is optional because hashing
    485 MB on every run is unnecessary; call ``verify_dataset`` explicitly
    when provenance needs to be re-established.
    """
    resolved = resolve_data_path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Dataset not found: {resolved}. See docs/dataset_provenance.md "
            "for download instructions or set DDM_ITC_CSV."
        )
    size = resolved.stat().st_size
    if size != EXPECTED_SIZE:
        raise ValueError(
            f"Unexpected dataset size: {size} bytes; expected {EXPECTED_SIZE}."
        )
    if check_hash:
        verify_dataset(resolved, check_hash=True)

    columns = list(usecols) if usecols is not None else LOAD_COLUMNS
    df = pd.read_csv(resolved, usecols=columns, low_memory=False)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def load_trials(
    path: str | os.PathLike | None = None,
    check_hash: bool = False,
) -> pd.DataFrame:
    """Load the columns needed for the classical-DDM analysis."""
    return load_raw(path, usecols=LOAD_COLUMNS, check_hash=check_hash)
