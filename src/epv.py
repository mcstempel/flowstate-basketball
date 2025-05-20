"""Utilities for EPV prediction and swing calculation."""

from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
import pandas as pd


_MODELS_DIR = "models"
_BASELINE_MODEL = os.path.join(_MODELS_DIR, "baseline.pkl")
_SEQUENCE_MODEL = os.path.join(_MODELS_DIR, "sequence.pkl")


def _load_model(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def _prep_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["points_scored"], errors="ignore")
    leak_cols = [c for c in df.columns if c.endswith("_team_id")]
    df = df.drop(columns=leak_cols)
    X = pd.get_dummies(df, drop_first=True)
    return X


def calculate_epv(game_id: str) -> pd.DataFrame:
    """Return baseline and sequence EPV for each possession of ``game_id``."""
    base_csv = f"data/baseline_{game_id}.csv"
    seq_csv = f"data/sequence_{game_id}.csv"
    if not os.path.exists(base_csv) or not os.path.exists(seq_csv):
        raise FileNotFoundError(
            "Required feature CSVs not found. Run features.py and sequence_features.py first."
        )

    base_df = pd.read_csv(base_csv)
    seq_df = pd.read_csv(seq_csv)

    # Prepare features
    X_base = _prep_features(base_df)
    X_seq = _prep_features(seq_df)

    # Load models
    m_base = _load_model(_BASELINE_MODEL)
    m_seq = _load_model(_SEQUENCE_MODEL)

    base_probs = m_base.predict_proba(X_base)
    seq_probs = m_seq.predict_proba(X_seq)

    base_epv = (base_probs * np.arange(base_probs.shape[1])).sum(axis=1)
    seq_epv = (seq_probs * np.arange(seq_probs.shape[1])).sum(axis=1)

    return pd.DataFrame(
        {
            "poss_id": base_df["poss_id"],
            "epv_baseline": base_epv,
            "epv_sequence": seq_epv,
        }
    )


def calculate_swing(game_id: str) -> pd.DataFrame:
    """Return top-20 possessions with largest EPV swing."""
    epv_df = calculate_epv(game_id)
    epv_df["swing"] = epv_df["epv_sequence"] - epv_df["epv_baseline"]
    return epv_df.reindex(columns=["poss_id", "swing"]).iloc[
        epv_df["swing"].abs().sort_values(ascending=False).index
    ].head(20)

