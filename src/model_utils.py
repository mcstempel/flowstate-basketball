import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from .train import load_csv, prep_xy


def _fit(tag: str, game_id: str):
    """Return trained model, poss_ids, categories, and feature matrix."""
    df = load_csv(tag, game_id)
    X, y, num_cls = prep_xy(df)
    categories = sorted(df["points_scored"].clip(0, 3).unique())
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=num_cls,
        n_estimators=60,
        max_depth=3,
        learning_rate=0.2,
        subsample=0.8,
        eval_metric="mlogloss",
        verbosity=0,
    )
    model.fit(X, y)
    return model, df["poss_id"], categories, X


def _epv_df(tag: str, game_id: str) -> pd.DataFrame:
    model, poss_ids, cats, X = _fit(tag, game_id)
    proba = model.predict_proba(X)
    epv = proba.dot(np.array(cats))
    return pd.DataFrame({"poss_id": poss_ids, "epv": epv})


def sequence_epv(game_id: str) -> pd.DataFrame:
    """EPV values using the sequence feature model."""
    return _epv_df("sequence", game_id)


def baseline_epv(game_id: str) -> pd.DataFrame:
    return _epv_df("baseline", game_id)


def swing(game_id: str, top_n: int = 20) -> pd.DataFrame:
    seq = sequence_epv(game_id)
    base = baseline_epv(game_id)
    merged = seq.merge(base, on="poss_id", suffixes=("_seq", "_base"))
    merged["swing"] = (merged["epv_seq"] - merged["epv_base"]).abs()
    merged = merged.sort_values("swing", ascending=False).head(top_n)
    return merged
