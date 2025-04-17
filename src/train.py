#!/usr/bin/env python3
"""
train.py
--------
Train two XGBoost models:

1. baseline  – features from baseline_<game_id>.csv   (memory‑0)
2. sequence  – features from sequence_<game_id>.csv  (memory‑3)

and report log‑loss + % improvement.

Usage
-----
python src/train.py 0022400001
# (game_id argument is optional; defaults to 0022400001)

Requires: pandas, scikit‑learn, xgboost (already in requirements.txt)
"""
import os, sys, json
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# --------------------------------------------------------------------------- #
#  helper functions                                                           #
# --------------------------------------------------------------------------- #

def load_csv(tag: str, gid: str) -> pd.DataFrame:
    path = f"data/{tag}_{gid}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found – run Day 3/4 scripts first.")
    return pd.read_csv(path)


def prep_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    * one‑hot encode categoricals
    * drop team_id columns (they leak target info in a single‑game dataset)
    * return X, y where y is remapped to contiguous 0..k‑1 labels
    """
    y_orig = df["points_scored"].clip(0, 3)          # cap at 3 for demo data
    df = df.drop(columns=["points_scored"])

    # drop obvious leakage / high‑cardinality IDs
    leak_cols = [c for c in df.columns if c.endswith("_team_id")]
    df = df.drop(columns=leak_cols)

    X = pd.get_dummies(df, drop_first=True)

    # robust label remap
    y_cat = y_orig.astype("category")
    y_contig = y_cat.cat.codes            # guarantees 0..k‑1 contiguous ints
    num_cls = int(y_contig.max()) + 1     # k

    return X, y_contig, num_cls


def train_xgb(X, y, num_cls, seed=42):
    """
    Train / test split (25%).  If dataset too small, train & eval on same set.
    Returns log‑loss.
    """
    if len(X) < 8:                       # tiny demo case
        X_train = X_test = X
        y_train = y_test = y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seed
        )

    clf = XGBClassifier(
        objective="multi:softprob",
        num_class=num_cls,
        n_estimators=60,
        max_depth=3,
        learning_rate=0.2,
        subsample=0.8,
        eval_metric="mlogloss",
        verbosity=0,
    )
    clf.fit(X_train, y_train)
    y_hat = clf.predict_proba(X_test)
    return log_loss(y_test, y_hat)


# --------------------------------------------------------------------------- #
#  main driver                                                                #
# --------------------------------------------------------------------------- #

def main(game_id: str = "0022400001"):
    # -------- baseline (memory‑0) ------------------------------------------ #
    base_df = load_csv("baseline", game_id)
    Xb, yb, k_base = prep_xy(base_df)
    ll_base = train_xgb(Xb, yb, k_base)

    # -------- sequence (memory‑3) ------------------------------------------ #
    seq_df = load_csv("sequence", game_id)
    Xs, ys, k_seq = prep_xy(seq_df)
    ll_seq = train_xgb(Xs, ys, k_seq)

    # -------- report ------------------------------------------------------- #
    pct_improve = (ll_base - ll_seq) / ll_base * 100 if ll_base else 0.0
    report = {
        "baseline_logloss": round(ll_base, 5),
        "sequence_logloss": round(ll_seq, 5),
        "improvement_%":    round(pct_improve, 2),
        "classes_base":     k_base,
        "classes_seq":      k_seq
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    gid = sys.argv[1] if len(sys.argv) > 1 else "0022400001"
    main(gid)
