import os
import pandas as pd
from src import features
from src import sequence_features


def build_data():
    base_path = features.build_baseline("0022400001")
    seq_path = sequence_features.add_sequence_feats("0022400001")
    base_df = pd.read_csv(base_path)
    seq_df = pd.read_csv(seq_path)
    return base_df, seq_df


def test_sequence_calculations():
    base_df, seq_df = build_data()

    # prev points columns
    assert list(seq_df["prev_pts_1"]) == [0, 2, 0]
    assert list(seq_df["prev_pts_2"]) == [0, 0, 2]
    assert list(seq_df["prev_pts_3"]) == [0, 0, 0]

    # prev bucket
    assert list(seq_df["prev_bucket_1"]) == ["none", "paint", "non_corner_three"]

    # tempo calculations
    assert list(seq_df["tempo_sec"]) == [28, 22, 17]
    assert pd.isna(seq_df.loc[0, "tempo_mean_last3"])
    assert seq_df.loc[1, "tempo_mean_last3"] == 28.0
    assert round(seq_df.loc[2, "tempo_mean_last3"], 1) == 25.0

    # streak flag
    assert list(seq_df["streak_scored_last3"]) == [0, 0, 0]
