import os
from src import features, sequence_features, train, epv

GID = "0022400001"


def prepare_models():
    features.build_baseline(GID)
    sequence_features.add_sequence_feats(GID)
    train.main(GID)


def test_epv_and_swing():
    prepare_models()
    assert os.path.exists("models/baseline.pkl")
    assert os.path.exists("models/sequence.pkl")

    epv_df = epv.calculate_epv(GID)
    assert list(epv_df.columns) == ["poss_id", "epv_baseline", "epv_sequence"]
    assert len(epv_df) == 3

    swing_df = epv.calculate_swing(GID)
    assert "swing" in swing_df.columns
    assert len(swing_df) <= len(epv_df)
