from src import train
from src import features
from src import sequence_features


def test_train_runs():
    # Ensure preprocessing steps run
    features.build_baseline("0022400001")
    sequence_features.add_sequence_feats("0022400001")

    # Should run without raising
    train.main("0022400001")
