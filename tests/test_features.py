import os
import pandas as pd
from src import features


def test_clock_to_seconds():
    assert features.clock_to_seconds('PT11M32.00S') == 11 * 60 + 32
    assert features.clock_to_seconds('PT0M05.00S') == 5
    assert features.clock_to_seconds('bad_string') == 0


def test_shot_bucket():
    assert features.shot_bucket(None) == 'no_shot'
    assert features.shot_bucket(2.5) == 'restricted_area'
    assert features.shot_bucket(10) == 'paint'
    assert features.shot_bucket(16) == 'midrange'
    assert features.shot_bucket(24) == 'corner_three'
    assert features.shot_bucket(27) == 'non_corner_three'


def test_build_baseline():
    out_csv = features.build_baseline('0022400001')
    assert os.path.exists(out_csv)
    df = pd.read_csv(out_csv)
    expected = pd.DataFrame({
        'poss_id': [1, 2, 3],
        'period': [1, 1, 1],
        'clock_start_sec': [692, 645, 603],
        'clock_end_sec': [664, 623, 586],
        'offense_team_id': [1610612749, 1610612738, 1610612749],
        'defense_team_id': [1610612738, 1610612749, 1610612738],
        'score_diff_start': [0, 2, 2],
        'shot_bucket': ['paint', 'non_corner_three', 'restricted_area'],
        'points_scored': [2, 0, 3],
    })
    pd.testing.assert_frame_equal(df.reset_index(drop=True), expected)
