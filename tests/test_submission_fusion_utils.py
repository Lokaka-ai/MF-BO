import numpy as np
import pandas as pd

from utils.submission_fusion_utils import EnsembleConfig, detect_format, fuse_average


def test_detect_format_keeps_short_nonnumeric_date_as_passthrough():
    comments = [f"comment {i} " + "x" * 220 for i in range(6)]
    first = pd.DataFrame(
        {
            "Insult": [0.1, 0.1, 0.2, 0.2, 0.1, 0.2],
            "Date": [None] + [f"2012061822225{i}Z" for i in range(5)],
            "Comment": comments,
        }
    )
    second = first.copy()
    second["Insult"] = [0.3, 0.3, 0.4, 0.4, 0.3, 0.4]

    fmt = detect_format(first, EnsembleConfig())

    assert fmt["id_col_names"] == []
    assert fmt["pred_col_names"] == ["Insult"]
    assert fmt["passthrough_col_names"] == ["Date", "Comment"]
    assert fmt["fusion"] == "average"
    fused = fuse_average(
        [first[fmt["pred_col_names"]], second[fmt["pred_col_names"]]],
        np.array([0.5, 0.5]),
    )
    np.testing.assert_allclose(fused[:, 0], [0.2, 0.2, 0.3, 0.3, 0.2, 0.3])


def test_detect_format_preserves_text_prediction_when_no_numeric_candidate_exists():
    submission = pd.DataFrame(
        {
            "id": ["0_0", "0_1", "0_2"],
            "after": ["Теперь", "Mountain", "."],
        }
    )

    fmt = detect_format(submission, EnsembleConfig())

    assert fmt["id_col_names"] == ["id"]
    assert fmt["pred_col_names"] == ["after"]
    assert fmt["passthrough_col_names"] == []
    assert fmt["fusion"] == "text_vote"
