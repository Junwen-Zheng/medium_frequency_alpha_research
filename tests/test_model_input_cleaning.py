import numpy as np
import pandas as pd

from src.models import fit_ridge


def test_ridge_drops_non_finite_model_inputs_and_returns_finite_predictions():
    train = pd.DataFrame(
        {
            "f1": [0.1, 0.2, np.inf, 0.4, 0.5],
            "f2": [1.0, 1.1, 1.2, np.nan, 1.4],
            "target": [0.01, 0.02, 0.03, 0.04, -0.01],
        }
    )
    test = pd.DataFrame(
        {
            "f1": [0.15, np.inf, 0.35],
            "f2": [1.05, 1.20, 1.30],
            "target": [0.01, 0.02, -0.01],
        }
    )

    result = fit_ridge(train, test, ["f1", "f2"], "target")

    assert result.name == "ridge"
    assert np.isfinite(result.predictions.to_numpy()).all()
    assert len(result.predictions) == 2
