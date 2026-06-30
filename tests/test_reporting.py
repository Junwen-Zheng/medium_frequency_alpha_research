import pandas as pd

from src.reporting import write_report


def test_report_includes_walk_forward_section(tmp_path):
    report_path = tmp_path / "report.md"
    walk_forward_metrics = pd.DataFrame(
        [
            {
                "fold_id": 1,
                "selected_model": "ridge",
                "test_mean_rank_ic": 0.01,
                "test_n_days": 10,
            }
        ]
    )

    write_report(
        path=report_path,
        title="Test Report",
        data_quality={"rows": 100},
        validation_summaries={},
        test_summaries={},
        backtest_metrics={},
        decay_table=pd.DataFrame(),
        family_comparison=pd.DataFrame(),
        regime_ic=pd.DataFrame(),
        walk_forward_metrics=walk_forward_metrics,
        walk_forward_diagnostics=pd.DataFrame(),
        walk_forward_summary=pd.DataFrame(
            [
                {
                    "n_folds": 1,
                    "positive_test_fold_rate": 1.0,
                    "selected_model_counts": "ridge:1",
                }
            ]
        ),
    )

    text = report_path.read_text()
    assert "## Walk-forward validation" in text
    assert "Aggregate robustness summary" in text
    assert "positive_test_fold_rate" in text
    assert "ridge" in text
    assert "outputs/walk_forward_fold_diagnostics.csv" in text
