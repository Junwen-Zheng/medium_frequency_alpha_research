import pandas as pd

from src.reporting import write_report


def test_generated_report_includes_transaction_cost_sensitivity(tmp_path):
    report_path = tmp_path / "generated.md"

    cost_sensitivity = pd.DataFrame(
        [
            {
                "cost_bps": 0.0,
                "annualized_return": 0.01,
                "average_daily_turnover": 0.5,
            },
            {
                "cost_bps": 25.0,
                "annualized_return": -0.02,
                "average_daily_turnover": 0.5,
            },
        ]
    )

    write_report(
        report_path,
        title="Toy Report",
        data_quality={},
        validation_summaries={},
        test_summaries={},
        backtest_metrics={"selected_model": "mock"},
        decay_table=pd.DataFrame(),
        cost_sensitivity=cost_sensitivity,
    )

    text = report_path.read_text()

    assert "## Transaction-cost sensitivity" in text
    assert "cost_bps" in text
    assert "25" in text
    assert "average_daily_turnover" in text


def test_generated_report_includes_rebalance_frequency_sensitivity(tmp_path):
    report_path = tmp_path / "generated_rebalance.md"

    rebalance_frequency_sensitivity = pd.DataFrame(
        [
            {
                "rebalance_frequency": "daily",
                "cost_bps": 10.0,
                "net_sharpe": 0.1,
                "average_daily_turnover": 0.7,
            },
            {
                "rebalance_frequency": "weekly",
                "cost_bps": 10.0,
                "net_sharpe": 0.2,
                "average_daily_turnover": 0.3,
            },
        ]
    )

    write_report(
        report_path,
        title="Toy Report",
        data_quality={},
        validation_summaries={},
        test_summaries={},
        backtest_metrics={"selected_model": "mock"},
        decay_table=pd.DataFrame(),
        rebalance_frequency_sensitivity=rebalance_frequency_sensitivity,
    )

    text = report_path.read_text()

    assert "## Rebalance-frequency sensitivity" in text
    assert "rebalance_frequency" in text
    assert "weekly" in text
    assert "net_sharpe" in text
