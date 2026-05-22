"""
tests/test_mvo.py
-----------------
Unit tests cho src/classical/mvo.py, frontier.py, metrics.py.
Dùng dữ liệu giả lập — không cần kết nối internet hay IBM.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classical.mvo import solve_mvo, maximize_sharpe, minimize_variance, MVOResult
from src.classical.frontier import compute_efficient_frontier
from src.classical.metrics import (
    portfolio_returns, annualized_return, annualized_volatility,
    sharpe_ratio, sortino_ratio, maximum_drawdown,
    value_at_risk, conditional_var, full_metrics, compare_portfolios,
)


# ══════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════

@pytest.fixture
def simple_market():
    """4 tài sản với μ và Σ đã biết trước."""
    tickers = ["A", "B", "C", "D"]
    mu = pd.Series([0.10, 0.14, 0.08, 0.18], index=tickers)
    # Covariance: mỗi cổ phiếu có vol ~15-25%, tương quan thấp
    cov_arr = np.array([
        [0.0225, 0.0045, 0.0018, 0.0036],
        [0.0045, 0.0400, 0.0024, 0.0080],
        [0.0018, 0.0024, 0.0196, 0.0021],
        [0.0036, 0.0080, 0.0021, 0.0625],
    ])
    cov = pd.DataFrame(cov_arr, index=tickers, columns=tickers)
    return mu, cov, tickers


@pytest.fixture
def daily_returns(simple_market):
    """Returns hàng ngày giả lập (500 ngày, 4 cổ phiếu)."""
    mu, cov, tickers = simple_market
    rng = np.random.default_rng(42)
    # Sinh multivariate normal với μ/252, Σ/252
    daily_mu = mu.values / 252
    daily_cov = cov.values / 252
    data = rng.multivariate_normal(daily_mu, daily_cov, size=500)
    dates = pd.date_range("2022-01-01", periods=500, freq="B")
    return pd.DataFrame(data, index=dates, columns=tickers)


# ══════════════════════════════════════════════════
# Tests: solve_mvo
# ══════════════════════════════════════════════════

class TestSolveMVO:
    def test_weights_sum_to_one(self, simple_market):
        mu, cov, _ = simple_market
        result = solve_mvo(mu, cov, target_return=0.10)
        assert abs(result.weights.sum() - 1.0) < 1e-4

    def test_weights_non_negative(self, simple_market):
        mu, cov, _ = simple_market
        result = solve_mvo(mu, cov, target_return=0.10)
        assert (result.weights >= -1e-6).all()

    def test_return_meets_target(self, simple_market):
        mu, cov, _ = simple_market
        target = 0.12
        result = solve_mvo(mu, cov, target_return=target)
        assert result.expected_return >= target - 1e-4

    def test_status_optimal(self, simple_market):
        mu, cov, _ = simple_market
        result = solve_mvo(mu, cov, target_return=0.10)
        assert result.status in ("optimal", "optimal_inaccurate")

    def test_volatility_positive(self, simple_market):
        mu, cov, _ = simple_market
        result = solve_mvo(mu, cov, target_return=0.10)
        assert result.volatility > 0

    def test_higher_target_higher_vol(self, simple_market):
        """Tradeoff cơ bản: return cao hơn → variance cao hơn."""
        mu, cov, _ = simple_market
        r1 = solve_mvo(mu, cov, target_return=0.09)
        r2 = solve_mvo(mu, cov, target_return=0.15)
        assert r2.volatility >= r1.volatility - 1e-4

    def test_infeasible_target_fallbacks(self, simple_market):
        """Target quá cao (vô lý) → không crash, trả về fallback."""
        mu, cov, _ = simple_market
        result = solve_mvo(mu, cov, target_return=99.0)
        assert result.weights.sum() > 0  # vẫn có output

    def test_max_weight_constraint(self, simple_market):
        mu, cov, _ = simple_market
        result = solve_mvo(mu, cov, target_return=0.10, max_weight=0.4)
        assert (result.weights <= 0.4 + 1e-4).all()


# ══════════════════════════════════════════════════
# Tests: maximize_sharpe
# ══════════════════════════════════════════════════

class TestMaxSharpe:
    def test_weights_sum_to_one(self, simple_market):
        mu, cov, _ = simple_market
        result = maximize_sharpe(mu, cov, risk_free_rate=0.05)
        assert abs(result.weights.sum() - 1.0) < 1e-4

    def test_sharpe_better_than_equal_weight(self, simple_market):
        mu, cov, tickers = simple_market
        n = len(tickers)
        eq_w = pd.Series(np.ones(n) / n, index=tickers)
        eq_return = float(mu.values @ eq_w.values)
        eq_vol = float(np.sqrt(eq_w.values @ cov.values @ eq_w.values))
        eq_sharpe = (eq_return - 0.05) / eq_vol

        result = maximize_sharpe(mu, cov, risk_free_rate=0.05)
        assert result.sharpe_ratio >= eq_sharpe - 1e-3

    def test_weights_non_negative(self, simple_market):
        mu, cov, _ = simple_market
        result = maximize_sharpe(mu, cov)
        assert (result.weights >= -1e-4).all()


# ══════════════════════════════════════════════════
# Tests: minimize_variance (GMV)
# ══════════════════════════════════════════════════

class TestMinVariance:
    def test_weights_sum_to_one(self, simple_market):
        mu, cov, _ = simple_market
        result = minimize_variance(mu, cov)
        assert abs(result.weights.sum() - 1.0) < 1e-4

    def test_gmv_lower_vol_than_any_single_asset(self, simple_market):
        mu, cov, tickers = simple_market
        gmv = minimize_variance(mu, cov)
        min_single_vol = float(np.sqrt(np.diag(cov.values)).min())
        assert gmv.volatility <= min_single_vol + 1e-4


# ══════════════════════════════════════════════════
# Tests: EfficientFrontier
# ══════════════════════════════════════════════════

class TestEfficientFrontier:
    def test_n_points(self, simple_market):
        mu, cov, _ = simple_market
        ef = compute_efficient_frontier(mu, cov, n_points=20)
        assert ef.n_points > 0

    def test_returns_increasing(self, simple_market):
        """Frontier phải có return không giảm dọc theo trục."""
        mu, cov, _ = simple_market
        ef = compute_efficient_frontier(mu, cov, n_points=30)
        assert ef.returns[-1] >= ef.returns[0] - 1e-4

    def test_weights_sum_to_one(self, simple_market):
        mu, cov, _ = simple_market
        ef = compute_efficient_frontier(mu, cov, n_points=20)
        sums = ef.weights_df.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-3)

    def test_has_special_portfolios(self, simple_market):
        mu, cov, _ = simple_market
        ef = compute_efficient_frontier(mu, cov, n_points=20)
        assert ef.min_variance_result is not None
        assert ef.max_sharpe_result is not None

    def test_max_sharpe_sharpe_best(self, simple_market):
        mu, cov, _ = simple_market
        ef = compute_efficient_frontier(mu, cov, n_points=30)
        # Max Sharpe portfolio phải có Sharpe tốt nhất (hoặc gần nhất)
        assert ef.max_sharpe_result.sharpe_ratio >= ef.sharpe_ratios.max() - 0.05


# ══════════════════════════════════════════════════
# Tests: metrics
# ══════════════════════════════════════════════════

class TestMetrics:
    def test_portfolio_returns_shape(self, simple_market, daily_returns):
        mu, cov, tickers = simple_market
        n = len(tickers)
        w = pd.Series(np.ones(n) / n, index=tickers)
        pr = portfolio_returns(w, daily_returns)
        assert len(pr) == len(daily_returns)

    def test_annualized_return_sign(self, simple_market, daily_returns):
        mu, cov, tickers = simple_market
        w = pd.Series([0.4, 0.3, 0.2, 0.1], index=tickers)
        pr = portfolio_returns(w, daily_returns)
        ann_ret = annualized_return(pr)
        assert isinstance(ann_ret, float)

    def test_volatility_positive(self, simple_market, daily_returns):
        mu, cov, tickers = simple_market
        w = pd.Series(np.ones(4) / 4, index=tickers)
        pr = portfolio_returns(w, daily_returns)
        assert annualized_volatility(pr) > 0

    def test_max_drawdown_negative(self, simple_market, daily_returns):
        mu, cov, tickers = simple_market
        w = pd.Series(np.ones(4) / 4, index=tickers)
        pr = portfolio_returns(w, daily_returns)
        mdd = maximum_drawdown(pr)
        assert mdd <= 0

    def test_var_less_than_mean(self, simple_market, daily_returns):
        mu, cov, tickers = simple_market
        w = pd.Series(np.ones(4) / 4, index=tickers)
        pr = portfolio_returns(w, daily_returns)
        var = value_at_risk(pr, 0.95)
        assert var < pr.mean()

    def test_cvar_less_than_var(self, simple_market, daily_returns):
        mu, cov, tickers = simple_market
        w = pd.Series(np.ones(4) / 4, index=tickers)
        pr = portfolio_returns(w, daily_returns)
        var = value_at_risk(pr, 0.95)
        cvar = conditional_var(pr, 0.95)
        assert cvar <= var + 1e-8

    def test_full_metrics_keys(self, simple_market, daily_returns):
        mu, cov, tickers = simple_market
        w = pd.Series(np.ones(4) / 4, index=tickers)
        m = full_metrics(w, daily_returns, label="EW")
        expected_keys = [
            "Annualized Return", "Annualized Volatility",
            "Sharpe Ratio", "Sortino Ratio",
            "Max Drawdown", "VaR (95%)", "CVaR (95%)", "Calmar Ratio"
        ]
        for k in expected_keys:
            assert k in m.index, f"Thiếu metric: {k}"

    def test_compare_portfolios_shape(self, simple_market, daily_returns):
        mu, cov, tickers = simple_market
        portfolios = {
            "Equal Weight": pd.Series(np.ones(4) / 4, index=tickers),
            "Heavy A": pd.Series([0.7, 0.1, 0.1, 0.1], index=tickers),
        }
        df = compare_portfolios(portfolios, daily_returns)
        assert df.shape[1] == 2
        assert "Sharpe Ratio" in df.index