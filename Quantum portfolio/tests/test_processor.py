"""
tests/test_processor.py
------------------------
Unit tests cho src/data/processor.py.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.processor import (
    PortfolioData,
    compute_covariance,
    compute_mu,
    compute_returns,
    ensure_positive_semidefinite,
    process,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_prices():
    """200 ngày giá giả lập, 4 cổ phiếu."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-01", periods=200, freq="B")
    data = {t: 100 + rng.normal(0, 1, 200).cumsum()
            for t in ["A", "B", "C", "D"]}
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_config(tmp_path):
    content = f"""
data:
  default_tickers: [A, B, C, D]
  period_years: 1
  interval: "1d"
  raw_dir: "{tmp_path}/raw"
  processed_dir: "{tmp_path}/processed"
  cache_days: 1
processing:
  returns_method: log
  annualize_factor: 252
  min_data_points: 50
  max_missing_pct: 0.05
"""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(content)
    return str(cfg)


# ── Tests: compute_returns ────────────────────────────────────────────────────

def test_log_returns_shape(sample_prices):
    ret = compute_returns(sample_prices, method="log")
    assert ret.shape == (199, 4)  # 200 rows → 199 returns


def test_simple_returns_shape(sample_prices):
    ret = compute_returns(sample_prices, method="simple")
    assert ret.shape == (199, 4)


def test_log_returns_no_nan(sample_prices):
    ret = compute_returns(sample_prices, method="log")
    assert not ret.isna().any().any()


def test_invalid_method_raises(sample_prices):
    with pytest.raises(ValueError, match="method"):
        compute_returns(sample_prices, method="wrong")


# ── Tests: compute_mu ─────────────────────────────────────────────────────────

def test_mu_length(sample_prices):
    ret = compute_returns(sample_prices)
    mu = compute_mu(ret)
    assert len(mu) == 4


def test_mu_annualized(sample_prices):
    ret = compute_returns(sample_prices)
    mu_daily = ret.mean()
    mu_annual = compute_mu(ret, annualize_factor=252)
    assert np.allclose(mu_annual.values, mu_daily.values * 252)


# ── Tests: compute_covariance ─────────────────────────────────────────────────

def test_cov_shape(sample_prices):
    ret = compute_returns(sample_prices)
    cov = compute_covariance(ret)
    assert cov.shape == (4, 4)


def test_cov_symmetric(sample_prices):
    ret = compute_returns(sample_prices)
    cov = compute_covariance(ret)
    assert np.allclose(cov.values, cov.values.T)


def test_cov_diagonal_positive(sample_prices):
    ret = compute_returns(sample_prices)
    cov = compute_covariance(ret)
    assert (np.diag(cov.values) > 0).all()


# ── Tests: ensure_positive_semidefinite ───────────────────────────────────────

def test_psd_fix():
    """Matrix với eigenvalue âm nhỏ phải được sửa về PSD."""
    tickers = ["A", "B", "C"]
    arr = np.array([[1.0, 0.9, 0.8],
                    [0.9, 1.0, 0.9],
                    [0.8, 0.9, 1.0]])
    # Inject eigenvalue âm nhỏ
    arr[0, 0] -= 0.01
    cov = pd.DataFrame(arr, index=tickers, columns=tickers)
    cov_fixed = ensure_positive_semidefinite(cov)
    eigvals = np.linalg.eigvalsh(cov_fixed.values)
    assert eigvals.min() >= -1e-8


def test_psd_valid_unchanged():
    """Matrix PSD hợp lệ không nên bị thay đổi đáng kể."""
    tickers = ["A", "B"]
    arr = np.array([[1.0, 0.3], [0.3, 1.0]])
    cov = pd.DataFrame(arr, index=tickers, columns=tickers)
    cov_fixed = ensure_positive_semidefinite(cov)
    assert np.allclose(cov.values, cov_fixed.values, atol=1e-10)


# ── Tests: process (pipeline đầy đủ) ─────────────────────────────────────────

def test_process_returns_portfolio_data(sample_prices, sample_config):
    result = process(sample_prices, config_path=sample_config, save=False)
    assert isinstance(result, PortfolioData)


def test_process_n_assets(sample_prices, sample_config):
    result = process(sample_prices, config_path=sample_config, save=False)
    assert result.n_assets == 4


def test_process_mu_cov_aligned(sample_prices, sample_config):
    result = process(sample_prices, config_path=sample_config, save=False)
    assert list(result.mu.index) == list(result.cov.index)
    assert list(result.cov.index) == list(result.cov.columns)


def test_process_saves_files(sample_prices, sample_config, tmp_path):
    process(sample_prices, config_path=sample_config, save=True)
    processed_dir = tmp_path / "processed"
    assert (processed_dir / "returns.csv").exists()
    assert (processed_dir / "covariance.csv").exists()
    assert (processed_dir / "mu.csv").exists()


def test_process_summary(sample_prices, sample_config):
    result = process(sample_prices, config_path=sample_config, save=False)
    summary = result.summary()
    assert "Expected Return (ann.)" in summary.columns
    assert "Volatility (ann.)" in summary.columns
    assert "Sharpe (rf=0)" in summary.columns
    assert len(summary) == 4