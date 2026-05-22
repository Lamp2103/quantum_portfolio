"""
src/data/processor.py
---------------------
Xử lý dữ liệu giá cổ phiếu thành các thống kê tài chính:
  - Log returns / Simple returns
  - Expected return (μ) hàng năm
  - Covariance matrix (Σ) hàng năm
  - Correlation matrix
  - Risk metrics cơ bản
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass
class PortfolioData:
    """
    Container chứa toàn bộ dữ liệu đã xử lý, sẵn sàng đưa vào solver.

    Attributes
    ----------
    prices : pd.DataFrame
        Giá gốc (ngày × ticker).
    returns : pd.DataFrame
        Returns theo ngày (ngày × ticker).
    mu : pd.Series
        Expected return hàng năm (ticker,).
    cov : pd.DataFrame
        Covariance matrix hàng năm (ticker × ticker).
    corr : pd.DataFrame
        Correlation matrix (ticker × ticker).
    tickers : list[str]
        Danh sách mã cổ phiếu hợp lệ.
    n_assets : int
        Số tài sản.
    n_days : int
        Số ngày dữ liệu.
    returns_method : str
        'log' hoặc 'simple'.
    """

    prices: pd.DataFrame
    returns: pd.DataFrame
    mu: pd.Series
    cov: pd.DataFrame
    corr: pd.DataFrame
    tickers: list
    n_assets: int
    n_days: int
    returns_method: str

    def summary(self) -> pd.DataFrame:
        """Bảng tóm tắt: return kỳ vọng, volatility, Sharpe (giả định rf=0) mỗi cổ phiếu."""
        vol = pd.Series(np.sqrt(np.diag(self.cov.values)), index=self.tickers)
        sharpe = self.mu / vol
        return pd.DataFrame(
            {
                "Expected Return (ann.)": self.mu.round(4),
                "Volatility (ann.)": vol.round(4),
                "Sharpe (rf=0)": sharpe.round(4),
            }
        )


def _load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding='utf-8') as f:
        return yaml.safe_load(f)


def compute_returns(
    prices: pd.DataFrame,
    method: str = "log",
) -> pd.DataFrame:
    """
    Tính returns từ chuỗi giá.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame giá (ngày × ticker).
    method : str
        'log'    → log(P_t / P_{t-1})
        'simple' → (P_t - P_{t-1}) / P_{t-1}

    Returns
    -------
    pd.DataFrame
        Returns hàng ngày, đã bỏ hàng NaN đầu tiên.
    """
    if method == "log":
        returns = np.log(prices / prices.shift(1))
    elif method == "simple":
        returns = prices.pct_change()
    else:
        raise ValueError(f"method phải là 'log' hoặc 'simple', nhận được: '{method}'")

    returns = returns.iloc[1:]  # bỏ hàng NaN đầu tiên
    return returns


def compute_mu(returns: pd.DataFrame, annualize_factor: int = 252) -> pd.Series:
    """
    Expected return hàng năm: μ = mean(r) × annualize_factor.

    Parameters
    ----------
    returns : pd.DataFrame
        Returns hàng ngày.
    annualize_factor : int
        252 cho ngày giao dịch, 52 cho tuần, 12 cho tháng.

    Returns
    -------
    pd.Series
        Expected return hàng năm, index = ticker.
    """
    return returns.mean() * annualize_factor


def compute_covariance(
    returns: pd.DataFrame,
    annualize_factor: int = 252,
) -> pd.DataFrame:
    """
    Covariance matrix hàng năm: Σ = cov(r) × annualize_factor.

    Parameters
    ----------
    returns : pd.DataFrame
        Returns hàng ngày.
    annualize_factor : int
        Hệ số nhân để annualize.

    Returns
    -------
    pd.DataFrame
        Covariance matrix hàng năm (ticker × ticker).
    """
    return returns.cov() * annualize_factor


def ensure_positive_semidefinite(cov: pd.DataFrame, epsilon: float = 1e-8) -> pd.DataFrame:
    """
    Đảm bảo covariance matrix là positive semi-definite (PSD).
    Nếu có eigenvalue âm nhỏ do lỗi số học, dịch chuyển về 0.

    Parameters
    ----------
    cov : pd.DataFrame
        Covariance matrix gốc.
    epsilon : float
        Ngưỡng: eigenvalue < -epsilon được coi là không hợp lệ.

    Returns
    -------
    pd.DataFrame
        Covariance matrix đã được điều chỉnh về PSD.
    """
    cov_arr = cov.values
    eigenvalues = np.linalg.eigvalsh(cov_arr)

    if eigenvalues.min() < -epsilon:
        logger.warning(
            f"Covariance matrix có eigenvalue âm ({eigenvalues.min():.2e}). "
            "Đang điều chỉnh về PSD..."
        )
        # Nearest PSD: clip eigenvalue về 0
        eigvals, eigvecs = np.linalg.eigh(cov_arr)
        eigvals = np.maximum(eigvals, 0)
        cov_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Đảm bảo symmetric hoàn toàn
        cov_psd = (cov_psd + cov_psd.T) / 2
        return pd.DataFrame(cov_psd, index=cov.index, columns=cov.columns)

    return cov


def process(
    prices: pd.DataFrame,
    config_path: str = "config.yaml",
    save: bool = True,
) -> PortfolioData:
    """
    Pipeline xử lý đầy đủ: prices → PortfolioData.

    Parameters
    ----------
    prices : pd.DataFrame
        Giá gốc từ fetcher.fetch_prices().
    config_path : str
        Đường dẫn config.yaml.
    save : bool
        Nếu True, lưu returns và covariance ra file CSV.

    Returns
    -------
    PortfolioData
        Container với đầy đủ thống kê tài chính.
    """
    cfg = _load_config(config_path)
    proc_cfg = cfg["processing"]
    data_cfg = cfg["data"]

    method = proc_cfg["returns_method"]
    annualize = proc_cfg["annualize_factor"]
    min_points = proc_cfg["min_data_points"]
    max_missing = proc_cfg["max_missing_pct"]

    logger.info(f"Bắt đầu xử lý: {prices.shape[1]} cổ phiếu, {prices.shape[0]} ngày")

    # --- 1. Lọc cổ phiếu có quá nhiều NaN ---
    missing_pct = prices.isna().mean()
    valid_tickers = missing_pct[missing_pct <= max_missing].index.tolist()
    dropped = set(prices.columns) - set(valid_tickers)
    if dropped:
        logger.warning(f"Loại bỏ do quá nhiều missing ({max_missing*100:.0f}%): {dropped}")
    prices = prices[valid_tickers]

    # --- 2. Forward-fill NaN (ngày không giao dịch) ---
    prices = prices.ffill().bfill()

    # --- 3. Tính returns ---
    returns = compute_returns(prices, method=method)

    # Kiểm tra số ngày đủ tối thiểu
    if len(returns) < min_points:
        raise ValueError(
            f"Chỉ có {len(returns)} ngày dữ liệu hợp lệ, "
            f"cần ít nhất {min_points} ngày."
        )

    logger.info(f"Returns: {returns.shape[0]} ngày, method={method}")

    # --- 4. Tính μ và Σ ---
    mu = compute_mu(returns, annualize)
    cov = compute_covariance(returns, annualize)
    cov = ensure_positive_semidefinite(cov)
    corr = returns.corr()

    logger.info(
        f"μ range: [{mu.min():.2%}, {mu.max():.2%}] | "
        f"Volatility range: [{np.sqrt(np.diag(cov.values)).min():.2%}, "
        f"{np.sqrt(np.diag(cov.values)).max():.2%}]"
    )

    # --- 5. Lưu file (tùy chọn) ---
    if save:
        proc_dir = Path(data_cfg["processed_dir"])
        proc_dir.mkdir(parents=True, exist_ok=True)

        returns.to_csv(proc_dir / "returns.csv")
        cov.to_csv(proc_dir / "covariance.csv")
        mu.to_frame("expected_return").to_csv(proc_dir / "mu.csv")
        corr.to_csv(proc_dir / "correlation.csv")
        logger.info(f"Đã lưu dữ liệu xử lý vào {proc_dir}/")

    return PortfolioData(
        prices=prices,
        returns=returns,
        mu=mu,
        cov=cov,
        corr=corr,
        tickers=valid_tickers,
        n_assets=len(valid_tickers),
        n_days=len(returns),
        returns_method=method,
    )