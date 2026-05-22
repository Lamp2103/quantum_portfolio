"""
src/classical/frontier.py
-------------------------
Quét nhiều mức return target để xây dựng Efficient Frontier.
Mỗi điểm trên frontier là kết quả của một bài toán MVO.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .mvo import MVOResult, solve_mvo, maximize_sharpe, minimize_variance

logger = logging.getLogger(__name__)


@dataclass
class EfficientFrontier:
    """
    Kết quả quét toàn bộ Efficient Frontier.

    Attributes
    ----------
    points : list[MVOResult]
        Danh sách kết quả MVO tại mỗi mức return.
    returns : np.ndarray
        Return tại mỗi điểm.
    volatilities : np.ndarray
        Volatility tại mỗi điểm.
    sharpe_ratios : np.ndarray
        Sharpe ratio tại mỗi điểm.
    weights_df : pd.DataFrame
        Ma trận trọng số (n_points × n_tickers).
    min_variance_result : MVOResult
        Danh mục Global Minimum Variance.
    max_sharpe_result : MVOResult
        Danh mục Maximum Sharpe.
    """

    points: list
    returns: np.ndarray
    volatilities: np.ndarray
    sharpe_ratios: np.ndarray
    weights_df: pd.DataFrame
    min_variance_result: MVOResult
    max_sharpe_result: MVOResult

    @property
    def n_points(self) -> int:
        return len(self.points)

    def summary(self) -> str:
        lines = [
            "=" * 52,
            "EFFICIENT FRONTIER",
            "=" * 52,
            f"Số điểm        : {self.n_points}",
            f"Return range   : [{self.returns.min():.2%}, {self.returns.max():.2%}]",
            f"Volatility range: [{self.volatilities.min():.2%}, {self.volatilities.max():.2%}]",
            "",
            f"Global Min Variance:",
            f"  Return={self.min_variance_result.expected_return:.2%}",
            f"  Vol={self.min_variance_result.volatility:.2%}",
            f"  Sharpe={self.min_variance_result.sharpe_ratio:.4f}",
            "",
            f"Maximum Sharpe Portfolio:",
            f"  Return={self.max_sharpe_result.expected_return:.2%}",
            f"  Vol={self.max_sharpe_result.volatility:.2%}",
            f"  Sharpe={self.max_sharpe_result.sharpe_ratio:.4f}",
        ]
        return "\n".join(lines)


def compute_efficient_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    n_points: int = 50,
    risk_free_rate: float = 0.05,
) -> EfficientFrontier:
    """
    Xây dựng Efficient Frontier bằng cách quét nhiều mức return.

    Parameters
    ----------
    mu : pd.Series
        Expected return hàng năm.
    cov : pd.DataFrame
        Covariance matrix hàng năm.
    n_points : int
        Số điểm trên frontier (mặc định 50).
    risk_free_rate : float
        Lãi suất phi rủi ro để tính Sharpe.

    Returns
    -------
    EfficientFrontier
        Đối tượng chứa toàn bộ frontier và các điểm đặc biệt.
    """
    logger.info(f"Đang quét Efficient Frontier với {n_points} điểm...")

    # Xác định khoảng return hợp lệ
    gmv = minimize_variance(mu, cov, risk_free_rate=risk_free_rate)
    return_min = gmv.expected_return
    return_max = float(mu.max()) * 0.99  # tránh infeasible ở biên

    if return_min >= return_max:
        return_min = float(mu.min())
        return_max = float(mu.max())

    target_returns = np.linspace(return_min, return_max, n_points)

    points: list[MVOResult] = []
    for i, target in enumerate(target_returns):
        result = solve_mvo(mu, cov, target_return=target, risk_free_rate=risk_free_rate)
        if result.status in ("optimal", "optimal_inaccurate"):
            points.append(result)
        else:
            logger.debug(f"  Điểm {i}: target={target:.2%} → {result.status}, bỏ qua.")

    if not points:
        raise RuntimeError("Không tính được bất kỳ điểm nào trên efficient frontier.")

    logger.info(f"  {len(points)}/{n_points} điểm hợp lệ.")

    # Maximum Sharpe
    max_sharpe = maximize_sharpe(mu, cov, risk_free_rate=risk_free_rate)

    # Tổng hợp mảng
    returns_arr = np.array([p.expected_return for p in points])
    vols_arr = np.array([p.volatility for p in points])
    sharpes_arr = np.array([p.sharpe_ratio for p in points])
    tickers = list(mu.index)
    weights_df = pd.DataFrame(
        [p.weights.values for p in points],
        columns=tickers,
    )

    return EfficientFrontier(
        points=points,
        returns=returns_arr,
        volatilities=vols_arr,
        sharpe_ratios=sharpes_arr,
        weights_df=weights_df,
        min_variance_result=gmv,
        max_sharpe_result=max_sharpe,
    )