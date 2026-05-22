"""
src/classical/mvo.py
--------------------
Markowitz Mean-Variance Optimization (MVO) dùng cvxpy.

Giải bài toán:
    minimize   w^T Σ w           (portfolio variance)
    subject to w^T μ >= target   (return constraint)
               sum(w) == 1       (fully invested)
               w >= 0            (long-only)

Đây là baseline để so sánh với Quantum QAOA solver.
"""

import logging
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MVOResult:
    """
    Kết quả tối ưu hóa Markowitz MVO.

    Attributes
    ----------
    weights : pd.Series
        Trọng số danh mục (ticker → weight), tổng = 1.
    expected_return : float
        Return kỳ vọng hàng năm của danh mục.
    volatility : float
        Độ lệch chuẩn hàng năm (rủi ro) của danh mục.
    sharpe_ratio : float
        Sharpe ratio = (return - rf) / volatility.
    status : str
        Trạng thái solver: 'optimal', 'infeasible', v.v.
    risk_free_rate : float
        Lãi suất phi rủi ro dùng để tính Sharpe.
    """

    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float
    status: str
    risk_free_rate: float

    def __str__(self) -> str:
        lines = [
            "=" * 48,
            "KẾT QUẢ MARKOWITZ MVO",
            "=" * 48,
            f"Trạng thái     : {self.status}",
            f"Expected Return: {self.expected_return:.2%}",
            f"Volatility     : {self.volatility:.2%}",
            f"Sharpe Ratio   : {self.sharpe_ratio:.4f}",
            "",
            "Phân bổ danh mục:",
        ]
        for ticker, w in self.weights[self.weights > 1e-4].sort_values(ascending=False).items():
            lines.append(f"  {ticker:<8} {w:.2%}  {'█' * int(w * 40)}")
        return "\n".join(lines)


def solve_mvo(
    mu: pd.Series,
    cov: pd.DataFrame,
    target_return: float | None = None,
    risk_free_rate: float = 0.05,
    allow_short: bool = False,
    max_weight: float = 1.0,
    min_weight: float = 0.0,
) -> MVOResult:
    """
    Giải bài toán MVO: minimize variance với ràng buộc return tối thiểu.

    Parameters
    ----------
    mu : pd.Series
        Expected return hàng năm (n,).
    cov : pd.DataFrame
        Covariance matrix hàng năm (n × n).
    target_return : float, optional
        Return mục tiêu tối thiểu. Nếu None, maximize Sharpe ratio.
    risk_free_rate : float
        Lãi suất phi rủi ro để tính Sharpe (mặc định 5%).
    allow_short : bool
        Cho phép bán khống (w < 0) nếu True.
    max_weight : float
        Trọng số tối đa mỗi tài sản (mặc định 100%).
    min_weight : float
        Trọng số tối thiểu mỗi tài sản (mặc định 0%).

    Returns
    -------
    MVOResult
        Kết quả tối ưu hóa.
    """
    tickers = list(mu.index)
    n = len(tickers)
    mu_arr = mu.values
    cov_arr = cov.values

    w = cp.Variable(n)

    # Objective: minimize portfolio variance
    portfolio_variance = cp.quad_form(w, cov_arr)
    objective = cp.Minimize(portfolio_variance)

    # Constraints
    constraints = [cp.sum(w) == 1]

    if not allow_short:
        constraints.append(w >= min_weight)

    constraints.append(w <= max_weight)

    if target_return is not None:
        constraints.append(mu_arr @ w >= target_return)

    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.CLARABEL, warm_start=True)
    except Exception:
        try:
            problem.solve(solver=cp.SCS)
        except Exception as e:
            logger.error(f"Solver thất bại: {e}")
            return MVOResult(
                weights=pd.Series(np.ones(n) / n, index=tickers),
                expected_return=float(mu_arr.mean()),
                volatility=float(np.sqrt(np.diag(cov_arr).mean())),
                sharpe_ratio=0.0,
                status="failed",
                risk_free_rate=risk_free_rate,
            )

    status = problem.status

    if status not in ("optimal", "optimal_inaccurate"):
        logger.warning(f"MVO solver status: {status}")
        # Fallback: equal weight
        w_opt = np.ones(n) / n
    else:
        w_opt = np.array(w.value).flatten()
        # Clip nhỏ để tránh floating point âm
        w_opt = np.clip(w_opt, 0, None)
        w_opt /= w_opt.sum()

    weights = pd.Series(w_opt, index=tickers)
    port_return = float(mu_arr @ w_opt)
    port_vol = float(np.sqrt(w_opt @ cov_arr @ w_opt))
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0.0

    return MVOResult(
        weights=weights,
        expected_return=port_return,
        volatility=port_vol,
        sharpe_ratio=sharpe,
        status=status,
        risk_free_rate=risk_free_rate,
    )


def maximize_sharpe(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float = 0.05,
    allow_short: bool = False,
) -> MVOResult:
    """
    Tìm danh mục có Sharpe ratio cao nhất (Maximum Sharpe Portfolio).

    Dùng kỹ thuật biến đổi biến y = w / (μ^T w - rf) để linearize.

    Parameters
    ----------
    mu : pd.Series
        Expected return hàng năm.
    cov : pd.DataFrame
        Covariance matrix hàng năm.
    risk_free_rate : float
        Lãi suất phi rủi ro.
    allow_short : bool
        Cho phép bán khống.

    Returns
    -------
    MVOResult
        Danh mục Maximum Sharpe.
    """
    tickers = list(mu.index)
    n = len(tickers)
    mu_arr = mu.values - risk_free_rate  # excess return
    cov_arr = cov.values

    # Biến đổi: y = w / k, k = (μ-rf)^T w > 0
    y = cp.Variable(n)
    kappa = cp.Variable(pos=True)

    objective = cp.Minimize(cp.quad_form(y, cov_arr))
    constraints = [
        mu_arr @ y == 1,
        cp.sum(y) == kappa,
    ]
    if not allow_short:
        constraints.append(y >= 0)

    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.CLARABEL)
    except Exception:
        problem.solve(solver=cp.SCS)

    if problem.status not in ("optimal", "optimal_inaccurate") or kappa.value is None:
        logger.warning(f"Max Sharpe solver status: {problem.status}. Fallback về min variance.")
        return solve_mvo(mu, cov, risk_free_rate=risk_free_rate)

    w_opt = np.array(y.value).flatten() / float(kappa.value)
    w_opt = np.clip(w_opt, 0, None)
    w_opt /= w_opt.sum()

    weights = pd.Series(w_opt, index=tickers)
    port_return = float(mu.values @ w_opt)
    port_vol = float(np.sqrt(w_opt @ cov_arr @ w_opt))
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0.0

    return MVOResult(
        weights=weights,
        expected_return=port_return,
        volatility=port_vol,
        sharpe_ratio=sharpe,
        status=problem.status,
        risk_free_rate=risk_free_rate,
    )


def minimize_variance(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float = 0.05,
) -> MVOResult:
    """
    Danh mục Global Minimum Variance (GMV) — không có ràng buộc return.

    Parameters
    ----------
    mu : pd.Series
        Expected return (chỉ dùng để tính Sharpe sau).
    cov : pd.DataFrame
        Covariance matrix.
    risk_free_rate : float
        Lãi suất phi rủi ro.

    Returns
    -------
    MVOResult
    """
    return solve_mvo(mu, cov, target_return=None, risk_free_rate=risk_free_rate)