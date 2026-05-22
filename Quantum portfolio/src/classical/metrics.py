"""
src/classical/metrics.py
------------------------
Tính toán các chỉ số hiệu suất danh mục:
  - Sharpe ratio
  - Sortino ratio
  - Maximum drawdown
  - Value at Risk (VaR) và Conditional VaR (CVaR)
  - Calmar ratio
  - Annualized return / volatility
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def portfolio_returns(
    weights: pd.Series,
    returns: pd.DataFrame,
) -> pd.Series:
    """
    Tính chuỗi return hàng ngày của danh mục.

    Parameters
    ----------
    weights : pd.Series
        Trọng số (ticker → weight), tổng = 1.
    returns : pd.DataFrame
        Returns hàng ngày (ngày × ticker).

    Returns
    -------
    pd.Series
        Return hàng ngày của danh mục.
    """
    common = weights.index.intersection(returns.columns)
    w = weights[common]
    w = w / w.sum()  # re-normalize nếu thiếu cột
    return returns[common] @ w


def annualized_return(port_returns: pd.Series, factor: int = 252) -> float:
    """Return hàng năm: mean(r) × factor."""
    return float(port_returns.mean() * factor)


def annualized_volatility(port_returns: pd.Series, factor: int = 252) -> float:
    """Volatility hàng năm: std(r) × sqrt(factor)."""
    return float(port_returns.std() * np.sqrt(factor))


def sharpe_ratio(
    port_returns: pd.Series,
    risk_free_rate: float = 0.05,
    factor: int = 252,
) -> float:
    """
    Sharpe ratio = (R_p - R_f) / σ_p (annualized).

    Parameters
    ----------
    port_returns : pd.Series
        Return hàng ngày của danh mục.
    risk_free_rate : float
        Lãi suất phi rủi ro hàng năm.
    factor : int
        Hệ số annualize (252 cho ngày giao dịch).
    """
    ann_return = annualized_return(port_returns, factor)
    ann_vol = annualized_volatility(port_returns, factor)
    if ann_vol == 0:
        return 0.0
    return (ann_return - risk_free_rate) / ann_vol


def sortino_ratio(
    port_returns: pd.Series,
    risk_free_rate: float = 0.05,
    factor: int = 252,
) -> float:
    """
    Sortino ratio = (R_p - R_f) / downside_deviation.
    Chỉ phạt phần return âm, không phạt return dương.
    """
    ann_return = annualized_return(port_returns, factor)
    downside = port_returns[port_returns < 0]
    if len(downside) == 0:
        return float("inf")
    downside_std = float(downside.std() * np.sqrt(factor))
    if downside_std == 0:
        return 0.0
    return (ann_return - risk_free_rate) / downside_std


def maximum_drawdown(port_returns: pd.Series) -> float:
    """
    Maximum Drawdown (MDD) = max peak-to-trough decline.

    Returns
    -------
    float
        MDD dạng tỷ lệ (âm), ví dụ -0.25 = -25%.
    """
    cumulative = (1 + port_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(drawdown.min())


def value_at_risk(
    port_returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """
    Value at Risk (VaR) theo phương pháp Historical Simulation.

    Parameters
    ----------
    port_returns : pd.Series
        Return hàng ngày.
    confidence : float
        Mức độ tin cậy (mặc định 95%).

    Returns
    -------
    float
        VaR hàng ngày (số âm), ví dụ -0.02 = tổn thất tối đa 2%/ngày
        ở mức 95% confidence.
    """
    return float(np.percentile(port_returns, (1 - confidence) * 100))


def conditional_var(
    port_returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """
    Conditional VaR (CVaR / Expected Shortfall).
    Trung bình return khi return < VaR — ước lượng tổn thất
    trong kịch bản xấu nhất.
    """
    var = value_at_risk(port_returns, confidence)
    tail = port_returns[port_returns <= var]
    if len(tail) == 0:
        return var
    return float(tail.mean())


def calmar_ratio(
    port_returns: pd.Series,
    risk_free_rate: float = 0.05,
    factor: int = 252,
) -> float:
    """
    Calmar ratio = annualized_return / |max_drawdown|.
    Đo lường return trên mỗi đơn vị rủi ro drawdown.
    """
    ann_return = annualized_return(port_returns, factor)
    mdd = maximum_drawdown(port_returns)
    if mdd == 0:
        return 0.0
    return (ann_return - risk_free_rate) / abs(mdd)


def full_metrics(
    weights: pd.Series,
    returns: pd.DataFrame,
    risk_free_rate: float = 0.05,
    factor: int = 252,
    label: str = "Portfolio",
) -> pd.Series:
    """
    Tính toàn bộ chỉ số hiệu suất cho một danh mục.

    Parameters
    ----------
    weights : pd.Series
        Trọng số danh mục.
    returns : pd.DataFrame
        Returns hàng ngày (ngày × ticker).
    risk_free_rate : float
        Lãi suất phi rủi ro.
    factor : int
        Hệ số annualize.
    label : str
        Tên danh mục (dùng làm name của Series).

    Returns
    -------
    pd.Series
        Tất cả chỉ số, dễ dàng so sánh bằng pd.DataFrame.
    """
    pr = portfolio_returns(weights, returns)

    metrics = {
        "Annualized Return": annualized_return(pr, factor),
        "Annualized Volatility": annualized_volatility(pr, factor),
        "Sharpe Ratio": sharpe_ratio(pr, risk_free_rate, factor),
        "Sortino Ratio": sortino_ratio(pr, risk_free_rate, factor),
        "Max Drawdown": maximum_drawdown(pr),
        "VaR (95%)": value_at_risk(pr, 0.95),
        "CVaR (95%)": conditional_var(pr, 0.95),
        "Calmar Ratio": calmar_ratio(pr, risk_free_rate, factor),
    }

    return pd.Series(metrics, name=label)


def compare_portfolios(
    portfolio_dict: dict,
    returns: pd.DataFrame,
    risk_free_rate: float = 0.05,
) -> pd.DataFrame:
    """
    So sánh nhiều danh mục cùng lúc.

    Parameters
    ----------
    portfolio_dict : dict[str, pd.Series]
        Dict: tên danh mục → weights Series.
    returns : pd.DataFrame
        Returns hàng ngày.
    risk_free_rate : float
        Lãi suất phi rủi ro.

    Returns
    -------
    pd.DataFrame
        Bảng so sánh: rows = metrics, columns = danh mục.

    Example
    -------
    >>> compare_portfolios(
    ...     {"MVO": mvo_weights, "Equal Weight": eq_weights},
    ...     returns_df
    ... )
    """
    series_list = [
        full_metrics(w, returns, risk_free_rate=risk_free_rate, label=name)
        for name, w in portfolio_dict.items()
    ]
    df = pd.DataFrame(series_list).T
    return df