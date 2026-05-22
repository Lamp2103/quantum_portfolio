"""
src/quantum/sampler.py
----------------------
Đọc kết quả bitstring từ quantum circuit và decode thành portfolio weights.

Sau khi QAOA tìm được bitstring tối ưu x* ∈ {0,1}^n:
  - x_i = 1: chọn tài sản i vào danh mục
  - x_i = 0: không chọn

Từ bitstring → portfolio weights theo nhiều chiến lược:
  1. Equal weight: mỗi tài sản được chọn nhận trọng số 1/B
  2. Risk-weighted: trọng số tỷ lệ nghịch với volatility
  3. Return-weighted: trọng số tỷ lệ thuận với expected return
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SamplingResult:
    """
    Kết quả sampling từ quantum circuit.

    Attributes
    ----------
    counts : dict[str, int]
        Phân phối đo được {bitstring: count}.
    top_bitstrings : pd.DataFrame
        Top-k bitstring tốt nhất (theo QUBO objective).
    best_bitstring : np.ndarray
        Bitstring tối ưu.
    best_objective : float
        Giá trị QUBO tại bitstring tốt nhất.
    probability_distribution : pd.Series
        Xác suất từng bitstring P(x) = count/shots.
    total_shots : int
        Tổng số lần đo.
    n_unique_states : int
        Số trạng thái duy nhất quan sát được.
    """

    counts: dict
    top_bitstrings: pd.DataFrame
    best_bitstring: np.ndarray
    best_objective: float
    probability_distribution: pd.Series
    total_shots: int
    n_unique_states: int


def parse_counts(
    counts: dict,
    Q: np.ndarray,
    budget: int,
    top_k: int = 10,
) -> SamplingResult:
    """
    Phân tích kết quả đo từ quantum circuit.

    Parameters
    ----------
    counts : dict
        {bitstring: count} từ Qiskit result.get_counts().
    Q : np.ndarray
        Ma trận QUBO để tính objective value.
    budget : int
        Số tài sản cần chọn.
    top_k : int
        Số bitstring tốt nhất cần giữ lại.

    Returns
    -------
    SamplingResult
    """
    total_shots = sum(counts.values())
    n_unique = len(counts)

    rows = []
    for bitstring, count in counts.items():
        # Qiskit: qubit 0 ở vị trí cuối bitstring → reverse
        x = np.array([int(b) for b in reversed(bitstring)], dtype=float)
        obj = float(x @ Q @ x)
        feasible = int(x.sum()) == budget
        prob = count / total_shots
        rows.append({
            "bitstring": bitstring,
            "x": x,
            "objective": obj,
            "count": count,
            "probability": prob,
            "feasible": feasible,
            "n_selected": int(x.sum()),
        })

    df = pd.DataFrame(rows).sort_values("objective")

    # Ưu tiên feasible khi lấy top-k
    feasible_df = df[df["feasible"]].head(top_k)
    if len(feasible_df) < top_k:
        infeasible_df = df[~df["feasible"]].head(top_k - len(feasible_df))
        top_df = pd.concat([feasible_df, infeasible_df])
    else:
        top_df = feasible_df

    # Best bitstring
    best_row = top_df.iloc[0]
    best_x = best_row["x"]
    best_obj = best_row["objective"]

    if not best_row["feasible"]:
        logger.warning(
            f"Best bitstring không thỏa budget={budget} "
            f"(có {best_row['n_selected']} tài sản). "
            "Xem xét tăng penalty λ hoặc số shots."
        )

    # Probability distribution
    prob_series = pd.Series(
        {row["bitstring"]: row["probability"] for _, row in df.iterrows()}
    )

    top_display = top_df[["bitstring", "objective", "count", "probability", "feasible", "n_selected"]].copy()

    logger.info(
        f"Sampling: {total_shots} shots, {n_unique} states, "
        f"best_obj={best_obj:.6f}, feasible={best_row['feasible']}"
    )

    return SamplingResult(
        counts=counts,
        top_bitstrings=top_display,
        best_bitstring=best_x,
        best_objective=best_obj,
        probability_distribution=prob_series,
        total_shots=total_shots,
        n_unique_states=n_unique,
    )


def decode_weights(
    x: np.ndarray,
    tickers: list,
    mu: pd.Series | None = None,
    cov: pd.DataFrame | None = None,
    method: str = "equal",
) -> pd.Series:
    """
    Chuyển bitstring x → trọng số danh mục.

    Parameters
    ----------
    x : np.ndarray
        Bitstring nhị phân {0,1}^n.
    tickers : list[str]
        Danh sách mã cổ phiếu.
    mu : pd.Series, optional
        Expected return (dùng cho method='return_weighted').
    cov : pd.DataFrame, optional
        Covariance matrix (dùng cho method='risk_weighted').
    method : str
        Chiến lược phân bổ trọng số:
        - 'equal'          : chia đều cho tài sản được chọn
        - 'return_weighted': tỷ lệ thuận với μ_i (chỉ tài sản dương)
        - 'risk_weighted'  : tỷ lệ nghịch với σ_i (1/vol)

    Returns
    -------
    pd.Series
        Trọng số (ticker → weight), tổng = 1.
    """
    n = len(tickers)
    selected = np.where(x == 1)[0]

    if len(selected) == 0:
        logger.warning("Không có tài sản nào được chọn. Trả về equal weight.")
        return pd.Series(np.ones(n) / n, index=tickers)

    weights = np.zeros(n)

    if method == "equal":
        weights[selected] = 1.0 / len(selected)

    elif method == "return_weighted":
        if mu is None:
            raise ValueError("return_weighted cần mu.")
        raw = np.maximum(mu.values[selected], 0)
        if raw.sum() == 0:
            weights[selected] = 1.0 / len(selected)
        else:
            weights[selected] = raw / raw.sum()

    elif method == "risk_weighted":
        if cov is None:
            raise ValueError("risk_weighted cần cov.")
        vols = np.sqrt(np.diag(cov.values))[selected]
        inv_vol = 1.0 / np.maximum(vols, 1e-8)
        weights[selected] = inv_vol / inv_vol.sum()

    else:
        raise ValueError(f"method không hợp lệ: '{method}'. Chọn: equal, return_weighted, risk_weighted.")

    return pd.Series(weights, index=tickers)


def aggregate_weights_from_distribution(
    sampling_result: SamplingResult,
    tickers: list,
    mu: pd.Series | None = None,
    cov: pd.DataFrame | None = None,
    weight_method: str = "equal",
    top_k: int = 5,
) -> pd.Series:
    """
    Tính trọng số danh mục bằng cách lấy trung bình có trọng số
    từ top-k bitstring tốt nhất (thay vì chỉ dùng 1 bitstring).

    Phương pháp này ổn định hơn khi quantum measurement có nhiễu.

    Parameters
    ----------
    sampling_result : SamplingResult
    tickers : list[str]
    mu : pd.Series, optional
    cov : pd.DataFrame, optional
    weight_method : str
        Cách phân bổ trọng số trong mỗi bitstring.
    top_k : int
        Số bitstring dùng để tổng hợp.

    Returns
    -------
    pd.Series
        Trọng số tổng hợp.
    """
    top = sampling_result.top_bitstrings.head(top_k)

    weighted_sum = np.zeros(len(tickers))
    total_prob = 0.0

    for _, row in top.iterrows():
        x = np.array([int(b) for b in reversed(row["bitstring"])], dtype=float)
        prob = row["probability"]
        w = decode_weights(x, tickers, mu=mu, cov=cov, method=weight_method)
        weighted_sum += prob * w.values
        total_prob += prob

    if total_prob > 0:
        final_weights = weighted_sum / total_prob
    else:
        final_weights = np.ones(len(tickers)) / len(tickers)

    # Re-normalize
    if final_weights.sum() > 0:
        final_weights /= final_weights.sum()

    return pd.Series(final_weights, index=tickers)


def print_sampling_summary(result: SamplingResult, qubo) -> None:
    """In tóm tắt kết quả sampling ra console."""
    print("\n" + "=" * 52)
    print("KẾT QUẢ QUANTUM SAMPLING")
    print("=" * 52)
    print(f"Tổng shots     : {result.total_shots:,}")
    print(f"Unique states  : {result.n_unique_states}")
    print(f"Best objective : {result.best_objective:.6f}")
    print(f"Feasible       : {'✓' if int(result.best_bitstring.sum()) == qubo.budget else '✗'}")
    print(f"Selected assets: {int(result.best_bitstring.sum())}/{qubo.budget} cần")
    print()
    print(f"Top bitstrings (objective nhỏ nhất):")
    print(result.top_bitstrings.head(8).to_string(index=False))