"""
src/quantum/qubo.py
-------------------
Chuyển bài toán tối ưu hóa danh mục sang dạng QUBO
(Quadratic Unconstrained Binary Optimization).

Bài toán MVO liên tục:
    minimize    w^T Σ w  -  q * μ^T w
    subject to  sum(w) = 1,  w >= 0

QUBO hóa bằng cách:
  1. Discretize: mỗi tài sản i được chọn hay không → biến nhị phân x_i ∈ {0,1}
  2. Encode ràng buộc sum(x) = B (budget = số cổ phiếu chọn) vào hàm phạt
  3. Ma trận QUBO Q: bài toán trở thành minimize x^T Q x

Công thức:
    H = x^T Σ x  -  q * μ^T x  +  λ * (sum(x) - B)^2

Khai triển (sum(x) - B)^2 = sum_i x_i^2 + 2*sum_{i<j} x_i x_j - 2B*sum_i x_i + B^2
Vì x_i ∈ {0,1}: x_i^2 = x_i → hấp thụ vào diagonal

QUBO matrix Q:
    Q_ii = Σ_ii - q*μ_i + λ*(1 - 2B)
    Q_ij = 2*Σ_ij + 2*λ      (i < j)
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class QUBOProblem:
    """
    Container chứa bài toán QUBO đã được formulate.

    Attributes
    ----------
    Q : np.ndarray
        Ma trận QUBO (n × n). Bài toán: minimize x^T Q x.
    tickers : list[str]
        Danh sách tài sản tương ứng với mỗi qubit.
    n_assets : int
        Số tài sản = số qubit cần thiết.
    budget : int
        Số cổ phiếu được chọn (B).
    risk_aversion : float
        Hệ số q: đánh đổi return vs risk.
    penalty : float
        Hệ số λ: phạt vi phạm ràng buộc budget.
    mu : pd.Series
        Expected return đã dùng để build QUBO.
    cov : pd.DataFrame
        Covariance matrix đã dùng để build QUBO.
    """

    Q: np.ndarray
    tickers: list
    n_assets: int
    budget: int
    risk_aversion: float
    penalty: float
    mu: pd.Series
    cov: pd.DataFrame

    def evaluate(self, x: np.ndarray) -> float:
        """
        Tính giá trị hàm mục tiêu QUBO cho vector nhị phân x.

        Parameters
        ----------
        x : np.ndarray
            Vector nhị phân {0,1}^n.

        Returns
        -------
        float
            Giá trị x^T Q x.
        """
        return float(x @ self.Q @ x)

    def decode_weights(self, x: np.ndarray) -> pd.Series:
        """
        Chuyển bitstring x → trọng số danh mục (equal weight trong nhóm được chọn).

        Parameters
        ----------
        x : np.ndarray
            Vector nhị phân {0,1}^n, x_i=1 nghĩa là chọn tài sản i.

        Returns
        -------
        pd.Series
            Trọng số (ticker → weight), tổng = 1.
            Các tài sản được chọn chia đều trọng số.
        """
        selected = np.where(x == 1)[0]
        weights = np.zeros(self.n_assets)
        if len(selected) > 0:
            weights[selected] = 1.0 / len(selected)
        return pd.Series(weights, index=self.tickers)

    def is_feasible(self, x: np.ndarray) -> bool:
        """Kiểm tra x có thỏa ràng buộc budget không."""
        return int(x.sum()) == self.budget

    def penalty_violation(self, x: np.ndarray) -> float:
        """Mức độ vi phạm ràng buộc budget."""
        return float((x.sum() - self.budget) ** 2)

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "QUBO FORMULATION",
            "=" * 50,
            f"Số tài sản (qubit)  : {self.n_assets}",
            f"Budget (B)          : {self.budget}",
            f"Risk aversion (q)   : {self.risk_aversion}",
            f"Penalty (λ)         : {self.penalty:.4f}",
            f"Q matrix shape      : {self.Q.shape}",
            f"Q diagonal range    : [{np.diag(self.Q).min():.4f}, {np.diag(self.Q).max():.4f}]",
            f"Q off-diag range    : [{np.triu(self.Q,1).min():.4f}, {np.triu(self.Q,1).max():.4f}]",
        ]
        return "\n".join(lines)


def build_qubo(
    mu: pd.Series,
    cov: pd.DataFrame,
    budget: int,
    risk_aversion: float = 0.5,
    penalty: float | None = None,
) -> QUBOProblem:
    """
    Xây dựng ma trận QUBO cho bài toán chọn danh mục nhị phân.

    Parameters
    ----------
    mu : pd.Series
        Expected return hàng năm (n,).
    cov : pd.DataFrame
        Covariance matrix hàng năm (n × n).
    budget : int
        Số tài sản được chọn (B). Thường = n//2.
    risk_aversion : float
        Hệ số q ∈ [0, 1]:
          q=0 → chỉ minimize risk
          q=1 → chỉ maximize return
    penalty : float, optional
        Hệ số phạt λ cho ràng buộc budget.
        Mặc định = max(|Σ|) * n để đảm bảo ràng buộc được thỏa.

    Returns
    -------
    QUBOProblem
        Đối tượng chứa Q và metadata.

    Notes
    -----
    Để QAOA hoạt động tốt, normalize μ và Σ về cùng scale trước khi build.
    """
    tickers = list(mu.index)
    n = len(tickers)

    if budget < 1 or budget > n:
        raise ValueError(f"budget phải trong [1, {n}], nhận được {budget}.")

    mu_arr = mu.values.copy()
    cov_arr = cov.values.copy()

    # Normalize để tránh penalty quá nhỏ so với objective
    # Scale cov về [0,1] theo max element
    cov_scale = np.abs(cov_arr).max()
    mu_scale = np.abs(mu_arr).max()

    if cov_scale > 0:
        cov_norm = cov_arr / cov_scale
    else:
        cov_norm = cov_arr

    if mu_scale > 0:
        mu_norm = mu_arr / mu_scale
    else:
        mu_norm = mu_arr

    # Penalty mặc định: đủ lớn để ràng buộc luôn được thỏa
    if penalty is None:
        penalty = float(np.abs(cov_norm).max() * n * 2)

    logger.info(
        f"Build QUBO: n={n}, budget={budget}, q={risk_aversion}, "
        f"λ={penalty:.4f}, cov_scale={cov_scale:.4f}"
    )

    # ── Xây dựng Q ────────────────────────────────────────────
    Q = np.zeros((n, n))

    # Phần risk: x^T Σ_norm x
    Q += cov_norm

    # Phần return: -q * μ_norm^T x → diagonal
    # (vì x_i^2 = x_i với x_i ∈ {0,1})
    Q += np.diag(-risk_aversion * mu_norm)

    # Phần penalty: λ * (sum(x) - B)^2
    # = λ * [sum_i x_i^2 + 2*sum_{i<j} x_i*x_j - 2B*sum_i x_i + B^2]
    # x_i^2 = x_i → diagonal += λ*(1 - 2B)
    # cross terms → off-diagonal += 2λ
    Q += np.diag(np.full(n, penalty * (1 - 2 * budget)))
    Q += penalty * 2 * (np.ones((n, n)) - np.eye(n))

    # Chỉ giữ upper triangular (QUBO convention) — nhưng để full matrix
    # để tiện tính x^T Q x (đối xứng hóa)
    Q = (Q + Q.T) / 2  # đảm bảo symmetric

    logger.info(f"Q built: diag=[{np.diag(Q).min():.3f}, {np.diag(Q).max():.3f}]")

    return QUBOProblem(
        Q=Q,
        tickers=tickers,
        n_assets=n,
        budget=budget,
        risk_aversion=risk_aversion,
        penalty=penalty,
        mu=mu,
        cov=cov,
    )


def get_optimal_budget(n_assets: int, max_qubits: int = 10) -> int:
    """
    Chọn budget tự động dựa trên số tài sản và giới hạn qubit.

    Parameters
    ----------
    n_assets : int
        Số tài sản.
    max_qubits : int
        Giới hạn qubit của backend.

    Returns
    -------
    int
        Budget tối ưu = min(n//2, max_qubits//2).
    """
    n_use = min(n_assets, max_qubits)
    return max(1, n_use // 2)


def brute_force_qubo(qubo: QUBOProblem) -> tuple[np.ndarray, float]:
    """
    Giải QUBO bằng brute-force (chỉ dùng để test với n nhỏ ≤ 20).

    Enumerate tất cả 2^n bitstrings, tìm x* có giá trị nhỏ nhất
    trong số những x thỏa ràng buộc budget.

    Parameters
    ----------
    qubo : QUBOProblem

    Returns
    -------
    tuple[np.ndarray, float]
        (x_optimal, objective_value)
    """
    n = qubo.n_assets
    if n > 20:
        raise ValueError(f"Brute-force chỉ dùng cho n ≤ 20, n={n}.")

    best_x = None
    best_val = float("inf")

    for bits in range(2**n):
        x = np.array([(bits >> i) & 1 for i in range(n)], dtype=float)
        if int(x.sum()) != qubo.budget:
            continue
        val = qubo.evaluate(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()

    if best_x is None:
        raise RuntimeError(f"Không tìm được bitstring với budget={qubo.budget}.")

    return best_x, best_val