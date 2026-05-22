"""
src/quantum/optimizer.py
------------------------
Vòng lặp tối ưu hóa hybrid classical-quantum cho QAOA.

Luồng:
  1. Khởi tạo tham số θ = [γ₁, β₁, ..., γₚ, βₚ]
  2. Gửi mạch QAOA(θ) lên backend (simulator hoặc IBM hardware)
  3. Đo kết quả → phân phối bitstring
  4. Tính expectation value ⟨H_C⟩ từ phân phối
  5. Classical optimizer (COBYLA/SPSA) cập nhật θ
  6. Lặp lại đến khi hội tụ
  7. Lấy bitstring tốt nhất → decode thành portfolio weights

Classical optimizers hỗ trợ:
  - COBYLA: gradient-free, ổn định, phù hợp noisy quantum
  - SPSA:   stochastic, hiệu quả với nhiều tham số
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from qiskit_algorithms.optimizers import COBYLA, SPSA
    from qiskit_algorithms import SamplingVQE, QAOA as QiskitQAOA
    from qiskit.primitives import StatevectorSampler
    QISKIT_ALGORITHMS_AVAILABLE = True
except ImportError:
    QISKIT_ALGORITHMS_AVAILABLE = False


@dataclass
class OptimizationResult:
    """
    Kết quả từ vòng lặp tối ưu hóa QAOA.

    Attributes
    ----------
    best_bitstring : np.ndarray
        Bitstring tốt nhất tìm được {0,1}^n.
    best_objective : float
        Giá trị hàm mục tiêu QUBO tại bitstring tốt nhất.
    optimal_params : np.ndarray
        Tham số θ* tối ưu [γ*, β*].
    weights : pd.Series
        Trọng số danh mục decode từ bitstring.
    n_iterations : int
        Số vòng lặp đã thực hiện.
    convergence_history : list[float]
        Lịch sử giá trị hàm mục tiêu qua các vòng lặp.
    feasible : bool
        True nếu bitstring thỏa ràng buộc budget.
    elapsed_seconds : float
        Thời gian chạy.
    backend_name : str
        Tên backend đã dùng.
    shots : int
        Số lần đo mỗi circuit.
    """

    best_bitstring: np.ndarray
    best_objective: float
    optimal_params: np.ndarray
    weights: pd.Series
    n_iterations: int
    convergence_history: list = field(default_factory=list)
    feasible: bool = False
    elapsed_seconds: float = 0.0
    backend_name: str = "unknown"
    shots: int = 1024

    def __str__(self) -> str:
        lines = [
            "=" * 52,
            "QAOA OPTIMIZATION RESULT",
            "=" * 52,
            f"Backend        : {self.backend_name}",
            f"Shots          : {self.shots}",
            f"Iterations     : {self.n_iterations}",
            f"Elapsed        : {self.elapsed_seconds:.1f}s",
            f"Best objective : {self.best_objective:.6f}",
            f"Feasible       : {'✓' if self.feasible else '✗'}",
            f"Bitstring      : {''.join(map(str, self.best_bitstring.astype(int)))}",
            "",
            "Portfolio weights:",
        ]
        for ticker, w in self.weights[self.weights > 1e-6].sort_values(ascending=False).items():
            lines.append(f"  {ticker:<8} {w:.2%}  {'█' * int(w * 30)}")
        return "\n".join(lines)


def counts_to_expectation(
    counts: dict,
    Q: np.ndarray,
    shots: int,
) -> float:
    """
    Tính expectation value ⟨H_C⟩ từ kết quả đo (counts dict).

    Parameters
    ----------
    counts : dict
        {bitstring: count}, ví dụ {'0110': 42, '1001': 58}.
    Q : np.ndarray
        Ma trận QUBO.
    shots : int
        Tổng số lần đo.

    Returns
    -------
    float
        ⟨H_C⟩ = Σ_x P(x) * x^T Q x
    """
    expectation = 0.0
    for bitstring, count in counts.items():
        # Qiskit trả về bitstring theo thứ tự ngược (qubit 0 ở cuối)
        x = np.array([int(b) for b in reversed(bitstring)], dtype=float)
        val = x @ Q @ x
        expectation += (count / shots) * val
    return expectation


def sample_best_bitstring(
    counts: dict,
    Q: np.ndarray,
    budget: int,
) -> tuple[np.ndarray, float]:
    """
    Chọn bitstring tốt nhất từ kết quả đo.

    Ưu tiên: (1) thỏa budget, (2) objective nhỏ nhất.

    Parameters
    ----------
    counts : dict
        {bitstring: count}.
    Q : np.ndarray
        Ma trận QUBO.
    budget : int
        Số tài sản cần chọn.

    Returns
    -------
    tuple[np.ndarray, float]
        (best_x, best_objective_value)
    """
    best_x = None
    best_val = float("inf")
    fallback_x = None
    fallback_val = float("inf")

    for bitstring in counts:
        x = np.array([int(b) for b in reversed(bitstring)], dtype=float)
        val = x @ Q @ x

        # Ưu tiên feasible
        if int(x.sum()) == budget:
            if val < best_val:
                best_val = val
                best_x = x.copy()
        else:
            if val < fallback_val:
                fallback_val = val
                fallback_x = x.copy()

    if best_x is not None:
        return best_x, best_val
    # Nếu không có feasible solution, trả về fallback
    logger.warning("Không tìm được bitstring thỏa budget. Dùng fallback.")
    return fallback_x, fallback_val


class COBYLAOptimizer:
    """
    COBYLA optimizer thuần Python (không cần qiskit_algorithms).
    Dùng scipy.optimize.minimize với method='COBYLA'.
    """

    def __init__(self, maxiter: int = 200, rhobeg: float = 0.5):
        self.maxiter = maxiter
        self.rhobeg = rhobeg

    def minimize(
        self,
        fun: Callable,
        x0: np.ndarray,
    ) -> tuple[np.ndarray, float, int]:
        """
        Parameters
        ----------
        fun : Callable
            Hàm mục tiêu f(θ) → float.
        x0 : np.ndarray
            Điểm khởi đầu.

        Returns
        -------
        tuple[np.ndarray, float, int]
            (x_optimal, f_optimal, n_evaluations)
        """
        from scipy.optimize import minimize

        n_evals = [0]
        history = []

        def tracked_fun(x):
            val = fun(x)
            n_evals[0] += 1
            history.append(val)
            return val

        result = minimize(
            tracked_fun,
            x0,
            method="COBYLA",
            options={"maxiter": self.maxiter, "rhobeg": self.rhobeg},
        )
        return result.x, result.fun, n_evals[0], history


class SPSAOptimizer:
    """
    SPSA (Simultaneous Perturbation Stochastic Approximation) optimizer.
    Hiệu quả khi hàm mục tiêu nhiễu (noisy quantum hardware).
    """

    def __init__(
        self,
        maxiter: int = 200,
        learning_rate: float = 0.1,
        perturbation: float = 0.1,
    ):
        self.maxiter = maxiter
        self.a = learning_rate
        self.c = perturbation

    def minimize(
        self,
        fun: Callable,
        x0: np.ndarray,
    ) -> tuple[np.ndarray, float, int, list]:
        x = x0.copy()
        n = len(x)
        n_evals = 0
        history = []

        for k in range(1, self.maxiter + 1):
            # Learning rate và perturbation giảm dần
            ak = self.a / (k + 1) ** 0.602
            ck = self.c / (k + 1) ** 0.101

            # Random perturbation vector (Bernoulli ±1)
            delta = np.where(np.random.random(n) > 0.5, 1, -1).astype(float)

            f_plus = fun(x + ck * delta)
            f_minus = fun(x - ck * delta)
            n_evals += 2

            # Gradient estimate
            grad = (f_plus - f_minus) / (2 * ck * delta)
            x -= ak * grad

            current_val = fun(x)
            n_evals += 1
            history.append(current_val)

        final_val = fun(x)
        return x, final_val, n_evals, history


def optimize_qaoa(
    qubo,
    backend,
    depth: int = 2,
    optimizer_name: str = "COBYLA",
    max_iterations: int = 200,
    shots: int = 1024,
    seed: int = 42,
) -> OptimizationResult:
    """
    Chạy vòng lặp tối ưu hóa QAOA hybrid.

    Parameters
    ----------
    qubo : QUBOProblem
        Bài toán QUBO đã được formulate.
    backend : BaseBackend
        Backend Qiskit (AerSimulator hoặc IBM Quantum).
    depth : int
        Số layer QAOA.
    optimizer_name : str
        'COBYLA' hoặc 'SPSA'.
    max_iterations : int
        Số vòng lặp tối đa.
    shots : int
        Số lần đo mỗi circuit.
    seed : int
        Random seed.

    Returns
    -------
    OptimizationResult
    """
    from .circuit import build_qaoa_circuit, get_circuit_config, bind_parameters

    start_time = time.time()
    Q = qubo.Q
    n = qubo.n_assets

    # Cấu hình và khởi tạo tham số
    config = get_circuit_config(n, depth=depth, seed=seed)
    x0 = config.initial_params

    # Build mạch parameterized
    qc_template = build_qaoa_circuit(Q, depth=depth)

    convergence_history = []

    def objective(params: np.ndarray) -> float:
        """Hàm mục tiêu: ⟨H_C⟩ với tham số params."""
        # Bind tham số vào mạch
        qc_bound = bind_parameters(qc_template, params, depth)

        # Chạy trên backend
        from qiskit import transpile
        qc_t = transpile(qc_bound, backend)
        job = backend.run(qc_t, shots=shots)
        result = job.result()
        counts = result.get_counts()

        # Tính expectation
        exp_val = counts_to_expectation(counts, Q, shots)
        convergence_history.append(exp_val)
        return exp_val

    # Chọn optimizer
    logger.info(f"Bắt đầu tối ưu hóa QAOA: depth={depth}, optimizer={optimizer_name}, maxiter={max_iterations}")

    if optimizer_name.upper() == "SPSA":
        opt = SPSAOptimizer(maxiter=max_iterations)
    else:
        opt = COBYLAOptimizer(maxiter=max_iterations)

    optimal_params, _, n_evals, history = opt.minimize(objective, x0)
    convergence_history = history

    # Lấy phân phối cuối cùng với tham số tối ưu
    from .circuit import bind_parameters
    from qiskit import transpile

    qc_final = bind_parameters(qc_template, optimal_params, depth)
    qc_t = transpile(qc_final, backend)
    job = backend.run(qc_t, shots=shots * 4)  # nhiều shot hơn ở bước cuối
    final_counts = job.result().get_counts()

    # Chọn bitstring tốt nhất
    best_x, best_val = sample_best_bitstring(final_counts, Q, qubo.budget)

    elapsed = time.time() - start_time
    weights = qubo.decode_weights(best_x)
    feasible = qubo.is_feasible(best_x)

    backend_name = getattr(backend, "name", str(backend))
    if callable(backend_name):
        backend_name = backend_name()

    logger.info(
        f"QAOA hoàn thành: {elapsed:.1f}s, {n_evals} evaluations, "
        f"best_obj={best_val:.6f}, feasible={feasible}"
    )

    return OptimizationResult(
        best_bitstring=best_x,
        best_objective=best_val,
        optimal_params=optimal_params,
        weights=weights,
        n_iterations=n_evals,
        convergence_history=convergence_history,
        feasible=feasible,
        elapsed_seconds=elapsed,
        backend_name=str(backend_name),
        shots=shots,
    )


def optimize_qaoa_statevector(
    qubo,
    depth: int = 2,
    optimizer_name: str = "COBYLA",
    max_iterations: int = 200,
    seed: int = 42,
) -> OptimizationResult:
    """
    Chạy QAOA với StatevectorSimulator (không có shot noise).
    Dùng để debug và verify circuit trước khi chạy hardware thật.

    Parameters
    ----------
    qubo : QUBOProblem
    depth : int
    optimizer_name : str
    max_iterations : int
    seed : int

    Returns
    -------
    OptimizationResult
    """
    from qiskit_aer import AerSimulator
    backend = AerSimulator(method="statevector")
    return optimize_qaoa(
        qubo=qubo,
        backend=backend,
        depth=depth,
        optimizer_name=optimizer_name,
        max_iterations=max_iterations,
        shots=2048,
        seed=seed,
    )