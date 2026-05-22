"""
src/quantum/circuit.py
----------------------
Xây dựng mạch QAOA (Quantum Approximate Optimization Algorithm)
cho bài toán QUBO danh mục đầu tư.

Cấu trúc mạch QAOA depth p:
  |+⟩^n  →  [Cost layer(γ₁)]  →  [Mixer layer(β₁)]
         →  [Cost layer(γ₂)]  →  [Mixer layer(β₂)]
         →  ...
         →  [Cost layer(γₚ)]  →  [Mixer layer(βₚ)]
         →  Measure

Cost layer:   RZZ(2γ * Q_ij) cho mỗi cặp (i,j) có Q_ij ≠ 0
              RZ(2γ * Q_ii)  cho diagonal
Mixer layer:  RX(2β) cho mỗi qubit

Tham số tối ưu hóa: θ = [γ₁, β₁, ..., γₚ, βₚ]  (2p tham số)
"""

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Import Qiskit — lazy import để module vẫn importable khi chưa cài
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter, ParameterVector
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    logger.warning("Qiskit chưa được cài. circuit.py chạy ở chế độ stub.")


@dataclass
class QAOACircuitConfig:
    """
    Cấu hình mạch QAOA.

    Attributes
    ----------
    n_qubits : int
        Số qubit = số tài sản.
    depth : int
        Số layer p của QAOA.
    n_params : int
        Tổng số tham số = 2 * depth (γ và β cho mỗi layer).
    initial_params : np.ndarray
        Điểm khởi tạo tham số [γ₁, β₁, ..., γₚ, βₚ].
    """

    n_qubits: int
    depth: int
    n_params: int
    initial_params: np.ndarray


def build_qaoa_circuit(
    Q: np.ndarray,
    depth: int = 2,
    gammas: np.ndarray | None = None,
    betas: np.ndarray | None = None,
) -> "QuantumCircuit":
    """
    Xây dựng mạch QAOA cho bài toán QUBO với ma trận Q.

    Parameters
    ----------
    Q : np.ndarray
        Ma trận QUBO (n × n), symmetric.
    depth : int
        Số layer QAOA (p). Nhiều layer → xấp xỉ tốt hơn nhưng nhiễu hơn.
    gammas : np.ndarray, optional
        Tham số phase (cost) [γ₁, ..., γₚ]. Nếu None → dùng ParameterVector.
    betas : np.ndarray, optional
        Tham số mixer [β₁, ..., βₚ]. Nếu None → dùng ParameterVector.

    Returns
    -------
    QuantumCircuit
        Mạch QAOA đã parameterized, sẵn sàng để bind tham số và chạy.

    Raises
    ------
    ImportError
        Nếu Qiskit chưa được cài đặt.
    """
    if not QISKIT_AVAILABLE:
        raise ImportError(
            "Qiskit chưa được cài. Chạy: pip install qiskit qiskit-aer"
        )

    n = Q.shape[0]

    # Tạo tham số symbolic nếu chưa có giá trị cụ thể
    use_params = (gammas is None or betas is None)
    if use_params:
        gamma_params = ParameterVector("γ", depth)
        beta_params = ParameterVector("β", depth)
    else:
        gamma_params = gammas
        beta_params = betas

    # ── Khởi tạo mạch ────────────────────────────────────────
    qc = QuantumCircuit(n, n)

    # Hadamard: đưa tất cả qubit về superposition |+⟩^n
    qc.h(range(n))
    qc.barrier()

    # ── p layers ─────────────────────────────────────────────
    for layer in range(depth):
        g = gamma_params[layer]
        b = beta_params[layer]

        # Cost unitary: U_C(γ) = exp(-iγH_C)
        # H_C = x^T Q x → RZZ cho off-diagonal, RZ cho diagonal
        _apply_cost_layer(qc, Q, g)
        qc.barrier()

        # Mixer unitary: U_M(β) = exp(-iβH_B), H_B = Σ X_i
        _apply_mixer_layer(qc, n, b)
        qc.barrier()

    # Đo tất cả qubit
    qc.measure(range(n), range(n))

    logger.info(
        f"QAOA circuit: {n} qubits, depth={depth}, "
        f"gates≈{qc.size()}, depth_circuit={qc.depth()}"
    )
    return qc


def _apply_cost_layer(qc: "QuantumCircuit", Q: np.ndarray, gamma) -> None:
    """
    Áp dụng Cost layer cho một QAOA layer.

    Cost Hamiltonian: H_C = Σ_i Q_ii x_i + Σ_{i<j} Q_ij x_i x_j
    Unitary: e^{-iγH_C}

    RZ(2γ Q_ii) cho diagonal
    RZZ(2γ Q_ij) cho cặp (i,j) có Q_ij ≠ 0
    """
    n = Q.shape[0]
    threshold = 1e-10  # bỏ qua coupling quá nhỏ

    # Diagonal terms → RZ gates
    for i in range(n):
        angle = 2 * gamma * Q[i, i]
        qc.rz(angle, i)

    # Off-diagonal terms → RZZ gates
    for i in range(n):
        for j in range(i + 1, n):
            if abs(Q[i, j]) > threshold:
                angle = 2 * gamma * Q[i, j]
                # RZZ = CNOT - RZ - CNOT
                qc.rzz(angle, i, j)


def _apply_mixer_layer(qc: "QuantumCircuit", n: int, beta) -> None:
    """
    Áp dụng Mixer layer: H_B = Σ_i X_i
    Unitary: e^{-iβH_B} = Π_i RX(2β)_i
    """
    for i in range(n):
        qc.rx(2 * beta, i)


def get_circuit_config(
    n_qubits: int,
    depth: int = 2,
    seed: int = 42,
) -> QAOACircuitConfig:
    """
    Tạo cấu hình QAOA và khởi tạo tham số ngẫu nhiên.

    Parameters
    ----------
    n_qubits : int
        Số qubit.
    depth : int
        Số layer p.
    seed : int
        Random seed để reproducible.

    Returns
    -------
    QAOACircuitConfig
    """
    n_params = 2 * depth
    rng = np.random.default_rng(seed)

    # Khởi tạo trong khoảng hợp lý:
    # γ ∈ [0, π],  β ∈ [0, π/2]
    gammas = rng.uniform(0, np.pi, depth)
    betas = rng.uniform(0, np.pi / 2, depth)
    initial_params = np.concatenate([gammas, betas])

    return QAOACircuitConfig(
        n_qubits=n_qubits,
        depth=depth,
        n_params=n_params,
        initial_params=initial_params,
    )


def bind_parameters(
    qc: "QuantumCircuit",
    params: np.ndarray,
    depth: int,
) -> "QuantumCircuit":
    """
    Gán giá trị cụ thể vào mạch QAOA đã parameterized.

    Parameters
    ----------
    qc : QuantumCircuit
        Mạch parameterized với ParameterVector γ và β.
    params : np.ndarray
        Vector [γ₁, ..., γₚ, β₁, ..., βₚ] (2p giá trị).
    depth : int
        Số layer p.

    Returns
    -------
    QuantumCircuit
        Mạch đã bind tham số, sẵn sàng để chạy.
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit chưa được cài.")

    gammas = params[:depth]
    betas = params[depth:]

    param_dict = {}
    for i, param in enumerate(qc.parameters):
        name = str(param)
        if name.startswith("γ"):
            idx = int(name.split("[")[1].rstrip("]"))
            param_dict[param] = gammas[idx]
        elif name.startswith("β"):
            idx = int(name.split("[")[1].rstrip("]"))
            param_dict[param] = betas[idx]

    return qc.assign_parameters(param_dict)


def circuit_to_dict(qc: "QuantumCircuit") -> dict:
    """Chuyển mạch thành dict để serialize/log."""
    if not QISKIT_AVAILABLE:
        return {}
    return {
        "n_qubits": qc.num_qubits,
        "n_classical": qc.num_clbits,
        "n_gates": qc.size(),
        "depth": qc.depth(),
        "parameters": [str(p) for p in qc.parameters],
    }