"""
src/quantum/backend.py
----------------------
Quản lý kết nối đến IBM Quantum và lựa chọn backend.

Hỗ trợ 3 chế độ:
  1. AerSimulator   — local, nhanh, không nhiễu (dùng để dev/test)
  2. FakeBackend    — local, giả lập noise model của IBM hardware
  3. IBM Quantum    — cloud, chạy trên qubit thật (cần IBM_QUANTUM_TOKEN)
"""

# src/quantum/backend.py

# src/quantum/backend.py

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    AER_AVAILABLE = True
except ImportError:
    AER_AVAILABLE = False

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2
    # ĐÃ SỬA: Bỏ import FakeBrisbanev2 trực tiếp ở đây để tránh crash 
    IBM_AVAILABLE = True
except ImportError:
    IBM_AVAILABLE = False


def get_aer_simulator(noise_model: bool = False, backend_name: str = "ibm_brisbane") -> "AerSimulator":
    """Trả về AerSimulator (local)."""
    if not AER_AVAILABLE:
        raise ImportError("qiskit-aer chưa được cài. Chạy: pip install qiskit-aer")

    # Nếu muốn dùng noise model của máy thật, ta sẽ lấy trực tiếp qua cloud service nếu có token
    if noise_model and IBM_AVAILABLE:
        try:
            token = os.getenv("IBM_QUANTUM_TOKEN")
            if token:
                service = QiskitRuntimeService(channel="ibm_quantum", token=token)
                real_backend = service.backend(backend_name)
                sim = AerSimulator.from_backend(real_backend)
                logger.info(f"AerSimulator với noise model thực tế từ cloud của {backend_name}")
                return sim
        except Exception as e:
            logger.warning(f"Không lấy được noise model từ cloud: {e}. Chuyển về ideal simulator.")

    logger.info("AerSimulator (ideal, không nhiễu)")
    return AerSimulator()


def get_fake_backend():
    """
    Trả về FakeBackend. Do phiên bản mới thay đổi vị trí, 
    ta fallback về một AerSimulator sạch để không làm gãy luồng chạy của hệ thống.
    """
    logger.warning("FakeBrisbanev2 không khả dụng trên phiên bản này. Trả về AerSimulator mặc định.")
    return AerSimulator()


def get_ibm_backend(
    backend_name: str = "ibm_brisbane",
    token: Optional[str] = None,
    # SỬA TẠI ĐÂY: Đổi từ "ibm_quantum" thành "ibm_quantum_platform"
    channel: str = "ibm_quantum_platform",
):
    """
    Kết nối IBM Quantum và lấy backend thật.
    ...
    """
    if not IBM_AVAILABLE:
        raise ImportError(
            "qiskit-ibm-runtime chưa được cài. "
            "Chạy: pip install qiskit-ibm-runtime"
        )

    token = token or os.getenv("IBM_QUANTUM_TOKEN")

    if token:
        logger.info("Đăng nhập IBM Quantum với token...")
        QiskitRuntimeService.save_account(
            channel=channel,
            token=token,
            overwrite=True,
        )

    try:
        service = QiskitRuntimeService(channel=channel)
    except Exception as e:
        raise ValueError(
            f"Không kết nối được IBM Quantum: {e}\n"
            "Kiểm tra IBM_QUANTUM_TOKEN trong file .env"
        )

    # Chọn backend ít queue nhất nếu backend_name là 'auto'
    if backend_name == "auto":
        backends = service.backends(
            simulator=False,
            operational=True,
            min_num_qubits=10,
        )
        if not backends:
            raise RuntimeError("Không tìm thấy backend IBM Quantum khả dụng.")
        backend = min(backends, key=lambda b: b.status().pending_jobs)
        logger.info(f"Auto-selected backend: {backend.name}")
    else:
        backend = service.backend(backend_name)
        logger.info(f"IBM Quantum backend: {backend.name}")

    status = backend.status()
    logger.info(
        f"  Operational: {status.operational}, "
        f"Pending jobs: {status.pending_jobs}, "
        f"Qubits: {backend.num_qubits}"
    )
    return backend


def get_backend(
    use_simulator: bool = True,
    noise_model: bool = False,
    ibm_backend_name: str = "ibm_brisbane",
    token: Optional[str] = None,
):
    """
    Factory function — chọn backend dựa trên config.

    Parameters
    ----------
    use_simulator : bool
        True → AerSimulator (local), False → IBM hardware (cloud).
    noise_model : bool
        Chỉ dùng khi use_simulator=True.
        True → inject noise model của IBM backend.
    ibm_backend_name : str
        Tên IBM backend (dùng khi use_simulator=False).
    token : str, optional
        IBM Quantum token.

    Returns
    -------
    Backend (AerSimulator hoặc IBMBackend)
    """
    if use_simulator:
        return get_aer_simulator(noise_model=noise_model, backend_name=ibm_backend_name)
    else:
        return get_ibm_backend(backend_name=ibm_backend_name, token=token)


def check_backend_compatibility(backend, n_qubits: int) -> bool:
    """
    Kiểm tra backend có đủ qubit cho bài toán không.

    Parameters
    ----------
    backend
        Qiskit backend.
    n_qubits : int
        Số qubit cần thiết.

    Returns
    -------
    bool
        True nếu backend hỗ trợ đủ qubit.
    """
    try:
        available = backend.num_qubits
        if available < n_qubits:
            logger.error(
                f"Backend chỉ có {available} qubit, "
                f"bài toán cần {n_qubits} qubit."
            )
            return False
        logger.info(f"Backend OK: {available} qubit (cần {n_qubits})")
        return True
    except AttributeError:
        # Một số fake backend không có num_qubits
        return True


def print_backend_info(backend) -> None:
    """In thông tin backend ra console."""
    try:
        name = backend.name if not callable(backend.name) else backend.name()
        print(f"\nBackend: {name}")
        try:
            print(f"  Qubits    : {backend.num_qubits}")
        except Exception:
            pass
        try:
            status = backend.status()
            print(f"  Operational: {status.operational}")
            print(f"  Pending jobs: {status.pending_jobs}")
        except Exception:
            print("  (local simulator — no queue)")
    except Exception as e:
        print(f"Backend info unavailable: {e}")