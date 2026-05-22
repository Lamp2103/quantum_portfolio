"""
tests/test_qubo.py
------------------
Unit tests cho src/quantum/: qubo, circuit (stub), sampler, optimizer (stub).

Không cần Qiskit để chạy các test này — circuit/backend được mock hoàn toàn.
Chỉ test phần logic thuần Python: QUBO formulation, decode, sampling analysis.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quantum.qubo import (
    QUBOProblem,
    brute_force_qubo,
    build_qubo,
    get_optimal_budget,
)
from src.quantum.sampler import (
    SamplingResult,
    aggregate_weights_from_distribution,
    decode_weights,
    parse_counts,
)
from src.quantum.optimizer import (
    COBYLAOptimizer,
    SPSAOptimizer,
    counts_to_expectation,
    sample_best_bitstring,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_market():
    """Thị trường 4 tài sản nhỏ — đủ để brute-force."""
    tickers = ["A", "B", "C", "D"]
    mu = pd.Series([0.20, 0.15, 0.10, 0.05], index=tickers)
    vols = np.array([0.25, 0.18, 0.12, 0.08])
    corr = np.array([
        [1.00, 0.50, 0.30, 0.10],
        [0.50, 1.00, 0.40, 0.20],
        [0.30, 0.40, 1.00, 0.30],
        [0.10, 0.20, 0.30, 1.00],
    ])
    cov_arr = np.outer(vols, vols) * corr
    cov = pd.DataFrame(cov_arr, index=tickers, columns=tickers)
    return mu, cov


@pytest.fixture
def qubo_2of4(small_market):
    """QUBO chọn 2 trong 4 tài sản."""
    mu, cov = small_market
    return build_qubo(mu, cov, budget=2, risk_aversion=0.5)


# ── Tests: build_qubo ─────────────────────────────────────────────────────────

class TestBuildQUBO:
    def test_returns_qubo_problem(self, small_market):
        mu, cov = small_market
        q = build_qubo(mu, cov, budget=2)
        assert isinstance(q, QUBOProblem)

    def test_q_shape(self, small_market):
        mu, cov = small_market
        q = build_qubo(mu, cov, budget=2)
        assert q.Q.shape == (4, 4)

    def test_q_symmetric(self, small_market):
        mu, cov = small_market
        q = build_qubo(mu, cov, budget=2)
        assert np.allclose(q.Q, q.Q.T, atol=1e-10), "Q phải symmetric"

    def test_budget_stored(self, small_market):
        mu, cov = small_market
        q = build_qubo(mu, cov, budget=3)
        assert q.budget == 3

    def test_n_assets(self, small_market):
        mu, cov = small_market
        q = build_qubo(mu, cov, budget=2)
        assert q.n_assets == 4
        assert q.tickers == ["A", "B", "C", "D"]

    def test_invalid_budget_raises(self, small_market):
        mu, cov = small_market
        with pytest.raises(ValueError):
            build_qubo(mu, cov, budget=0)
        with pytest.raises(ValueError):
            build_qubo(mu, cov, budget=10)

    def test_risk_aversion_effect(self, small_market):
        """Risk aversion cao hơn → thay đổi diagonal của Q."""
        mu, cov = small_market
        q_low = build_qubo(mu, cov, budget=2, risk_aversion=0.1)
        q_high = build_qubo(mu, cov, budget=2, risk_aversion=0.9)
        # Diagonal phải khác nhau
        assert not np.allclose(np.diag(q_low.Q), np.diag(q_high.Q))

    def test_custom_penalty(self, small_market):
        mu, cov = small_market
        q = build_qubo(mu, cov, budget=2, penalty=100.0)
        assert q.penalty == 100.0

    def test_summary_string(self, qubo_2of4):
        s = qubo_2of4.summary()
        assert "QUBO" in s
        assert "budget" in s.lower() or "Budget" in s


# ── Tests: QUBOProblem methods ────────────────────────────────────────────────

class TestQUBOProblem:
    def test_evaluate_feasible_solution(self, qubo_2of4):
        """x với 2 tài sản → evaluate trả về float."""
        x = np.array([1, 1, 0, 0], dtype=float)
        val = qubo_2of4.evaluate(x)
        assert isinstance(val, float)

    def test_evaluate_zero_for_zero_vector(self, qubo_2of4):
        """x = 0 → x^T Q x = 0."""
        x = np.zeros(4)
        assert qubo_2of4.evaluate(x) == pytest.approx(0.0)

    def test_is_feasible_correct(self, qubo_2of4):
        assert qubo_2of4.is_feasible(np.array([1, 1, 0, 0], dtype=float))
        assert qubo_2of4.is_feasible(np.array([0, 1, 0, 1], dtype=float))
        assert not qubo_2of4.is_feasible(np.array([1, 0, 0, 0], dtype=float))
        assert not qubo_2of4.is_feasible(np.array([1, 1, 1, 0], dtype=float))

    def test_decode_weights_equal(self, qubo_2of4):
        x = np.array([1, 0, 1, 0], dtype=float)
        w = qubo_2of4.decode_weights(x)
        assert isinstance(w, pd.Series)
        assert abs(w.sum() - 1.0) < 1e-10
        assert w["A"] == pytest.approx(0.5)
        assert w["C"] == pytest.approx(0.5)
        assert w["B"] == pytest.approx(0.0)

    def test_decode_weights_all_zero(self, qubo_2of4):
        """x = 0 → equal weight fallback."""
        x = np.zeros(4)
        w = qubo_2of4.decode_weights(x)
        # Khi không chọn gì, tổng vẫn phải là 0 (no selection)
        assert w.sum() == pytest.approx(0.0)

    def test_penalty_violation_zero_for_feasible(self, qubo_2of4):
        x = np.array([1, 1, 0, 0], dtype=float)
        assert qubo_2of4.penalty_violation(x) == pytest.approx(0.0)

    def test_penalty_violation_nonzero_for_infeasible(self, qubo_2of4):
        x = np.array([1, 0, 0, 0], dtype=float)  # chỉ 1 tài sản thay vì 2
        assert qubo_2of4.penalty_violation(x) > 0


# ── Tests: brute_force_qubo ───────────────────────────────────────────────────

class TestBruteForce:
    def test_finds_feasible_solution(self, qubo_2of4):
        x_opt, val = brute_force_qubo(qubo_2of4)
        assert qubo_2of4.is_feasible(x_opt), "Solution phải thỏa budget"

    def test_returns_ndarray(self, qubo_2of4):
        x_opt, val = brute_force_qubo(qubo_2of4)
        assert isinstance(x_opt, np.ndarray)
        assert isinstance(val, float)

    def test_optimal_better_than_any_feasible(self, qubo_2of4):
        """Brute-force solution phải tốt hơn hoặc bằng mọi feasible solution khác."""
        x_opt, best_val = brute_force_qubo(qubo_2of4)
        # Thử tất cả feasible combinations
        n = qubo_2of4.n_assets
        for bits in range(2 ** n):
            x = np.array([(bits >> i) & 1 for i in range(n)], dtype=float)
            if qubo_2of4.is_feasible(x):
                val = qubo_2of4.evaluate(x)
                assert best_val <= val + 1e-8, \
                    f"Brute-force tìm {best_val:.4f} nhưng {x} cho {val:.4f}"

    def test_raises_for_large_n(self, small_market):
        mu, cov = small_market
        # Tạo QUBO với n=25 (giả lập)
        big_mu = pd.Series(np.ones(25) * 0.1, index=[f"T{i}" for i in range(25)])
        big_cov = pd.DataFrame(np.eye(25) * 0.04,
                               index=big_mu.index, columns=big_mu.index)
        big_qubo = build_qubo(big_mu, big_cov, budget=5)
        with pytest.raises(ValueError, match="Brute-force"):
            brute_force_qubo(big_qubo)


# ── Tests: get_optimal_budget ─────────────────────────────────────────────────

class TestGetOptimalBudget:
    def test_budget_at_least_one(self):
        assert get_optimal_budget(2, max_qubits=10) >= 1

    def test_budget_does_not_exceed_assets(self):
        b = get_optimal_budget(4, max_qubits=10)
        assert b <= 4

    def test_budget_respects_max_qubits(self):
        b = get_optimal_budget(20, max_qubits=6)
        assert b <= 3  # max_qubits//2

    def test_budget_half_of_assets(self):
        assert get_optimal_budget(8, max_qubits=20) == 4


# ── Tests: sampler / parse_counts ─────────────────────────────────────────────

class TestParseCounts:
    @pytest.fixture
    def sample_counts(self, qubo_2of4):
        """Giả lập counts từ quantum measurement."""
        return {
            "0011": 200,   # qubit 0,1 = 1 → chọn A,B (feasible, budget=2)
            "0101": 150,   # qubit 0,2 = 1 → chọn A,C (feasible)
            "1001": 100,   # qubit 0,3 = 1 → chọn A,D (feasible)
            "0001": 80,    # chỉ qubit 0 = 1 → infeasible
            "1111": 20,    # 4 tài sản → infeasible
        }

    def test_parse_counts_returns_sampling_result(self, qubo_2of4, sample_counts):
        result = parse_counts(sample_counts, qubo_2of4.Q, qubo_2of4.budget)
        assert isinstance(result, SamplingResult)

    def test_total_shots(self, qubo_2of4, sample_counts):
        result = parse_counts(sample_counts, qubo_2of4.Q, qubo_2of4.budget)
        assert result.total_shots == sum(sample_counts.values())

    def test_n_unique_states(self, qubo_2of4, sample_counts):
        result = parse_counts(sample_counts, qubo_2of4.Q, qubo_2of4.budget)
        assert result.n_unique_states == len(sample_counts)

    def test_best_bitstring_is_feasible(self, qubo_2of4, sample_counts):
        result = parse_counts(sample_counts, qubo_2of4.Q, qubo_2of4.budget)
        assert qubo_2of4.is_feasible(result.best_bitstring), \
            "Best bitstring phải thỏa budget khi có feasible solutions"

    def test_probability_sums_to_one(self, qubo_2of4, sample_counts):
        result = parse_counts(sample_counts, qubo_2of4.Q, qubo_2of4.budget)
        assert result.probability_distribution.sum() == pytest.approx(1.0, abs=1e-6)

    def test_top_bitstrings_dataframe(self, qubo_2of4, sample_counts):
        result = parse_counts(sample_counts, qubo_2of4.Q, qubo_2of4.budget)
        assert isinstance(result.top_bitstrings, pd.DataFrame)
        assert "objective" in result.top_bitstrings.columns


# ── Tests: decode_weights ─────────────────────────────────────────────────────

class TestDecodeWeights:
    @pytest.fixture
    def tickers(self):
        return ["A", "B", "C", "D"]

    def test_equal_weights_sum_to_one(self, tickers):
        x = np.array([1, 0, 1, 0], dtype=float)
        w = decode_weights(x, tickers, method="equal")
        assert abs(w.sum() - 1.0) < 1e-10

    def test_equal_weights_correct(self, tickers):
        x = np.array([1, 0, 1, 0], dtype=float)
        w = decode_weights(x, tickers, method="equal")
        assert w["A"] == pytest.approx(0.5)
        assert w["C"] == pytest.approx(0.5)
        assert w["B"] == pytest.approx(0.0)
        assert w["D"] == pytest.approx(0.0)

    def test_risk_weighted_sum_to_one(self, tickers, small_market):
        mu, cov = small_market
        x = np.array([1, 1, 0, 0], dtype=float)
        w = decode_weights(x, tickers, cov=cov, method="risk_weighted")
        assert abs(w.sum() - 1.0) < 1e-10

    def test_risk_weighted_lower_vol_gets_more(self, tickers, small_market):
        """B có vol thấp hơn A → B nhận trọng số cao hơn khi dùng risk_weighted."""
        mu, cov = small_market
        x = np.array([1, 1, 0, 0], dtype=float)  # chọn A và B
        w = decode_weights(x, tickers, cov=cov, method="risk_weighted")
        # vol_A = 0.25 > vol_B = 0.18 → w_B > w_A
        assert w["B"] > w["A"], f"Expected w_B > w_A, got w_B={w['B']:.4f}, w_A={w['A']:.4f}"

    def test_return_weighted_sum_to_one(self, tickers, small_market):
        mu, cov = small_market
        x = np.array([1, 1, 0, 0], dtype=float)
        w = decode_weights(x, tickers, mu=mu, method="return_weighted")
        assert abs(w.sum() - 1.0) < 1e-10

    def test_return_weighted_higher_return_gets_more(self, tickers, small_market):
        """A có return cao hơn B → A nhận trọng số cao hơn."""
        mu, cov = small_market
        x = np.array([1, 1, 0, 0], dtype=float)
        w = decode_weights(x, tickers, mu=mu, method="return_weighted")
        assert w["A"] > w["B"]

    def test_invalid_method_raises(self, tickers):
        x = np.array([1, 0, 1, 0], dtype=float)
        with pytest.raises(ValueError, match="method"):
            decode_weights(x, tickers, method="unknown")

    def test_no_selection_returns_equal(self, tickers):
        """Không chọn tài sản nào → fallback."""
        x = np.zeros(4)
        w = decode_weights(x, tickers, method="equal")
        # Tổng = 0 (không có tài sản được chọn)
        assert w.sum() == pytest.approx(0.0)


# ── Tests: optimizer helpers ──────────────────────────────────────────────────

class TestOptimizerHelpers:
    def test_counts_to_expectation(self, qubo_2of4):
        counts = {"0011": 500, "0101": 500}
        shots = 1000
        exp = counts_to_expectation(counts, qubo_2of4.Q, shots)
        assert isinstance(exp, float)
        # Xác minh thủ công
        x1 = np.array([1, 1, 0, 0], dtype=float)  # "0011" reversed
        x2 = np.array([1, 0, 1, 0], dtype=float)  # "0101" reversed
        expected = 0.5 * (x1 @ qubo_2of4.Q @ x1) + 0.5 * (x2 @ qubo_2of4.Q @ x2)
        assert exp == pytest.approx(expected, rel=1e-6)

    def test_sample_best_bitstring_prefers_feasible(self, qubo_2of4):
        counts = {
            "0011": 100,   # feasible (A,B)
            "0001": 500,   # infeasible nhưng xuất hiện nhiều hơn
        }
        x_best, val = sample_best_bitstring(counts, qubo_2of4.Q, qubo_2of4.budget)
        assert qubo_2of4.is_feasible(x_best), "Phải ưu tiên feasible solution"

    def test_cobyla_optimizer_minimizes(self):
        """COBYLA phải tìm minimum của hàm bậc 2 đơn giản."""
        opt = COBYLAOptimizer(maxiter=500)
        x_opt, f_opt, n_evals, history = opt.minimize(
            fun=lambda x: (x[0] - 2.0) ** 2 + (x[1] + 1.0) ** 2,
            x0=np.array([0.0, 0.0]),
        )
        assert f_opt < 0.01, f"COBYLA không hội tụ: f={f_opt:.4f}"
        assert abs(x_opt[0] - 2.0) < 0.1
        assert abs(x_opt[1] + 1.0) < 0.1

    def test_cobyla_tracks_history(self):
        opt = COBYLAOptimizer(maxiter=50)
        _, _, n_evals, history = opt.minimize(
            fun=lambda x: x[0] ** 2,
            x0=np.array([5.0]),
        )
        assert len(history) > 0
        assert n_evals == len(history)

    def test_spsa_optimizer_reduces_objective(self):
        """SPSA phải giảm được objective so với điểm khởi đầu."""
        opt = SPSAOptimizer(maxiter=100, learning_rate=0.3, perturbation=0.2)
        x0 = np.array([5.0, 5.0])
        f0 = x0[0] ** 2 + x0[1] ** 2  # = 50
        x_opt, f_opt, _, _ = opt.minimize(
            fun=lambda x: x[0] ** 2 + x[1] ** 2,
            x0=x0,
        )
        assert f_opt < f0, f"SPSA không giảm được objective: f0={f0}, f_opt={f_opt:.4f}"


# ── Tests: aggregate_weights ──────────────────────────────────────────────────

class TestAggregateWeights:
    def test_aggregate_weights_sum_to_one(self, qubo_2of4, small_market):
        mu, cov = small_market
        counts = {"0011": 400, "0101": 300, "1001": 200, "0110": 100}
        sampling_result = parse_counts(counts, qubo_2of4.Q, qubo_2of4.budget)
        tickers = qubo_2of4.tickers
        w = aggregate_weights_from_distribution(
            sampling_result, tickers, mu=mu, cov=cov,
            weight_method="equal", top_k=3,
        )
        assert abs(w.sum() - 1.0) < 1e-6

    def test_aggregate_weights_non_negative(self, qubo_2of4, small_market):
        mu, cov = small_market
        counts = {"0011": 400, "0101": 300, "1001": 200, "0110": 100}
        sampling_result = parse_counts(counts, qubo_2of4.Q, qubo_2of4.budget)
        w = aggregate_weights_from_distribution(
            sampling_result, qubo_2of4.tickers, top_k=3
        )
        assert (w >= -1e-10).all()