"""
run_experiment.py
-----------------
CLI để chạy toàn bộ pipeline từ dòng lệnh.

Cách dùng:
    python run_experiment.py --stage data
    python run_experiment.py --stage classical
    python run_experiment.py --stage quantum
    python run_experiment.py --stage all
    python run_experiment.py --stage quantum --use-ibm
    python run_experiment.py --stage quantum --noise-model
    python run_experiment.py --stage data --tickers AAPL MSFT GOOGL --force-download
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# GIAI ĐOẠN 1: Thu thập & xử lý dữ liệu
# ══════════════════════════════════════════════════════════════════

def run_data_stage(args):
    """Giai đoạn 1: Thu thập và xử lý dữ liệu."""
    from src.data.fetcher import fetch_prices
    from src.data.processor import process
    from src.data.validator import run_all_checks

    tickers = args.tickers or None

    logger.info("=" * 55)
    logger.info("GIAI ĐOẠN 1: THU THẬP & XỬ LÝ DỮ LIỆU")
    logger.info("=" * 55)

    prices = fetch_prices(
        tickers=tickers,
        config_path=args.config,
        force_download=args.force_download,
    )
    print(f"\n✓ Đã tải: {prices.shape[1]} cổ phiếu × {prices.shape[0]} ngày")
    print(prices.tail(3).to_string())

    portfolio_data = process(prices, config_path=args.config, save=True)

    run_all_checks(
        portfolio_data.prices,
        portfolio_data.mu,
        portfolio_data.cov,
        raise_on_error=True,
    )

    print("\n" + "=" * 55)
    print("TÓM TẮT DANH MỤC")
    print("=" * 55)
    print(portfolio_data.summary().to_string())
    print(f"\n✓ Giai đoạn 1 hoàn thành. Dữ liệu lưu tại data/processed/")

    return portfolio_data


# ══════════════════════════════════════════════════════════════════
# GIAI ĐOẠN 2: Classical Solver
# ══════════════════════════════════════════════════════════════════

def run_classical_stage(args, portfolio_data=None):
    """Giai đoạn 2: Chạy Classical Solver (Markowitz MVO)."""
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import yaml

    from src.classical.frontier import compute_efficient_frontier
    from src.classical.metrics import compare_portfolios
    from src.classical.mvo import maximize_sharpe, minimize_variance, solve_mvo

    logger.info("=" * 55)
    logger.info("GIAI ĐOẠN 2: CLASSICAL SOLVER (MARKOWITZ MVO)")
    logger.info("=" * 55)

    if portfolio_data is None:
        proc_dir = Path("data/processed")
        if not (proc_dir / "mu.csv").exists():
            raise FileNotFoundError("Chưa có dữ liệu xử lý. Chạy --stage data trước.")
        mu = pd.read_csv(proc_dir / "mu.csv", index_col=0).squeeze()
        cov = pd.read_csv(proc_dir / "covariance.csv", index_col=0)
        returns = pd.read_csv(proc_dir / "returns.csv", index_col=0, parse_dates=True)
        logger.info(f"Đọc dữ liệu từ {proc_dir}/: {len(mu)} cổ phiếu")
    else:
        mu = portfolio_data.mu
        cov = portfolio_data.cov
        returns = portfolio_data.returns

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rf = cfg["classical"]["risk_free_rate"]
    n_points = cfg["classical"]["num_frontier_points"]

    print("\n" + "─" * 55)
    print("1. GLOBAL MINIMUM VARIANCE PORTFOLIO")
    print("─" * 55)
    gmv = minimize_variance(mu, cov, risk_free_rate=rf)
    print(gmv)

    print("\n" + "─" * 55)
    print("2. MAXIMUM SHARPE PORTFOLIO")
    print("─" * 55)
    msp = maximize_sharpe(mu, cov, risk_free_rate=rf)
    print(msp)

    print("\n" + "─" * 55)
    print("3. EFFICIENT FRONTIER")
    print("─" * 55)
    ef = compute_efficient_frontier(mu, cov, n_points=n_points, risk_free_rate=rf)
    print(ef.summary())

    print("\n" + "─" * 55)
    print("4. SO SÁNH HIỆU SUẤT (backtested trên returns)")
    print("─" * 55)
    portfolios = {
        "Min Variance":  gmv.weights,
        "Max Sharpe":    msp.weights,
        "Equal Weight":  pd.Series(np.ones(len(mu)) / len(mu), index=mu.index),
    }
    comparison = compare_portfolios(portfolios, returns, risk_free_rate=rf)
    fmt = comparison.copy()
    pct_rows = ["Annualized Return", "Annualized Volatility", "Max Drawdown", "VaR (95%)", "CVaR (95%)"]
    for row in pct_rows:
        if row in fmt.index:
            fmt.loc[row] = fmt.loc[row].map(lambda x: f"{x:.2%}")
    float_rows = ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"]
    for row in float_rows:
        if row in fmt.index:
            fmt.loc[row] = fmt.loc[row].map(lambda x: f"{x:.4f}")
    print(fmt.to_string())

    results_dir = Path("data/processed")
    gmv.weights.to_csv(results_dir / "classical_gmv_weights.csv", header=["weight"])
    msp.weights.to_csv(results_dir / "classical_msp_weights.csv", header=["weight"])
    ef.weights_df.to_csv(results_dir / "classical_frontier_weights.csv")
    pd.DataFrame({
        "return":     ef.returns,
        "volatility": ef.volatilities,
        "sharpe":     ef.sharpe_ratios,
    }).to_csv(results_dir / "classical_frontier.csv", index=False)

    print(f"\n✓ Giai đoạn 2 hoàn thành. Kết quả lưu tại {results_dir}/")
    return gmv, msp, ef


# ══════════════════════════════════════════════════════════════════
# GIAI ĐOẠN 3: Quantum Solver (QAOA)
# ══════════════════════════════════════════════════════════════════

def run_quantum_stage(args, portfolio_data=None):
    """Giai đoạn 3: Chạy Quantum Solver (QAOA)."""
    import json
    import os
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import yaml

    from src.classical.metrics import compare_portfolios
    from src.quantum.backend import get_backend, print_backend_info
    from src.quantum.circuit import bind_parameters, build_qaoa_circuit
    from src.quantum.optimizer import optimize_qaoa
    from src.quantum.qubo import brute_force_qubo, build_qubo, get_optimal_budget
    from src.quantum.sampler import (
        aggregate_weights_from_distribution,
        parse_counts,
        print_sampling_summary,
    )

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    logger.info("=" * 55)
    logger.info("GIAI ĐOẠN 3: QUANTUM SOLVER (QAOA)")
    logger.info("=" * 55)

    # ── Load dữ liệu ─────────────────────────────────────────────
    if portfolio_data is None:
        proc_dir = Path("data/processed")
        if not (proc_dir / "mu.csv").exists():
            raise FileNotFoundError("Chưa có dữ liệu xử lý. Chạy --stage data trước.")
        mu      = pd.read_csv(proc_dir / "mu.csv",         index_col=0).squeeze()
        cov     = pd.read_csv(proc_dir / "covariance.csv", index_col=0)
        returns = pd.read_csv(proc_dir / "returns.csv",    index_col=0, parse_dates=True)
        logger.info(f"Đọc từ {proc_dir}/: {len(mu)} cổ phiếu")
    else:
        mu      = portfolio_data.mu
        cov     = portfolio_data.cov
        returns = portfolio_data.returns

    # ── Đọc config ───────────────────────────────────────────────
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    q_cfg = cfg["quantum"]
    rf    = cfg["classical"]["risk_free_rate"]

    use_ibm     = getattr(args, "use_ibm",     False)
    noise_model = getattr(args, "noise_model", False)
    depth       = q_cfg["qaoa_depth"]
    optimizer   = q_cfg["optimizer"]
    max_iter    = q_cfg["max_iterations"]
    shots       = q_cfg["shots"]
    max_qubits  = q_cfg["max_qubits"]

    # ── 1. Giới hạn số tài sản theo max_qubits ───────────────────
    n_use = min(len(mu), max_qubits)
    if n_use < len(mu):
        logger.info(f"Giới hạn {len(mu)} → {n_use} tài sản (max_qubits={max_qubits})")
        vols = pd.Series(np.sqrt(np.diag(cov.values)), index=mu.index)
        sharpe_approx = mu / vols
        top_tickers = sharpe_approx.nlargest(n_use).index.tolist()
        mu  = mu[top_tickers]
        cov = cov.loc[top_tickers, top_tickers]

    budget = get_optimal_budget(len(mu), max_qubits)
    logger.info(f"Tài sản: {list(mu.index)}")
    logger.info(f"Budget B={budget} (chọn {budget}/{len(mu)} tài sản)")

    # ── 2. Build QUBO ─────────────────────────────────────────────
    print("\n" + "─" * 55)
    print("1. QUBO FORMULATION")
    print("─" * 55)
    qubo = build_qubo(mu, cov, budget=budget, risk_aversion=0.5)
    print(qubo.summary())

    # ── 3. Brute-force reference (n ≤ 20) ────────────────────────
    w_bf = None
    if len(mu) <= 20:
        print("\n" + "─" * 55)
        print("2. BRUTE-FORCE REFERENCE")
        print("─" * 55)
        x_bf, val_bf = brute_force_qubo(qubo)
        w_bf = qubo.decode_weights(x_bf)
        selected = [t for t, v in zip(qubo.tickers, x_bf) if v == 1]
        print(f"  Optimal bitstring : {''.join(map(str, x_bf.astype(int)))}")
        print(f"  Selected          : {selected}")
        print(f"  QUBO objective    : {val_bf:.6f}")
        print(f"  Weights           : {dict(w_bf[w_bf > 1e-4].round(4))}")

    # ── 4. Backend ────────────────────────────────────────────────
    print("\n" + "─" * 55)
    print("3. BACKEND")
    print("─" * 55)
    backend = get_backend(
        use_simulator=not use_ibm,
        noise_model=noise_model,
        ibm_backend_name=q_cfg["ibm_backend"],
        token=os.getenv("IBM_QUANTUM_TOKEN"),
    )
    print_backend_info(backend)

    # ── 5. QAOA Optimization ──────────────────────────────────────
    print("\n" + "─" * 55)
    print("4. QAOA OPTIMIZATION")
    print("─" * 55)
    print(f"  depth={depth}, optimizer={optimizer}, maxiter={max_iter}, shots={shots}")

    opt_result = optimize_qaoa(
        qubo=qubo,
        backend=backend,
        depth=depth,
        optimizer_name=optimizer,
        max_iterations=max_iter,
        shots=shots,
    )
    print(opt_result)

    # ── 6. Sampling Analysis ──────────────────────────────────────
    print("\n" + "─" * 55)
    print("5. SAMPLING ANALYSIS (final circuit)")
    print("─" * 55)
    from qiskit import transpile
    qc = build_qaoa_circuit(qubo.Q, depth=depth)
    qc_bound = bind_parameters(qc, opt_result.optimal_params, depth)
    qc_t = transpile(qc_bound, backend)
    job = backend.run(qc_t, shots=shots * 4)
    final_counts = job.result().get_counts()
    sampling = parse_counts(final_counts, qubo.Q, qubo.budget, top_k=10)
    print_sampling_summary(sampling, qubo)

    w_qaoa = aggregate_weights_from_distribution(
        sampling, qubo.tickers, mu=mu, cov=cov,
        weight_method="risk_weighted", top_k=5,
    )

    # ── 7. So sánh ────────────────────────────────────────────────
    print("\n" + "─" * 55)
    print("6. SO SÁNH: QUANTUM vs CLASSICAL")
    print("─" * 55)
    portfolios = {
        "QAOA":         w_qaoa,
        "Equal Weight": pd.Series(np.ones(len(mu)) / len(mu), index=mu.index),
    }
    if w_bf is not None:
        portfolios["Brute-Force"] = w_bf

    common = [t for t in mu.index if t in returns.columns]
    comparison = compare_portfolios(portfolios, returns[common], risk_free_rate=rf)
    print(comparison.round(4).to_string())

    # ── 8. Lưu kết quả ───────────────────────────────────────────
    proc_dir = Path("data/processed")
    w_qaoa.to_csv(proc_dir / "quantum_qaoa_weights.csv", header=["weight"])
    opt_result.weights.to_csv(proc_dir / "quantum_best_bitstring_weights.csv", header=["weight"])

    summary = {
        "best_objective": opt_result.best_objective,
        "feasible":       opt_result.feasible,
        "n_iterations":   opt_result.n_iterations,
        "elapsed_s":      round(opt_result.elapsed_seconds, 2),
        "backend":        opt_result.backend_name,
        "shots":          opt_result.shots,
        "depth":          depth,
        "budget":         budget,
        "n_assets":       len(mu),
        "tickers":        list(mu.index),
        "bitstring":      "".join(map(str, opt_result.best_bitstring.astype(int))),
    }
    with open(proc_dir / "quantum_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Giai đoạn 3 hoàn thành. Kết quả lưu tại {proc_dir}/")
    return opt_result, w_qaoa


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Quantum Portfolio Optimizer — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=["data", "classical", "quantum", "all"],
        default="data",
        help="Giai đoạn cần chạy (mặc định: data)",
    )
    parser.add_argument(
        "--tickers", nargs="+",
        help="Danh sách mã cổ phiếu, VD: AAPL MSFT GOOGL",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Đường dẫn config.yaml (mặc định: config.yaml)",
    )
    parser.add_argument(
        "--force-download", action="store_true",
        help="Bỏ qua cache, tải lại từ Yahoo Finance",
    )
    parser.add_argument(
        "--use-ibm", action="store_true",
        help="Dùng IBM Quantum hardware thật (cần IBM_QUANTUM_TOKEN)",
    )
    parser.add_argument(
        "--noise-model", action="store_true",
        help="Inject noise model IBM Brisbane vào AerSimulator",
    )

    args = parser.parse_args()

    try:
        portfolio_data = None

        if args.stage in ("data", "all"):
            portfolio_data = run_data_stage(args)

        if args.stage in ("classical", "all"):
            run_classical_stage(args, portfolio_data=portfolio_data)

        if args.stage in ("quantum", "all"):
            run_quantum_stage(args, portfolio_data=portfolio_data)

    except Exception as e:
        logger.error(f"Lỗi: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()