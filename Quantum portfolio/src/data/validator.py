"""
src/data/validator.py
---------------------
Kiểm tra chất lượng dữ liệu trước khi đưa vào solver.
Trả về báo cáo chi tiết và raise lỗi nếu dữ liệu không đạt yêu cầu tối thiểu.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """
    Kết quả kiểm tra dữ liệu.

    Attributes
    ----------
    passed : bool
        True nếu toàn bộ kiểm tra bắt buộc đều qua.
    errors : list[str]
        Danh sách lỗi nghiêm trọng (làm solver fail).
    warnings : list[str]
        Danh sách cảnh báo (solver vẫn chạy được nhưng nên chú ý).
    stats : dict
        Thống kê tóm tắt về dữ liệu.
    """

    passed: bool = True
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def __str__(self) -> str:
        lines = ["=" * 55, "BÁO CÁO KIỂM TRA DỮ LIỆU", "=" * 55]
        lines.append(f"Kết quả: {'✓ PASSED' if self.passed else '✗ FAILED'}")

        if self.stats:
            lines.append("\nThống kê:")
            for k, v in self.stats.items():
                lines.append(f"  {k}: {v}")

        if self.errors:
            lines.append("\nLỗi (bắt buộc phải sửa):")
            for e in self.errors:
                lines.append(f"  ✗ {e}")

        if self.warnings:
            lines.append("\nCảnh báo:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")

        return "\n".join(lines)


def validate_prices(
    prices: pd.DataFrame,
    min_rows: int = 100,
    min_tickers: int = 2,
    max_missing_pct: float = 0.05,
) -> ValidationReport:
    """
    Kiểm tra DataFrame giá gốc.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame giá (ngày × ticker).
    min_rows : int
        Số ngày tối thiểu yêu cầu.
    min_tickers : int
        Số cổ phiếu tối thiểu yêu cầu.
    max_missing_pct : float
        Tỷ lệ NaN tối đa cho phép mỗi cột.

    Returns
    -------
    ValidationReport
    """
    report = ValidationReport()

    # --- Thống kê cơ bản ---
    report.stats = {
        "Số ngày": len(prices),
        "Số cổ phiếu": len(prices.columns),
        "Từ ngày": str(prices.index.min().date()) if len(prices) > 0 else "N/A",
        "Đến ngày": str(prices.index.max().date()) if len(prices) > 0 else "N/A",
        "% NaN toàn bảng": f"{prices.isna().mean().mean():.2%}",
    }

    # --- Kiểm tra kích thước ---
    if len(prices) < min_rows:
        report.errors.append(
            f"Chỉ có {len(prices)} ngày, cần ít nhất {min_rows}."
        )
        report.passed = False

    if len(prices.columns) < min_tickers:
        report.errors.append(
            f"Chỉ có {len(prices.columns)} cổ phiếu, cần ít nhất {min_tickers}."
        )
        report.passed = False

    # --- Kiểm tra giá trị âm ---
    negative_mask = (prices < 0).any()
    bad_tickers = negative_mask[negative_mask].index.tolist()
    if bad_tickers:
        report.errors.append(f"Giá âm ở cột: {bad_tickers}")
        report.passed = False

    # --- Kiểm tra missing ---
    missing_pct = prices.isna().mean()
    high_missing = missing_pct[missing_pct > max_missing_pct].to_dict()
    if high_missing:
        for t, pct in high_missing.items():
            report.warnings.append(f"{t}: {pct:.1%} dữ liệu thiếu")

    # --- Kiểm tra giá trị bất thường (daily return > 50%) ---
    simple_returns = prices.pct_change().iloc[1:]
    extreme = (simple_returns.abs() > 0.5).any()
    extreme_tickers = extreme[extreme].index.tolist()
    if extreme_tickers:
        report.warnings.append(
            f"Biến động bất thường (>50%/ngày) ở: {extreme_tickers}"
        )

    # --- Kiểm tra index là DatetimeIndex ---
    if not isinstance(prices.index, pd.DatetimeIndex):
        report.errors.append("Index của prices phải là DatetimeIndex.")
        report.passed = False

    # --- Kiểm tra thứ tự thời gian ---
    if not prices.index.is_monotonic_increasing:
        report.warnings.append("Index không theo thứ tự thời gian tăng dần.")

    return report


def validate_portfolio_data(
    mu: pd.Series,
    cov: pd.DataFrame,
    min_assets: int = 2,
) -> ValidationReport:
    """
    Kiểm tra μ và Σ trước khi đưa vào solver.

    Parameters
    ----------
    mu : pd.Series
        Expected return hàng năm.
    cov : pd.DataFrame
        Covariance matrix hàng năm.
    min_assets : int
        Số tài sản tối thiểu.

    Returns
    -------
    ValidationReport
    """
    report = ValidationReport()

    n = len(mu)
    report.stats = {
        "Số tài sản": n,
        "μ min": f"{mu.min():.2%}",
        "μ max": f"{mu.max():.2%}",
        "Volatility min": f"{np.sqrt(np.diag(cov.values)).min():.2%}",
        "Volatility max": f"{np.sqrt(np.diag(cov.values)).max():.2%}",
    }

    # --- Kích thước ---
    if n < min_assets:
        report.errors.append(f"Cần ít nhất {min_assets} tài sản, có {n}.")
        report.passed = False

    # --- mu và cov phải cùng tickers ---
    if not mu.index.equals(cov.index) or not mu.index.equals(cov.columns):
        report.errors.append("mu và cov phải có cùng index/columns.")
        report.passed = False

    # --- NaN trong mu / cov ---
    if mu.isna().any():
        bad = mu[mu.isna()].index.tolist()
        report.errors.append(f"NaN trong μ: {bad}")
        report.passed = False

    if cov.isna().any().any():
        report.errors.append("NaN trong covariance matrix.")
        report.passed = False

    # --- Symmetric ---
    cov_arr = cov.values
    if not np.allclose(cov_arr, cov_arr.T, atol=1e-8):
        report.errors.append("Covariance matrix không symmetric.")
        report.passed = False

    # --- Positive semi-definite ---
    try:
        eigenvalues = np.linalg.eigvalsh(cov_arr)
        min_eig = eigenvalues.min()
        if min_eig < -1e-6:
            report.errors.append(
                f"Covariance matrix không PSD (eigenvalue nhỏ nhất = {min_eig:.2e})."
            )
            report.passed = False
        elif min_eig < 0:
            report.warnings.append(
                f"Eigenvalue âm nhỏ ({min_eig:.2e}) — có thể do lỗi số học, đã xử lý."
            )
    except np.linalg.LinAlgError:
        report.errors.append("Không thể tính eigenvalue của covariance matrix.")
        report.passed = False

    # --- Condition number (cảnh báo nếu ill-conditioned) ---
    try:
        cond = np.linalg.cond(cov_arr)
        report.stats["Condition number"] = f"{cond:.2e}"
        if cond > 1e10:
            report.warnings.append(
                f"Covariance matrix ill-conditioned (cond={cond:.2e}). "
                "Xem xét giảm số tài sản hoặc dùng regularization."
            )
    except Exception:
        pass

    return report


def run_all_checks(
    prices: pd.DataFrame,
    mu: pd.Series,
    cov: pd.DataFrame,
    raise_on_error: bool = True,
) -> tuple[ValidationReport, ValidationReport]:
    """
    Chạy toàn bộ kiểm tra và in báo cáo.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame giá gốc.
    mu : pd.Series
        Expected return.
    cov : pd.DataFrame
        Covariance matrix.
    raise_on_error : bool
        Nếu True, raise ValueError khi có lỗi nghiêm trọng.

    Returns
    -------
    tuple[ValidationReport, ValidationReport]
        (price_report, portfolio_report)
    """
    price_report = validate_prices(prices)
    portfolio_report = validate_portfolio_data(mu, cov)

    logger.info("\n" + str(price_report))
    logger.info("\n" + str(portfolio_report))

    if raise_on_error:
        all_errors = price_report.errors + portfolio_report.errors
        if all_errors:
            raise ValueError(
                "Dữ liệu không hợp lệ:\n" + "\n".join(f"  - {e}" for e in all_errors)
            )

    return price_report, portfolio_report