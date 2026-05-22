"""
src/data/fetcher.py
-------------------
Thu thập dữ liệu giá cổ phiếu từ Yahoo Finance qua yfinance.
Hỗ trợ cache để tránh download lại mỗi lần chạy.
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
import yfinance as yf

logger = logging.getLogger(__name__)


def _load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding='utf-8') as f:
        return yaml.safe_load(f)


def _is_cache_valid(filepath: Path, max_age_days: int) -> bool:
    """Kiểm tra file cache còn hợp lệ không (chưa quá max_age_days)."""
    if not filepath.exists():
        return False
    mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
    return datetime.now() - mtime < timedelta(days=max_age_days)


def fetch_prices(
    tickers: Optional[list[str]] = None,
    period_years: Optional[int] = None,
    interval: Optional[str] = None,
    config_path: str = "config.yaml",
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Tải dữ liệu giá đóng cửa (Adjusted Close) cho danh sách cổ phiếu.

    Parameters
    ----------
    tickers : list[str], optional
        Danh sách mã cổ phiếu. Mặc định lấy từ config.yaml.
    period_years : int, optional
        Số năm lịch sử cần lấy. Mặc định từ config.yaml.
    interval : str, optional
        Độ phân giải dữ liệu ('1d', '1wk', '1mo'). Mặc định từ config.yaml.
    config_path : str
        Đường dẫn đến file config.yaml.
    force_download : bool
        Nếu True, bỏ qua cache và tải lại từ Yahoo Finance.

    Returns
    -------
    pd.DataFrame
        DataFrame với index là ngày, columns là mã cổ phiếu,
        values là giá Adjusted Close.

    Raises
    ------
    ValueError
        Nếu không tải được dữ liệu cho bất kỳ ticker nào.
    """
    cfg = _load_config(config_path)
    data_cfg = cfg["data"]

    tickers = tickers or data_cfg["default_tickers"]
    period_years = period_years or data_cfg["period_years"]
    interval = interval or data_cfg["interval"]
    cache_days = data_cfg["cache_days"]

    raw_dir = Path(data_cfg["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Tên file cache theo tập tickers + tham số
    ticker_hash = "_".join(sorted(tickers))[:50]
    cache_file = raw_dir / f"prices_{ticker_hash}_{period_years}y_{interval}.csv"

    # Trả về cache nếu còn hợp lệ
    if not force_download and _is_cache_valid(cache_file, cache_days):
        logger.info(f"Đọc từ cache: {cache_file}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        logger.info(f"Cache: {df.shape[0]} ngày × {df.shape[1]} cổ phiếu")
        return df

    # Tính khoảng thời gian
    end_date = datetime.today()
    start_date = end_date - timedelta(days=period_years * 365)

    logger.info(
        f"Đang tải {len(tickers)} cổ phiếu từ {start_date.date()} đến {end_date.date()} ..."
    )

    # Download từ Yahoo Finance
    raw = yf.download(
        tickers=tickers,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=True,   # dùng Adjusted Close
        progress=False,
        threads=True,
    )

    if raw.empty:
        raise ValueError("yfinance không trả về dữ liệu. Kiểm tra lại tickers hoặc kết nối mạng.")

    # Lấy cột 'Close' (đã adjusted khi auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Loại bỏ cột toàn NaN (ticker không hợp lệ)
    invalid = prices.columns[prices.isna().all()].tolist()
    if invalid:
        logger.warning(f"Loại bỏ ticker không có dữ liệu: {invalid}")
        prices = prices.drop(columns=invalid)

    if prices.empty:
        raise ValueError(f"Không có dữ liệu hợp lệ cho: {tickers}")

    # Lưu cache
    prices.to_csv(cache_file)
    logger.info(f"Đã lưu cache: {cache_file}")
    logger.info(f"Kết quả: {prices.shape[0]} ngày × {prices.shape[1]} cổ phiếu")

    return prices


def get_ticker_info(ticker: str) -> dict:
    """
    Lấy thông tin cơ bản của một cổ phiếu (tên công ty, sector, v.v.).

    Parameters
    ----------
    ticker : str
        Mã cổ phiếu, ví dụ 'AAPL'.

    Returns
    -------
    dict
        Từ điển chứa: longName, sector, industry, marketCap, currency.
    """
    try:
        info = yf.Ticker(ticker).info
        return {
            "ticker": ticker,
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", None),
            "currency": info.get("currency", "USD"),
        }
    except Exception as e:
        logger.warning(f"Không lấy được info cho {ticker}: {e}")
        return {"ticker": ticker, "name": ticker}