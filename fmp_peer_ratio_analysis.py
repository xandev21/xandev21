#!/usr/bin/env python3
"""Peer ratio analysis using Financial Modeling Prep (FMP).

This script accepts a list of U.S. tickers and, for each ticker:
1. Fetches its FMP industry classification.
2. Builds an industry peer group from FMP stock screener data.
3. Calculates these ratios for the company and each peer:
   - EBITDA margin = EBITDA / Revenue
   - Debt/Equity = Total Debt / Total Stockholders' Equity
   - Cash/Equity = Cash and Cash Equivalents / Total Stockholders' Equity
4. Computes peer-group summary stats (mean + standard deviation).
5. Flags statistical significance when |z-score| > 2.
6. Prints a per-ticker table and saves all results to CSV.

Usage examples:
    python fmp_peer_ratio_analysis.py --tickers AAPL,MSFT,NVDA --api-key <FMP_KEY>
    python fmp_peer_ratio_analysis.py --input-csv tickers.csv --ticker-column Ticker \
        --output-csv peer_ratios.csv

Notes:
- You can pass the API key by --api-key or FMP_API_KEY environment variable.
- Script focuses on U.S. exchanges (NASDAQ, NYSE, AMEX) for peers.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
import requests

BASE_URL = "https://financialmodelingprep.com/api/v3"
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 1.0
SIGNIFICANCE_Z_THRESHOLD = 2.0

RATIO_COLUMNS = [
    "ebitda_margin",
    "debt_to_equity",
    "cash_to_equity",
]

RATIO_LABELS = {
    "ebitda_margin": "EBITDA Margin",
    "debt_to_equity": "Debt/Equity",
    "cash_to_equity": "Cash/Equity",
}


class FMPClient:
    """Minimal client for selected FMP endpoints with retry/error handling."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.session = requests.Session()

    def get_json(self, endpoint: str, params: Optional[Dict[str, str]] = None) -> Optional[object]:
        """Return parsed JSON for endpoint or None if the request fails."""
        url = f"{BASE_URL}/{endpoint}"
        query = {"apikey": self.api_key}
        if params:
            query.update(params)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.session.get(url, params=query, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                print(
                    f"[WARN] Request failed ({endpoint}, attempt {attempt}/{MAX_RETRIES}): {exc}",
                    file=sys.stderr,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_SLEEP_SECONDS)

        print(f"[ERROR] Giving up on endpoint: {endpoint}", file=sys.stderr)
        return None

    def get_profile(self, ticker: str) -> Optional[Dict[str, object]]:
        data = self.get_json(f"profile/{ticker}")
        if isinstance(data, list) and data:
            return data[0]
        return None

    def get_income_ttm(self, ticker: str) -> Optional[Dict[str, object]]:
        data = self.get_json(f"income-statement-ttm/{ticker}")
        if isinstance(data, list) and data:
            return data[0]
        return None

    def get_balance_ttm(self, ticker: str) -> Optional[Dict[str, object]]:
        data = self.get_json(f"balance-sheet-statement-ttm/{ticker}")
        if isinstance(data, list) and data:
            return data[0]
        return None

    def get_industry_peers(self, industry: str, limit: int = 200) -> List[str]:
        """Return peer symbols for a given industry from stock screener."""
        data = self.get_json(
            "stock-screener",
            params={
                "industry": industry,
                "exchange": "NASDAQ,NYSE,AMEX",
                "limit": str(limit),
            },
        )
        if not isinstance(data, list):
            return []

        symbols: List[str] = []
        for row in data:
            if isinstance(row, dict):
                symbol = row.get("symbol")
                if isinstance(symbol, str) and symbol:
                    symbols.append(symbol.upper())
        return list(dict.fromkeys(symbols))


def safe_float(value: object) -> Optional[float]:
    """Convert value to float safely, returning None for invalid inputs."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_ratios(income_ttm: Dict[str, object], balance_ttm: Dict[str, object]) -> Dict[str, Optional[float]]:
    """Compute requested ratios from TTM income and balance-sheet payloads."""
    revenue = safe_float(income_ttm.get("revenue"))
    ebitda = safe_float(income_ttm.get("ebitda"))
    total_debt = safe_float(balance_ttm.get("totalDebt"))
    equity = safe_float(balance_ttm.get("totalStockholdersEquity"))
    cash = safe_float(balance_ttm.get("cashAndCashEquivalents"))

    ebitda_margin = (ebitda / revenue) if (ebitda is not None and revenue not in (None, 0.0)) else None
    debt_to_equity = (total_debt / equity) if (total_debt is not None and equity not in (None, 0.0)) else None
    cash_to_equity = (cash / equity) if (cash is not None and equity not in (None, 0.0)) else None

    return {
        "ebitda_margin": ebitda_margin,
        "debt_to_equity": debt_to_equity,
        "cash_to_equity": cash_to_equity,
    }


def load_tickers(input_csv: Optional[str], ticker_column: str, tickers_arg: Optional[str]) -> List[str]:
    """Load and normalize tickers from CSV and/or CLI string."""
    tickers: List[str] = []

    if input_csv:
        try:
            df = pd.read_csv(input_csv)
        except Exception as exc:
            raise ValueError(f"Failed reading CSV '{input_csv}': {exc}") from exc

        if ticker_column not in df.columns:
            raise ValueError(f"Ticker column '{ticker_column}' not found in {input_csv}")

        csv_tickers = [str(v).strip().upper() for v in df[ticker_column].dropna().tolist() if str(v).strip()]
        tickers.extend(csv_tickers)

    if tickers_arg:
        cli_tickers = [token.strip().upper() for token in tickers_arg.split(",") if token.strip()]
        tickers.extend(cli_tickers)

    unique = list(dict.fromkeys(tickers))
    if not unique:
        raise ValueError("No tickers provided. Use --tickers and/or --input-csv.")

    return unique


def analyze_tickers(client: FMPClient, tickers: Sequence[str], peer_limit: int) -> pd.DataFrame:
    """Run peer ratio analysis and return long-form result table."""
    rows: List[Dict[str, object]] = []
    ratio_cache: Dict[str, Dict[str, Optional[float]]] = {}
    industry_peers_cache: Dict[str, List[str]] = {}

    def get_ratios_for_symbol(symbol: str) -> Optional[Dict[str, Optional[float]]]:
        if symbol in ratio_cache:
            return ratio_cache[symbol]

        income = client.get_income_ttm(symbol)
        balance = client.get_balance_ttm(symbol)
        if not income or not balance:
            ratio_cache[symbol] = {}
            return None

        ratios = compute_ratios(income, balance)
        ratio_cache[symbol] = ratios
        return ratios

    for ticker in tickers:
        profile = client.get_profile(ticker)
        if not profile:
            print(f"[WARN] Missing profile data for {ticker}; skipping.", file=sys.stderr)
            continue

        industry = profile.get("industry")
        if not isinstance(industry, str) or not industry.strip():
            print(f"[WARN] Missing industry for {ticker}; skipping.", file=sys.stderr)
            continue

        industry = industry.strip()
        if industry not in industry_peers_cache:
            industry_peers_cache[industry] = client.get_industry_peers(industry, limit=peer_limit)

        peer_symbols = [s for s in industry_peers_cache[industry] if s != ticker]
        company_ratios = get_ratios_for_symbol(ticker)
        if not company_ratios:
            print(f"[WARN] Missing financial statements for {ticker}; skipping.", file=sys.stderr)
            continue

        peer_ratio_records: List[Dict[str, Optional[float]]] = []
        for peer in peer_symbols:
            peer_ratios = get_ratios_for_symbol(peer)
            if peer_ratios:
                peer_ratio_records.append(peer_ratios)

        peer_df = pd.DataFrame(peer_ratio_records)

        for ratio_key in RATIO_COLUMNS:
            company_value = company_ratios.get(ratio_key)

            if ratio_key in peer_df.columns:
                peer_series = pd.to_numeric(peer_df[ratio_key], errors="coerce").dropna()
            else:
                peer_series = pd.Series(dtype="float64")

            peer_avg = float(peer_series.mean()) if not peer_series.empty else None
            peer_std = float(peer_series.std(ddof=0)) if not peer_series.empty else None
            n_peers = int(peer_series.shape[0])

            z_score: Optional[float] = None
            significance = "NO"
            if company_value is not None and peer_avg is not None and peer_std not in (None, 0.0):
                z_score = (company_value - peer_avg) / peer_std
                if abs(z_score) > SIGNIFICANCE_Z_THRESHOLD:
                    significance = "YES"

            rows.append(
                {
                    "Ticker": ticker,
                    "Industry": industry,
                    "Ratio": RATIO_LABELS[ratio_key],
                    "Company X": company_value,
                    "Peer Avg": peer_avg,
                    "Std Dev": peer_std,
                    "N Peers": n_peers,
                    "Z-Score": z_score,
                    "Significance": significance,
                }
            )

    return pd.DataFrame(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FMP-based peer ratio significance analysis.")
    parser.add_argument("--tickers", help="Comma-separated list of U.S. tickers, e.g. AAPL,MSFT,NVDA")
    parser.add_argument("--input-csv", help="Optional input CSV containing ticker symbols")
    parser.add_argument(
        "--ticker-column",
        default="Ticker",
        help="Column name in input CSV that contains ticker symbols (default: Ticker)",
    )
    parser.add_argument(
        "--output-csv",
        default="peer_ratio_analysis.csv",
        help="Output CSV path (default: peer_ratio_analysis.csv)",
    )
    parser.add_argument(
        "--peer-limit",
        type=int,
        default=200,
        help="Max companies to pull per industry peer set via stock screener (default: 200)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("FMP_API_KEY"),
        help="FMP API key (or set FMP_API_KEY environment variable)",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.api_key:
        print("[ERROR] Missing FMP API key. Use --api-key or set FMP_API_KEY.", file=sys.stderr)
        return 2

    try:
        tickers = load_tickers(args.input_csv, args.ticker_column, args.tickers)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    client = FMPClient(args.api_key)
    result_df = analyze_tickers(client, tickers=tickers, peer_limit=args.peer_limit)

    if result_df.empty:
        print("[WARN] No analysis rows produced. Check tickers/API responses.", file=sys.stderr)
        return 1

    numeric_cols = ["Company X", "Peer Avg", "Std Dev", "Z-Score"]
    for col in numeric_cols:
        result_df[col] = pd.to_numeric(result_df[col], errors="coerce")

    print("\n=== Peer Ratio Analysis ===")
    for ticker in result_df["Ticker"].drop_duplicates().tolist():
        ticker_df = result_df[result_df["Ticker"] == ticker][
            ["Ticker", "Ratio", "Company X", "Peer Avg", "Std Dev", "Significance"]
        ]
        print(f"\nTicker: {ticker}")
        print(ticker_df.to_string(index=False))

    result_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved full results to: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
