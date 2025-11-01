"""Simple EURUSD backtesting scaffold using OHLC data.

This module shows how to:

- load OHLCV data from CSV (or any pandas-compatible source)
- generate trading signals from a moving-average crossover strategy
- simulate PnL with basic FX-specific frictions (per-side trading cost)
- compute headline performance metrics (CAGR, Sharpe, Max DD)

The goal is to give you a starting point that you can extend with more
advanced position sizing, risk management, or multi-asset support.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def load_ohlc(csv_path: str | Path) -> pd.DataFrame:
    """Load OHLCV data from ``csv_path`` into a cleaned pandas DataFrame.

    The loader expects a column named ``time`` (case insensitive) that can be
    parsed into timestamps. Remaining columns should include the usual
    ``open``, ``high``, ``low``, ``close`` (any order or casing). Volume is
    optional but kept when present.

    Returns a DataFrame indexed by timestamp, sorted in ascending order, with
    at least a ``close`` column.
    """

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find OHLC data at {path!s}")

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=False, errors="coerce")
        if df["time"].isna().any():
            raise ValueError("Some timestamp values could not be parsed; check your CSV.")
        df = df.sort_values("time").set_index("time")
    else:
        df = df.sort_index()

    required_cols = {"open", "high", "low", "close"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLC columns: {sorted(missing)}")

    # Keep a clean column order: OHLC plus any extras (e.g., volume)
    ordered_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    extra_cols = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + extra_cols]

    return df


class Strategy:
    """Base class for trading strategies."""

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


@dataclass
class MovingAverageCrossStrategy(Strategy):
    """Simple fast/slow moving-average crossover strategy."""

    fast_window: int = 10
    slow_window: int = 30

    def __post_init__(self) -> None:
        if self.fast_window <= 0 or self.slow_window <= 0:
            raise ValueError("Moving-average windows must be positive integers.")
        if self.fast_window >= self.slow_window:
            raise ValueError("'fast_window' must be strictly less than 'slow_window'.")

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["close"].astype(float)
        signals = pd.DataFrame(index=data.index)
        signals["fast_ma"] = close.rolling(self.fast_window, min_periods=1).mean()
        signals["slow_ma"] = close.rolling(self.slow_window, min_periods=1).mean()

        # Long when fast MA is above slow MA, short when below.
        signals["raw_position"] = 0.0
        signals.loc[signals["fast_ma"] > signals["slow_ma"], "raw_position"] = 1.0
        signals.loc[signals["fast_ma"] < signals["slow_ma"], "raw_position"] = -1.0
        signals["position"] = signals["raw_position"].ffill().fillna(0.0)

        return signals[["position", "fast_ma", "slow_ma"]]


@dataclass
class BacktestResult:
    """Container for the outcomes of a backtest run."""

    signals: pd.DataFrame
    performance: pd.DataFrame
    metrics: Dict[str, float]


class Backtester:
    """Single-asset, long/short backtester for EURUSD using OHLC data."""

    def __init__(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        trading_cost: float = 0.00002,
        initial_capital: float = 10000.0,
    ) -> None:
        if "close" not in data.columns:
            raise ValueError("Input data must contain a 'close' column.")

        self.data = data.copy()
        self.strategy = strategy
        self.trading_cost = float(trading_cost)
        self.initial_capital = float(initial_capital)

    def run(self) -> BacktestResult:
        signals = self.strategy.generate_signals(self.data)
        aligned = self.data.join(signals, how="left")
        aligned["position"] = aligned["position"].ffill().fillna(0.0)

        perf = self._simulate(aligned)
        metrics = self._compute_metrics(perf)

        return BacktestResult(signals=signals, performance=perf, metrics=metrics)

    def _simulate(self, df: pd.DataFrame) -> pd.DataFrame:
        perf = df.copy()
        perf["log_return"] = np.log(perf["close"]).diff()
        perf["strategy_return"] = perf["position"].shift(1).fillna(0.0) * perf["log_return"].fillna(0.0)

        # Apply per-turnover trading costs (both entry and exit count)
        turnover = perf["position"].diff().abs().fillna(0.0)
        perf["strategy_return"] -= turnover * self.trading_cost

        perf["equity_curve"] = (perf["strategy_return"].fillna(0.0).add(1.0)).cumprod() * self.initial_capital
        perf["drawdown"] = perf["equity_curve"] / perf["equity_curve"].cummax() - 1.0

        return perf

    def _compute_metrics(self, perf: pd.DataFrame) -> Dict[str, float]:
        returns = perf["strategy_return"].dropna()
        if returns.empty:
            return {
                "total_return": 0.0,
                "cagr": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "volatility": 0.0,
            }

        periods_per_year = self._infer_periods_per_year(perf.index)
        total_return = perf["equity_curve"].iloc[-1] / self.initial_capital - 1.0
        mean_return = returns.mean()
        vol = returns.std(ddof=0)

        cagr = (1.0 + total_return) ** (periods_per_year / max(len(returns), 1)) - 1.0
        sharpe = (np.sqrt(periods_per_year) * mean_return / vol) if vol > 0 else np.nan
        max_drawdown = perf["drawdown"].min()

        return {
            "total_return": float(total_return),
            "cagr": float(cagr),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "volatility": float(vol * np.sqrt(periods_per_year)),
        }

    @staticmethod
    def _infer_periods_per_year(index: pd.Index) -> int:
        if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
            return 252

        median_step = index.to_series().diff().dropna().median()
        if median_step <= pd.Timedelta(minutes=1):
            return 365 * 24 * 60
        if median_step <= pd.Timedelta(hours=1):
            return 365 * 24
        if median_step <= pd.Timedelta(days=1):
            return 252
        if median_step <= pd.Timedelta(weeks=1):
            return 52
        return 12


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple EURUSD backtest.")
    parser.add_argument(
        "--csv",
        default="data/example_eurusd.csv",
        help="Path to OHLC CSV file with a 'time' column.",
    )
    parser.add_argument(
        "--fast",
        type=int,
        default=10,
        help="Fast moving-average window length (in bars).",
    )
    parser.add_argument(
        "--slow",
        type=int,
        default=30,
        help="Slow moving-average window length (in bars).",
    )
    parser.add_argument(
        "--cost",
        type=float,
        default=0.00002,
        help=(
            "Per-unit transaction cost applied on each position change.\n"
            "Example: 0.0001 ~ 1 pip; default 0.00002 approximates 0.2 pip per turn."
        ),
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial notional capital used to scale the equity curve.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    data = load_ohlc(args.csv)
    strategy = MovingAverageCrossStrategy(fast_window=args.fast, slow_window=args.slow)
    backtester = Backtester(
        data=data,
        strategy=strategy,
        trading_cost=args.cost,
        initial_capital=args.capital,
    )

    result = backtester.run()

    print("Summary metrics:")
    for key, value in result.metrics.items():
        print(f"  {key:>15}: {value: .4f}")

    print("\nLast 5 rows of performance data:")
    print(
        result.performance[
            ["close", "position", "strategy_return", "equity_curve", "drawdown"]
        ].tail()
    )


if __name__ == "__main__":
    main()
