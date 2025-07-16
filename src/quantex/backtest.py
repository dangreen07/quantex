"""Backtest orchestration utilities.

This module exposes:
    * ``BacktestResult`` – a dataclass aggregating NAV, orders, fills and metrics.
    * ``BacktestRunner`` – a convenience wrapper that wires together a Strategy,
      one or more DataSource objects, an execution simulator and the internal
      EventBus. End-users typically instantiate ``BacktestRunner`` once per
      test and call `run()` to obtain a `BacktestResult`.

"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Mapping, Any

import numpy as np
import pandas as pd

from quantex.engine import EventBus
from quantex.execution import ImmediateFillSimulator, NextBarSimulator
from quantex.sources import BacktestingDataSource
from quantex.strategy import Strategy
from quantex.models import Order, Fill


class Metrics(dict):
    """Lightweight container that formats metrics nicely when printed."""

    def __str__(self) -> str:  # noqa: DunderAll – explicit override
        def _fmt(val: Any) -> str:
            # Uniform float formatting to 4 decimals while keeping ints/others intact
            if isinstance(val, float):
                return f"{val:.4f}"
            return str(val)

        return "\n".join(f"{k:<20}: {_fmt(v)}" for k, v in sorted(self.items()))

    __repr__ = __str__


@dataclass
class BacktestResult:
    """Contains the results of a backtest.

    Attributes:
        nav: A pandas Series representing the Net Asset Value (NAV) over time.
        orders: A list of all orders generated during the backtest.
        fills: A list of all fills executed during the backtest.
        metrics: A dictionary of performance metrics.
    """

    nav: pd.Series
    orders: list[Order]
    fills: list[Fill]
    metrics: Metrics


# ---------------------------------------------------------------------------
# Backtest Runner
# ---------------------------------------------------------------------------


class BacktestRunner:
    """User-facing helper that wires Strategy, EventBus, and Simulator."""

    def __init__(
        self,
        strategy: Strategy,
        data_sources: Mapping[str, BacktestingDataSource],
        risk_free_rate: float = 0.0,
        min_holding_period: pd.Timedelta | None = None,
        simulator: ImmediateFillSimulator | NextBarSimulator | None = None,
    ):
        """Initializes the BacktestRunner.

        Args:
            strategy: The trading strategy to be backtested.
            data_sources: A dictionary of data sources for the backtest.
            risk_free_rate: The risk-free rate to use for the Sharpe ratio.
            periods_per_year: The number of periods per year to use for the Sharpe ratio.
            min_holding_period: Optional minimum holding period for positions.
            simulator: Execution simulator to use. Defaults to NextBarSimulator
                for more realistic order execution timing.
        """
        self.strategy = strategy
        self.data_sources = data_sources
        if simulator is None:
            self.simulator = NextBarSimulator(
                self.strategy.portfolio,
                min_holding_period=min_holding_period,
            )
        else:
            self.simulator = simulator
        self.event_bus = EventBus(strategy, data_sources, self.simulator)
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = BacktestRunner._find_periods_per_year(data_sources)

    @staticmethod
    def _find_periods_per_year(
        data_sources: Mapping[str, BacktestingDataSource],
    ) -> int:
        """Infer *periods_per_year* for annualising metrics.

        The function first tries to find the modal time-delta using timestamps
        common to **all** data sources. If fewer than two common timestamps
        exist (feeds with disjoint calendars or heavy gaps), it falls back to
        computing the modal bar size **per source** and subsequently picks the
        *smallest* step – i.e. the highest frequency – across the set. This
        approach supports back-tests where assets are intentionally
        mis-aligned while preserving compatibility with single-source
        scenarios.
        """
        # 1. Intersection of all indices
        intersection: pd.Index | None = None
        for ds in data_sources.values():
            idx = ds.get_raw_data().index
            intersection = (
                idx if intersection is None else intersection.intersection(idx)
            )

        if intersection is None or len(intersection) < 2:
            # Fallback – infer the modal bar size from individual data sources
            # rather than their intersection. This supports scenarios where the
            # feeds are intentionally mis-aligned (e.g. assets trading on
            # slightly different calendars or with missing bars).

            step_candidates: list[float] = []
            for ds in data_sources.values():
                idx_ds = pd.to_datetime(ds.get_raw_data().index).sort_values()
                if len(idx_ds) < 2:
                    # Cannot infer a step from fewer than two timestamps
                    continue

                deltas_ds = idx_ds.to_series().diff().dropna().dt.total_seconds()
                if not deltas_ds.empty:
                    # Use the mode (most frequent) delta for this datasource
                    step_candidates.append(float(deltas_ds.mode().iat[0]))

            if not step_candidates:
                raise ValueError(
                    "Unable to infer bar frequency – insufficient timestamp data across sources"
                )

            # Choose the smallest (highest-frequency) step among all candidates
            step_sec = min(step_candidates)
        else:
            # 2. Consecutive deltas in seconds – use the common intersection
            idx = pd.to_datetime(intersection).sort_values()
            deltas_sec = idx.to_series().diff().dropna().dt.total_seconds()

            # 3. Typical step = mode of the delta distribution
            step_sec = deltas_sec.mode().iat[0]

        if step_sec == 0:
            raise ValueError("Zero-length step encountered")

        # 4. Convert to periods per year
        if step_sec == 60:
            periods_per_year = 252 * 6.5 * 60
        elif step_sec == 60 * 60:
            periods_per_year = 252 * 6.5
        elif step_sec == 60 * 60 * 24:
            periods_per_year = 252
        else:
            # Generic conversion for uncommon frequencies – fall back to
            # seconds-in-year divided by step size.
            seconds_in_year = 365 * 24 * 60 * 60
            periods_per_year = seconds_in_year / step_sec

        return math.ceil(periods_per_year)

    def run(self, *, metrics_style: str = "default") -> BacktestResult:
        """Runs the back-test.

        Args:
            metrics_style: Controls how certain metrics are computed. Accepted
                values:

                * ``"default"`` – sample‐stdev Sharpe (ddof=1) and other
                  academic conventions (the library default).
                * ``"bt"`` – compatibility mode mirroring the `backtesting.py`
                  package: population st-dev (ddof=0) and risk-free rate 0.0.

        Returns:
            A :class:`BacktestResult` instance containing NAV, orders, fills
            and metrics.
        """
        self.event_bus.run()
        nav_series = pd.Series(
            self.event_bus.nav, index=self.event_bus.timestamps, name="NAV"
        )

        metrics: dict = {}

        if not nav_series.empty and nav_series.iloc[0] != 0:
            metrics["total_return"] = nav_series.iloc[-1] / nav_series.iloc[0] - 1

            # Maximum drawdown ---------------------------------------------
            metrics["max_drawdown"] = _max_drawdown(nav_series)

            # Sharpe ratio -------------------------------------------------
            if len(nav_series) > 1:
                # Population vs sample st-dev depending on selected style
                ddof = 0 if metrics_style == "bt" else 1
                metrics["sharpe_ratio"] = _annualised_sharpe(
                    nav_series, self.risk_free_rate, self.periods_per_year, ddof
                )

            # --- Additional metrics ------------------------------------
            periods = len(nav_series)
            if periods > 1:
                # Annualised (geometric) return – CAGR
                #
                # Very short back-tests (few bars) can yield a huge exponent
                # *(periods_per_year / periods)* which, when combined with a
                # ratio slightly >1, overflows the double-precision range and
                # emits a RuntimeWarning. We compute the power inside a NumPy
                # *errstate* context that suppresses the warning while still
                # returning ``inf`` for pathological cases.
                with np.errstate(over="ignore", invalid="ignore"):
                    cagr_val = (
                        np.power(
                            nav_series.iloc[-1] / nav_series.iloc[0],
                            self.periods_per_year / periods,
                        )
                        - 1.0
                    )

                # Cast to Python float for stable downstream formatting
                metrics["cagr"] = float(cagr_val)

                returns = nav_series.pct_change().dropna()
                # Annualised volatility – align ddof with Sharpe selection
                metrics["volatility_annualised"] = returns.std(ddof=ddof) * np.sqrt(
                    self.periods_per_year
                )

                # Sortino ratio (downside risk only)
                downside = returns[returns < 0]
                if len(downside) > 0 and downside.std(ddof=0) != 0:
                    metrics["sortino_ratio"] = (
                        returns.mean() / downside.std(ddof=0)
                    ) * np.sqrt(self.periods_per_year)
                else:
                    metrics["sortino_ratio"] = float("nan")

                # Calmar ratio – CAGR divided by abs(max drawdown)
                max_dd = abs(metrics["max_drawdown"])
                metrics["calmar_ratio"] = (
                    metrics["cagr"] / max_dd if max_dd > 0 else float("nan")
                )

                # Buy & hold return of first symbol (if available)
                price_df = getattr(self.event_bus, "_price_df", None)
                if price_df is not None and not price_df.empty:
                    first_symbol = price_df.columns[0]
                    metrics["buy_hold_return"] = (
                        price_df[first_symbol].iloc[-1] / price_df[first_symbol].iloc[0]
                        - 1
                    )

        return BacktestResult(
            nav_series, self.event_bus.orders, self.event_bus.fills, Metrics(metrics)
        )


def _annualised_sharpe(
    nav: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 98_280,
    ddof: int = 1,
) -> float:
    """Compute the annualised Sharpe ratio for a NAV series.

    Args:
        nav: Series of portfolio values indexed by timestamp.
        risk_free_rate: Annual risk-free rate expressed as a decimal. Defaults to
            4.3 % (0.043).
        periods_per_year: Number of return observations in a year. For US
            equity market minutes this is ``252 * 390 = 98_280``. For assets
            trading 24/7 (e.g. crypto) use ``365 * 1_440``.
        ddof: Delta Degrees of Freedom for the standard deviation.

    Returns:
        Annualised Sharpe ratio. If the standard deviation of excess returns is
        zero the function returns ``np.nan``.
    """

    # Minute-to-minute returns
    returns = nav.pct_change().dropna()
    if returns.empty:
        return float("nan")

    # Per-period risk-free rate
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    excess = returns - rf_per_period

    # Sample standard deviation (ddof=1) matches common Sharpe implementations
    std = excess.std(ddof=ddof)
    if std == 0:
        return float("nan")

    return np.sqrt(periods_per_year) * excess.mean() / std


def _max_drawdown(nav: pd.Series) -> float:
    """Compute the maximum drawdown for a NAV series.

    Args:
        nav: Series of portfolio values indexed by timestamp.

    Returns:
        Maximum drawdown as a percentage (e.g., -0.15 for 15% drawdown).
        Returns 0.0 if the series is empty or has only one value.
    """
    if nav.empty or len(nav) <= 1:
        return 0.0

    # Calculate running maximum (peak values)
    running_max = nav.expanding().max()

    # Calculate drawdown as percentage from peak
    drawdown = (nav - running_max) / running_max

    # Return the maximum (most negative) drawdown
    return float(drawdown.min())
