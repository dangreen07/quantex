from dataclasses import dataclass
import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from .broker import Order
from .strategy import Strategy
from .enums import CommissionType
import copy
import itertools
from typing import Callable, Any
from tqdm import tqdm
import concurrent.futures
import pickle
import os
import gc

def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()
    return float(abs(max_dd))  # return as positive percentage

def _infer_periods_per_year(index: pd.Index, default: int = 252 * 24 * 60) -> int:
    # Simple inference; falls back to minute trading year if uncertain
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return default
    dt = np.diff(index.values).astype("timedelta64[s]").astype(float)
    if not np.isfinite(dt).any():
        return default
    med_sec = np.median(dt[dt > 0])
    if not np.isfinite(med_sec) or med_sec <= 0:
        return default
    periods_per_day = 86400.0 / med_sec
    # Assume 252 trading days/year
    return int(round(252 * periods_per_day))

def _worker_init(pickled_strategy: bytes, cash: float, commision: float,
                 commision_type, lot_size: int):
    """
    Initializer for worker processes. Stores a pickled strategy and
    backtest config in module globals so each worker reuses them.
    """
    global _WORKER_PICKLED_STRAT, _WORKER_BT_CONFIG
    _WORKER_PICKLED_STRAT = pickled_strategy
    _WORKER_BT_CONFIG = {
        "cash": cash,
        "commision": commision,
        "commision_type": commision_type,
        "lot_size": lot_size,
    }

def _worker_eval(param_items):
    """
    Worker evaluation function.

    param_items: sequence of (key, value) pairs (tuple) to reconstruct dict
    Returns a small dict with metrics (no heavy objects).
    """
    global _WORKER_PICKLED_STRAT, _WORKER_BT_CONFIG
    # Reconstruct params dict
    params = dict(param_items)

    # Unpickle a fresh strategy instance for this task
    strat = pickle.loads(_WORKER_PICKLED_STRAT)

    # Apply param overrides
    for k, v in params.items():
        setattr(strat, k, v)

    # Run backtest locally in worker (no progress bar)
    bt = SimpleBacktester(
        strat,
        cash=_WORKER_BT_CONFIG["cash"],
        commission=_WORKER_BT_CONFIG["commision"],
        commission_type=_WORKER_BT_CONFIG["commision_type"],
        lot_size=_WORKER_BT_CONFIG["lot_size"],
    )
    report = bt.run(progress_bar=False)

    # Compute metrics (same logic as before)
    equity = report.PnlRecord.astype(float)
    returns = equity.pct_change().dropna()

    annual_rf = 0.04
    rf_per_period = annual_rf / report.periods_per_year

    if len(returns) < 2 or returns.std(ddof=1) == 0:
        sharpe = float("nan")
    else:
        excess = returns - rf_per_period
        mean = excess.mean()
        vol = excess.std(ddof=1)
        sharpe = float((mean / vol) * (report.periods_per_year ** 0.5))

    running_max = equity.cummax()
    drawdown = ((equity - running_max) / running_max).min()
    mdd = float(abs(drawdown))

    tot_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)

    # Keep worker returned payload small — don't send large objects back.
    result = {
        "params": params,
        "final_cash": report.final_cash,
        "total_return": tot_return,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "trades": len(report.orders),
    }

    # Cleanup references to free memory inside worker
    del strat, bt, report, equity, returns
    gc.collect()

    return result

@dataclass
class BacktestReport:
    starting_cash: np.float64
    final_cash: np.float64
    PnlRecord: pd.Series
    orders: list[Order]

    @property
    def periods_per_year(self):
        return _infer_periods_per_year(self.PnlRecord.astype(float).index, 252 * 24 * 60)
    
    def plot(self, figsize: tuple = (10, 5)) -> None:
        """
        Plot the equity curve and optionally the drawdown.

        Args:
            show_drawdown (bool): Whether to include drawdown chart.
            figsize (tuple): Figure size for matplotlib.
        """
        equity = self.PnlRecord.astype(float)
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max

        fig, ax = plt.subplots(
            2, figsize=figsize, sharex=True
        )

        ax_eq, ax_dd = ax

        ax_eq.plot(equity.index, equity.values, label="Equity", color="tab:blue")
        ax_eq.set_ylabel("Equity Value")
        ax_eq.set_title("Equity Curve")
        ax_eq.legend()
        ax_eq.grid(alpha=0.3)

        ax_dd.fill_between(
            drawdown.index,
            drawdown.values,
            color="tab:red",
            alpha=0.3,
            label="Drawdown",
        )
        ax_dd.set_ylabel("Drawdown")
        ax_dd.set_xlabel("Date")
        ax_dd.legend()
        ax_dd.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    def __str__(self) -> str:
        equity = self.PnlRecord.astype(float)
        returns = equity.pct_change().dropna()

        # Risk-free per period from an annual rate
        annual_rf = 0.04
        rf_per_period = annual_rf / self.periods_per_year

        if len(returns) < 2 or returns.std(ddof=1) == 0:
            sharpe = np.nan
            lo = np.nan
            hi = np.nan
        else:
            excess = returns - rf_per_period
            mean = excess.mean()
            vol = excess.std(ddof=1)
            sharpe = (mean / vol) * np.sqrt(self.periods_per_year)

            # Standard error of Sharpe (i.i.d. normal approx)
            n = len(excess)
            se = np.sqrt((1 + 0.5 * sharpe**2) / n)
            z = 1.96  # 95% CI
            lo = sharpe - z * se
            hi = sharpe + z * se

        # Max drawdown on equity curve
        running_max = equity.cummax()
        drawdown = ((equity - running_max) / running_max).min()
        mdd = float(abs(drawdown))

        tot_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        tot_orders = len(self.orders)

        return (
            f"Starting Cash: ${self.starting_cash:,.2f}\n"
            f"Final Cash: ${self.final_cash:,.2f}\n"
            f"Total Return: {tot_return:.2%}\n"
            f"Sharpe Ratio: {sharpe:.2f}" if np.isfinite(sharpe) else
            f"Sharpe Ratio: nan"
        ) + (
            f"\nSharpe Confidence Interval: [{lo:.4f}, {hi:.4f}]"
            if np.isfinite(sharpe) else "\nSharpe Confidence Interval: [nan, nan]"
        ) + (
            f"\nMax Drawdown: {mdd:.2%}\n"
            f"Total Trades: {tot_orders:,}"
        )

class SimpleBacktester():
    def __init__(self, 
                 strategy: Strategy,
                cash: float = 10_000, 
                commission: float = 0.002, 
                commission_type: CommissionType = CommissionType.PERCENTAGE,
                lot_size: int = 1,
                margin_call: float = 0.5 ## 50% of the cash lost
                ):
        self.strategy = copy.deepcopy(strategy)
        self.cash = cash
        self.commission = commission
        self.commission_type = commission_type
        self.lot_size = lot_size
        self.margin_call = margin_call
        source = self.strategy.positions[list(self.strategy.positions.keys())[0]].source
        self.PnLRecord = np.zeros(len(source.data['Close']), dtype=np.float64)
    def run(self, progress_bar: bool = False) -> BacktestReport:
        for key in self.strategy.positions.keys():
            self.strategy.positions[key].cash = np.float64(self.cash)
            self.strategy.positions[key].lot_size = self.lot_size
            self.strategy.positions[key].margin_call = self.margin_call
            self.strategy.positions[key].commision = np.float64(self.commission)
            self.strategy.positions[key].commision_type = self.commission_type

        self.strategy.init()
        ## Simple backtesting loop
        for i in tqdm(range(1, max([len(i) for i in self.strategy.data.values()])), disable=(not progress_bar)):
            for val in self.strategy.data.values():
                val.current_index = i
            for val in self.strategy.positions.values():
                val._iterate(i)
            for item in self.strategy.indicators:
                item._i = i
            self.strategy.next()
        orders: list[Order] = []
        for val in self.strategy.positions.values():
            val.close()
            self.PnLRecord += val.PnLRecord
            orders.extend(val.complete_orders)

        index = list(self.strategy.positions.values())[0].source.data['Close'].index
        return BacktestReport(
            starting_cash=np.float64(self.cash),
            final_cash=self.PnLRecord[-1],
            PnlRecord=pd.Series(self.PnLRecord, index=index),
            orders=orders)
    
    def optimize(self, params: dict[str, range], constraint: Callable[[dict[str, Any]], bool] | None = None):
        """
        Perform a grid search over the provided parameter ranges.

        params: dict mapping strategy attribute names to iterables of candidate values.
        constraint: optional callable that takes the candidate parameter dict and returns True
                    to evaluate the combo or False to skip it. For example:
                        lambda p: p["fast_period"] < p["slow_period"]

        Returns:
            best_params: dict[str, Any]
            best_report: BacktestReport
            results: pd.DataFrame with metrics per combination (only valid/kept combos)
        """
        if not params:
            raise ValueError("params must not be empty")

        keys = list(params.keys())
        value_lists = []
        for k in keys:
            vals = params[k]
            # Ensure iterability and materialize to list for cartesian product
            try:
                candidates = list(vals)
            except TypeError:
                raise TypeError(f"Parameter '{k}' must be iterable")
            if len(candidates) == 0:
                raise ValueError(f"Parameter '{k}' has no candidate values")
            value_lists.append(candidates)

        results_rows = []
        best_report = None
        best_params = None
        best_score = -np.inf

        total_combos = len(list(itertools.product(*value_lists)))

        for combo in tqdm(itertools.product(*value_lists), total=(total_combos)):
            # Build parameter dict for this combo
            row_params = {k: v for k, v in zip(keys, combo)}

            # Apply optional constraint; skip combo if it returns False or raises
            if constraint is not None:
                try:
                    if not bool(constraint(row_params)):
                        continue
                except Exception:
                    # If the constraint itself errors, treat as invalid combo
                    continue

            # Fresh strategy copy per combo
            strat_copy = copy.deepcopy(self.strategy)
            for k, v in row_params.items():
                setattr(strat_copy, k, v)

            # Run a fresh backtest instance retaining runtime settings
            bt = SimpleBacktester(
                strat_copy,
                cash=self.cash,
                commission=self.commission,
                commission_type=self.commission_type,
                lot_size=self.lot_size,
            )
            report = bt.run(progress_bar=False)

            # Compute metrics
            equity = report.PnlRecord.astype(float)
            returns = equity.pct_change().dropna()

            # Risk-free per period from an annual rate
            annual_rf = 0.04
            rf_per_period = annual_rf / report.periods_per_year

            if len(returns) < 2 or returns.std(ddof=1) == 0:
                sharpe = np.nan
                lo = np.nan
                hi = np.nan
            else:
                excess = returns - rf_per_period
                mean = excess.mean()
                vol = excess.std(ddof=1)
                sharpe = (mean / vol) * np.sqrt(report.periods_per_year)

            # Max drawdown on equity curve
            running_max = equity.cummax()
            drawdown = ((equity - running_max) / running_max).min()
            mdd = float(abs(drawdown))

            tot_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)

            row = dict(row_params)
            row.update(
                {
                    "final_cash": report.final_cash,
                    "total_return": tot_return,
                    "sharpe": sharpe,
                    "max_drawdown": mdd,
                    "trades": report.orders,
                }
            )
            results_rows.append(row)

            # Selection score: prefer Sharpe, then total return, then final cash
            score = sharpe
            if not np.isfinite(score):
                score = -1000 ## Really bad

            if score > best_score:
                best_score = score
                best_params = {k: v for k, v in zip(keys, combo)}
                best_report = report

        results_df = pd.DataFrame(results_rows)

        # Sort results by composite score (Sharpe desc, then return, then cash)
        if not results_df.empty:
            def _to_score(val):
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    return None
                return v if np.isfinite(v) else None

            scores = []
            for _, r in results_df.iterrows():
                s = _to_score(r.get("sharpe"))
                if s is None:
                    s = _to_score(r.get("total_return"))
                if s is None:
                    s = _to_score(r.get("final_cash"))
                scores.append(s if s is not None else float("-inf"))
            results_df["_score"] = scores
            results_df.sort_values(by=["_score"], ascending=False, inplace=True, kind="mergesort")
            results_df.drop(columns=["_score"], inplace=True)

        return best_params or {}, best_report, results_df
    
    def optimize_parallel(self,
             params: dict[str, range],
             constraint: Callable[[dict[str, Any]], bool] | None = None,
             workers: int | None = None,
             chunksize: int = 1):
        """
        Parallel grid search.

        - workers: max number of worker processes to use (default: min(os.cpu_count()-1, 4))
        - chunksize: passed to Executor.map -- tune when you have many small tasks.

        Returns (best_params, best_report, results_df)
        """
        if not params:
            raise ValueError("params must not be empty")

        keys = list(params.keys())
        value_lists = []
        lens = []
        for k in keys:
            vals = params[k]
            try:
                candidates = list(vals)
            except TypeError:
                raise TypeError(f"Parameter '{k}' must be iterable")
            if len(candidates) == 0:
                raise ValueError(f"Parameter '{k}' has no candidate values")
            value_lists.append(candidates)
            lens.append(len(candidates))

        # determine total combos without materializing them
        total_combos = 1
        for L in lens:
            total_combos *= L

        # prepare iterable of param dicts as sequences of items (so pickling is slightly cheaper)
        def _param_items_iter():
            for combo in itertools.product(*value_lists):
                row_params = {k: v for k, v in zip(keys, combo)}
                if constraint is not None:
                    try:
                        if not bool(constraint(row_params)):
                            continue
                    except Exception:
                        continue
                # yield as tuple of items for stable order and smaller IPC
                yield tuple(row_params.items())

        # choose worker count conservatively to avoid RAM hogging
        cpu_count = os.cpu_count() or 1
        if workers is None:
            workers = max(1, min(cpu_count - 1, 4))
        else:
            workers = max(1, int(workers))

        # pickle the base strategy once and send bytes to worker initializer
        pickled_strategy = pickle.dumps(self.strategy)

        results_rows = []
        # Use ProcessPoolExecutor with worker initializer so each worker holds
        # exactly one copy of the strategy in memory.
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(
                pickled_strategy,
                self.cash,
                self.commission,
                self.commission_type,
                self.lot_size,
            ),
        ) as exe:
            # map the worker over param item tuples
            # use list() on map to iterate with tqdm and collect results
            it = exe.map(_worker_eval, _param_items_iter(), chunksize=chunksize)
            # iterate with progress display
            for res in tqdm(it, total=total_combos, disable=(total_combos <= 1)):
                results_rows.append(res)

        # Build DataFrame of small metrics returned from workers
        results_df = pd.DataFrame(results_rows)
        # Compute a composite score like before: prefer sharpe, then return, then final_cash
        if not results_df.empty:
            def _to_score(val):
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    return None
                return v if np.isfinite(v) else None

            scores = []
            for _, r in results_df.iterrows():
                ret = r.get("total_return")
                s = None
                if (not ret == None and ret > 0):
                    s = _to_score(r.get("sharpe"))
                scores.append(s if s is not None else float("-inf"))
            results_df["_score"] = scores
            results_df.sort_values(by=["_score"], ascending=False, inplace=True, kind="mergesort")
            results_df.drop(columns=["_score"], inplace=True)

        # Determine best params from results_df if any
        if results_df.empty:
            return {}, None, results_df

        best_row = results_df.iloc[0]
        best_params = best_row["params"]

        # Re-run full backtest locally in main process for best_params to obtain
        # the full BacktestReport (includes PnLRecord and orders)
        strat_copy = copy.deepcopy(self.strategy)
        for k, v in best_params.items():
            setattr(strat_copy, k, v)
        bt = SimpleBacktester(
            strat_copy,
            cash=self.cash,
            commission=self.commission,
            commission_type=self.commission_type,
            lot_size=self.lot_size,
        )
        best_report = bt.run(progress_bar=False)

        return best_params, best_report, results_df