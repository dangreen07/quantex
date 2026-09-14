from quantex.commission import Commission
from quantex.datasource import DataSource, PricingData
from quantex.strategy import Indicator, Strategy
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from enum import Enum
import pandas as pd
import numpy as np
import optuna
import tqdm
import math
import copy


class SearchType(Enum):
    GRID = 1
    OPTUNA = 2


@dataclass
class Result:
    """
    This class contains the result of a backtest.
    """

    def __init__(
        self,
        equity: np.ndarray,
        starting_cash: float,
        run_strategy: Strategy,
        total_trades: int = 0,
    ):
        self.equity = equity
        self.starting_cash = starting_cash
        self.run_strategy = run_strategy
        self.total_trades = total_trades

    @property
    def total_return(self) -> float:
        """
        The total return of the backtest as a percentage.
        """
        return (self.equity[-1] - self.starting_cash) / self.starting_cash

    def sharpe_ratio(self, risk_free_rate: float = 0.04) -> float:
        """
        The yearly sharpe ratio of the backtest.

        Parameters:
            risk_free_rate:
                The risk free rate to be used in the calculation.

        Returns:
            The yearly sharpe ratio of the backtest.
        """
        total_time = (
            self.run_strategy.data.Timestamp[-1] - self.run_strategy.data.Timestamp[0]
        )
        years = total_time.days / 365.25
        periods_per_year = len(self.equity) / years
        returns = pd.Series(
            self.equity, index=self.run_strategy.data.Timestamp
        ).pct_change()
        risk_free = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        std = returns.std()
        if std == 0:
            return 0
        sharpe = (returns.mean() - risk_free) / std * np.sqrt(periods_per_year)
        return sharpe

    @property
    def annualized_return(self):
        """
        The annualized return of the backtest.
        """
        total_time = (
            self.run_strategy.data.Timestamp[-1] - self.run_strategy.data.Timestamp[0]
        )
        years = total_time.days / (365.25)
        return (1 + self.total_return) ** (1 / years) - 1

    @property
    def max_drawdown(self) -> tuple[float, float]:
        """
        The maximum drawdown of the backtest

        Returns:
            The maximum drawdown in dollars and percentage, respectively.
        """
        peak = self.equity[0]
        max_drawdown_dollars = 0
        max_drawdown_percent = 0
        for value in self.equity:
            drawdown_dollars = value - peak
            drawdown_percent = (value - peak) / peak
            max_drawdown_dollars = min(
                max_drawdown_dollars,
                drawdown_dollars,
            )
            max_drawdown_percent = min(
                max_drawdown_percent,
                drawdown_percent,
            )
        return abs(max_drawdown_dollars), abs(max_drawdown_percent)


class Backtester:
    def __init__(
        self,
        strategy: type[Strategy],
        commission: Commission | None = None,
        cash: float = 10_000,
    ):
        """
        This class is used to run a backtest on a strategy.

        Parameters:
            strategy:
                The strategy to be run.
            commission:
                The commission to be applied to the strategy. If None, no commission will be applied.
            cash:
                The starting cash to be used in the backtest.
        """
        self.strategy = strategy
        self.data = PricingData()
        self.cash = cash
        self.commission = commission or Commission()

    def add_data(self, data: DataSource, name: str | None = None):
        """
        Adds a data source to the backtest.

        Parameters:
            data:
                The data source to be added.
            name:
                The name of the data source. If None, the name will be the same as the data source.
        """
        if name is None:
            name = data.name
        self.data.add_data(data, name)

    def run(self, params: dict | None = None) -> Result:
        """
        Runs the backtest.

        Parameters:
            params: The parameters to use for the backtest. If None, the default parameters will be used.

        Returns:
            Result: The result of the backtest.
        """
        data = copy.deepcopy(self.data)
        strat = self.strategy(data, cash=self.cash)
        if params is not None:
            for name, value in params.items():
                setattr(strat, name, value)
        strat.init()
        indicators = [
            i for i in strat.__dict__.keys() if isinstance(strat.__dict__[i], Indicator)
        ]
        for name in data.datas.keys():
            data.datas[name]._current = 0  ## Reset the current index
        for indicator in indicators:
            strat.__dict__[indicator]._current = 0
        initial_cash = strat.broker.cash
        equity = np.full(len(data.index), initial_cash, dtype=np.float64)
        for i in range(len(data.index)):
            for name in data.datas.keys():
                data.datas[
                    name
                ]._current += (
                    1  ## TODO: Handle multiple data sources with different indexes
                )
            skip_nan = False
            for indicator in indicators:
                strat.__dict__[
                    indicator
                ]._current += 1  ## Shouldn't have to handle multiple indexers as they should all be the same
                ind: Indicator = strat.__dict__[indicator]
                if np.isnan(ind[-1]):
                    skip_nan = True
            if skip_nan:
                continue
            strat.broker.__process_orders__()
            strat.next()
            equity[i] = strat.broker.equity()
        result = Result(equity, initial_cash, strat, strat.broker.total_trades)
        return result

    def optimize(
        self,
        params: dict[str, list],
        constraint: Callable[[dict], bool] | None = None,
        max_trials: int = 100,
        search_type: SearchType = SearchType.GRID,
        risk_free_rate: float = 0.04,
        seed: int = 0,
    ) -> tuple[float, dict]:
        """
        Optimizes the backtest with the given parameters.

        Parameters:
            params: The parameters to optimize.
            max_trials: The maximum number of trials to run.
            search_type: The search type to use.
        """
        self.results = []
        self.max_sharpe = -np.inf
        self.best_trial = {}
        if search_type == SearchType.GRID:
            search_space = (
                dict(zip(params.keys(), values)) for values in product(*params.values())
            )
            search_space = list(search_space)
            if constraint is not None:
                search_space = [trial for trial in search_space if constraint(trial)]
            search_space = search_space[:max_trials]

            def run_trial(trial):
                result = self.run(params=trial)
                sharpe = result.sharpe_ratio(risk_free_rate=risk_free_rate)
                if sharpe > self.max_sharpe:
                    self.max_sharpe = sharpe
                    self.best_trial = trial
                self.results.append((trial, sharpe))

            for trial in tqdm.tqdm(search_space, desc="Processing"):
                run_trial(trial)
        elif search_type == SearchType.OPTUNA:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            sampler = optuna.samplers.TPESampler(seed=seed)
            study = optuna.create_study(
                direction="maximize",
                sampler=sampler,
            )

            def objective(optuna_trial: optuna.Trial):
                trial_params = {
                    name: optuna_trial.suggest_categorical(name, values)
                    for name, values in params.items()
                }

                if constraint is not None and not constraint(trial_params):
                    raise optuna.TrialPruned("Parameter constraint failed")

                result = self.run(params=trial_params)
                sharpe = result.sharpe_ratio(risk_free_rate=risk_free_rate)

                if sharpe is None or not math.isfinite(sharpe):
                    raise optuna.TrialPruned("Invalid Sharpe ratio")

                self.results.append((trial_params, sharpe))
                return sharpe

            study.optimize(objective, n_trials=max_trials, show_progress_bar=True)

            if study.best_trial is None:
                raise ValueError("No valid parameter combination was found")

            self.best_trial = study.best_params
            self.max_sharpe = study.best_value
        return (self.max_sharpe, self.best_trial)
