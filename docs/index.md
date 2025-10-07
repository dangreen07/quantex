# QuantEx Documentation

Welcome to the **QuantEx** documentation! QuantEx is a lightweight, flexible quantitative finance library designed for building trading strategies, backtesting, and analyzing financial market data.

## What is QuantEx?

QuantEx provides a simple yet powerful foundation for:

- **Strategy Development**: Create custom trading strategies using an intuitive API
- **Data Management**: Handle historical market data from various sources
- **Backtesting**: Test your strategies against historical data with realistic execution simulation
- **Performance Analysis**: Comprehensive metrics and reporting for strategy evaluation
- **Optimization**: Built-in parameter optimization with parallel processing support

## Key Features

### 🚀 Lightweight & Fast
- Minimal dependencies and overhead
- Optimized for performance with NumPy/Pandas integration
- Support for high-frequency data processing

### 📊 Comprehensive Backtesting
- Realistic order execution simulation
- Commission and slippage modeling
- Multiple timeframes and instruments
- Detailed performance metrics (Sharpe ratio, max drawdown, etc.)

### 🔧 Flexible Strategy Framework
- Abstract base class for easy strategy implementation
- Event-driven architecture
- Support for multiple indicators and signals
- Easy integration with existing pandas workflows

### ⚡ Parallel Optimization
- Multi-core parameter optimization
- Grid search and constraint-based optimization
- Memory-efficient worker processes

## Quick Start

### Installation

```bash
pip install quantex
```

### Basic Example

```python
from quantex import Strategy, SimpleBacktester, CSVDataSource
import pandas as pd
import numpy as np

class MovingAverageStrategy(Strategy):
    def __init__(self, fast_period=10, slow_period=30):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period

    def init(self):
        # Load data and create indicators
        data = CSVDataSource('data/EURUSD.csv')
        self.add_data(data, 'EURUSD')

        # Create moving averages
        close_prices = self.data['EURUSD'].Close
        self.fast_ma = self.Indicator(self.sma(close_prices, self.fast_period))
        self.slow_ma = self.Indicator(self.sma(close_prices, self.slow_period))

    def next(self):
        # Trading logic
        if len(self.fast_ma) < 2:
            return

        # Buy signal: fast MA crosses above slow MA
        if self.fast_ma[-1] > self.slow_ma[-1] and self.fast_ma[-2] <= self.slow_ma[-2]:
            self.positions['EURUSD'].buy(0.5)  # Buy with 50% of available cash

        # Sell signal: fast MA crosses below slow MA
        elif self.fast_ma[-1] < self.slow_ma[-1] and self.fast_ma[-2] >= self.slow_ma[-2]:
            self.positions['EURUSD'].sell(0.5)  # Sell 50% of position

    def sma(self, series, period):
        return pd.Series(series).rolling(window=period).mean().values

# Run backtest
strategy = MovingAverageStrategy(fast_period=10, slow_period=30)
backtester = SimpleBacktester(strategy, cash=10000)
report = backtester.run()

print(f"Total Return: {report.total_return:.2%}")
print(f"Sharpe Ratio: {report.sharpe:.2f}")
print(f"Max Drawdown: {report.max_drawdown:.2%}")
```

## Documentation Structure

This documentation is organized into several sections:

### 📚 [Guides](./usage/)
Step-by-step tutorials and guides for common tasks:

- **[Strategies](./usage/strategy.md)**: Learn how to create and implement trading strategies
- **[Data Sources](./usage/data-sources.md)**: Handle market data from various sources
- **[Backtesting](./usage/backtesting.md)**: Master the backtesting engine
- **[Execution](./usage/execution.md)**: Understand order execution simulation
- **[Indicators](./usage/indicators.md)**: Work with technical indicators
- **[Optimization](./usage/optimizer.md)**: Optimize strategy parameters

### 🔧 [API Reference](./reference/)
Complete API documentation for all modules:

- **[Models](./reference/quantex.models.md)**: Core data models and structures
- **[Sources](./reference/quantex.sources.md)**: Data source implementations
- **[Strategy](./reference/quantex.strategy.md)**: Strategy framework
- **[Engine](./reference/quantex.engine.md)**: Backtesting engine
- **[Execution](./reference/quantex.execution.md)**: Order execution system
- **[Backtest](./reference/quantex.backtest.md)**: Backtesting classes
- **[Optimizer](./reference/quantex.optimizer.md)**: Optimization tools

## Architecture Overview

QuantEx follows a modular architecture with clear separation of concerns:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Strategy    │    │   Data Source   │    │   Backtester    │
│                 │    │                 │    │                 │
│ • Trading Logic │◄──►│ • Market Data   │◄──►│ • Performance   │
│ • Indicators    │    │ • OHLCV Bars    │    │ • Risk Metrics  │
│ • Signals       │    │ • Time Series   │    │ • Optimization  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                       ┌─────────────────┐
                       │     Broker      │
                       │                 │
                       │ • Order Mgmt    │
                       │ • Position     │
                       │ • PnL Tracking │
                       └─────────────────┘
```

## Contributing

We welcome contributions! Please see our [GitHub repository](https://github.com/dangreen07/quantex) for:

- Bug reports and feature requests
- Contributing guidelines
- Development setup instructions
- Code of conduct

## License

QuantEx is released under the MIT License. See [LICENSE](../LICENSE.md) for details.

## Support

- 📖 **Documentation**: Comprehensive guides and API reference
- 🐛 **Issues**: Report bugs and request features on GitHub
- 💬 **Discussions**: Join the community discussions
- 📧 **Email**: Contact the maintainers for questions

---

**Ready to get started?** Head over to the [Strategies guide](./usage/strategy.md) to learn how to build your first trading strategy!