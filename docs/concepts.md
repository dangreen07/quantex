# Key Concepts

Understanding **QuantEx**'s building blocks will help you extend the library or plug it into a larger research stack.

---

## DataSource

A `DataSource` acts as an adapter around *anything* that can emit market bars (or ticks). The back-testing engine interacts with it in a loop:

1. `get_current_bar()` – the engine fetches the bar for the current timestamp.
2. Your strategy can call `get_lookback_data(n)` on the source to get historical context.
3. `_increment_index()` – the engine advances the source to the next bar.

Concrete types:

* `CSVDataSource` – ships with QuantEx for OHLCV CSVs.
* You can subclass `BacktestingDataSource` for SQLite, Parquet, Arrow, etc.

## Models

* `Bar`, `Tick` – immutable market data records (`dataclass(frozen=True)`).
* `Order`, `Fill` – represent trading intent and its execution. They are simple dataclasses.
* `Position`, `Portfolio` – stateful classes that track exposure, P&L, and cash. They contain the core position-keeping logic.

The data-oriented classes (`Bar`, `Tick`, `Order`, `Fill`) are simple and easy to serialize, while `Position` and `Portfolio` encapsulate behavior.

## Strategy

Subclass `quantex.strategy.Strategy` and override `run()`.
A strategy instance owns:

* `data_sources` – mapping of name → DataSource
* `portfolio` – holds cash & positions
* `buy()`, `sell()` – helpers to create market or limit orders.
* `close_position()` – helper to exit a position for a given symbol.
* `submit_order()` – the underlying method to queue any `Order` object.

Because strategies share a code-path between back-test and live trading, sticking to the public API keeps you portable.

## EventBus & Execution Simulator

The `EventBus` keeps time in sync across all data sources:

1. Snapshot current bars & create a price dictionary *(common timeline, see note below)*
2. Call `strategy.run()` (strategy may queue orders)
3. Forward queued orders to a `NextBarSimulator` (default)
4. Update NAV, record the timestamp
5. Increment indices

> **Timeline semantics** – QuantEx now uses the **intersection** of all data-source indices as its global timeline. This means a bar is processed **only if every source provides a record for that exact timestamp**. Missing observations are *not* forward-filled across symbols; instead the bar is skipped entirely. This guarantees that strategy helpers (e.g. `get_price`, `price_history`) never encounter `NaN` values.

`NextBarSimulator` queues orders raised during bar *t* and converts them into `Fill`s at the *open* of bar *t+1*, then updates the portfolio. For immediate, zero-latency fills you can opt into `ImmediateFillSimulator` instead.

By default the `EventBus` uses `NextBarSimulator`. You can swap in alternative execution models (e.g. `ImmediateFillSimulator`) by passing a custom `simulator` instance to `BacktestRunner`.

## BacktestRunner

High-level façade that wires everything and returns a `BacktestResult` with:

* `nav` – pandas Series of net-asset-value
* `orders`, `fills` – raw execution logs
* `metrics` – dict of summary stats (currently `total_return`, more soon)

Builds on the lower-level EventBus so you remain free to craft your own loop if needed.

---

## Extending QuantEx

* **Custom metrics** – The `BacktestResult` contains a raw `nav` series and `fills` list. You can compute custom metrics from these after a backtest has run. To embed them directly into the result object, you can subclass `BacktestRunner` and override its `run` method.
* **Alternative data** – implement a new `DataSource`.
* **Execution realism** – The framework is designed to support this, but currently the `EventBus` is tightly coupled to `ImmediateFillSimulator`. Future work will introduce an abstract execution handler to make this pluggable.
* **CLI / Web UI** – QuantEx is library-first; a `quant backtest` wrapper is planned for convenience. 