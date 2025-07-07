# Roadmap

QuantEx is an open-source project designed to evolve from a powerful backtesting library into a comprehensive, end-to-end platform for algorithmic trading. Our roadmap is structured in phases, each building upon the last to deliver a robust and user-friendly experience for quants and developers.

Below is a detailed look at our planned journey.

---

### Phase 1: Core Backtesting SDK (Complete)

This initial phase focused on building the foundational library for local strategy development and backtesting. The core components are now in place.

*   **Core Data Models**: Immutable `Bar` and `Tick` data, `Order`, `Fill`, and stateful `Position` and `Portfolio` objects.
*   **Event-Driven Engine**: A central `EventBus` that coordinates data, strategy, and execution, ensuring a design that mirrors live trading.
*   **Strategy API**: A simple base `Strategy` class for users to inherit from, with methods like `run()` and `submit_order()`.
*   **Data Handling**: A `DataSource` abstraction with an initial `CSVDataSource` for loading local historical data.
*   **Execution Simulation**: An `ImmediateFillSimulator` to execute trades with hooks for future commission and slippage models.
*   **Backtest Runner**: A top-level `BacktestRunner` that wires all components together and produces results, including an initial `total_return` metric.
*   **Comprehensive Testing**: A suite of unit tests to ensure the reliability of the core components.
*   **Initial Documentation**: Foundational API documentation and a quickstart guide to get users started.

---

### Phase 2: Enhanced Tooling & Developer Experience (Current Focus)

With the core SDK in place, this phase is about enriching the developer experience and providing more powerful analytics tools.

*   **Command-Line Interface (CLI)**: A `quant` CLI to simplify running backtests and managing projects.
*   **Advanced Performance Metrics**: Implement key performance indicators beyond simple returns, such as:
    *   Sharpe Ratio & Sortino Ratio
    *   Maximum Drawdown
    *   Calmar Ratio
*   **Results Visualization**: Add plotting capabilities (e.g., using Plotly) to generate equity curves, drawdown charts, and other relevant visualizations from backtest results.
*   **Example Strategies**: Develop and include several example strategies (e.g., Moving Average Crossover, RSI-based) to showcase library features and provide a starting point for users.
*   **Containerization**: Provide a `Dockerfile` and build scripts to containerize trading strategies, ensuring reproducible environments for backtesting and future deployment.
*   **Packaging & CI/CD**: Finalize the Python package for a stable release on PyPI and enhance the CI/CD pipeline for automated testing, linting, and publishing.

---

### Phase 3: Cloud Deployment & Paper Trading (Planned)

This phase bridges the gap from local research to cloud-based testing, introducing the first elements of the deployment platform.

*   **Platform API**: A secure REST/gRPC API (e.g., using FastAPI) to manage and monitor strategies.
*   **Cloud Deployment Workflow**: A mechanism within the `quant` CLI (`quant deploy`) to push containerized strategies to a cloud environment (e.g., Kubernetes).
*   **Internal Broker Simulator**: A simulated paper trading environment that mimics a real brokerage API, allowing users to test their strategies in a pseudo-live environment without financial risk.
*   **Configuration Management**: A secure system for managing strategy configurations and secrets (e.g., API keys) in the cloud.
*   **Basic Observability**: Centralized logging for deployed strategies to facilitate debugging and monitoring.

---

### Phase 4: Live Trading & Web Interface (Planned)

This phase introduces live trading capabilities and a user-friendly web interface for managing the entire trading lifecycle.

*   **Live Broker Integration**: Implement a connector for a live brokerage (e.g., Alpaca, Interactive Brokers) and create a pluggable SDK for users to build custom broker integrations.
*   **Web Dashboard**: A React/Next.js-based web UI for:
    *   Monitoring live strategy performance with real-time equity curves and P&L.
    *   Viewing and comparing historical backtest results.
    *   Managing, configuring, and deploying strategies.
*   **Security Hardening**: Implement robust security practices, including API key encryption, network policies, and role-based access control (RBAC) in the dashboard.
*   **Advanced Observability**: Integrate comprehensive monitoring with metrics (e.g., Prometheus) and traces (e.g., OpenTelemetry) for live strategies.

---

### Phase 5: Public Launch & Community Growth (Planned)

The final phase is focused on polishing the platform, launching it to the public, and fostering a vibrant community.

*   **Beta Program**: Invite a cohort of users to test the end-to-end platform and provide structured feedback.
*   **Documentation Overhaul**: Refine all documentation, tutorials, and examples based on beta feedback to ensure a smooth onboarding experience for new users.
*   **Community Building**: Establish community channels (e.g., Discord, GitHub Discussions) to encourage collaboration, support, and knowledge sharing.
*   **v1.0 Release**: Officially launch the platform, marking the first stable, production-ready version.
*   **Long-Term Support**: Establish a clear process for ongoing maintenance, bug fixes, and feature development driven by community input.

---

*Last updated: {{ date | today }}* 