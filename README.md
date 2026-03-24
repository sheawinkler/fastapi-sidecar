# Crypto AI Ensemble Trading System

## 🚀 Overview

Can be used as a sidecar for other applications to create distinct application surfaces for model running, or can be built out into a fully integrated product.
Supports: 
- Apple Neural Engine Private API ([ANE repo](https://github.com/mhttps://github.com/maderix/ANE.git)) (Apple M-Series)
  - Unreal matrix solving
  - 3800x faster than base runtime (pytorch/tensorflow) in small test
- CoreML fallback (Apple M-Series)
  - 39x faster than base runtime in small test
- regular runtime for final fallback / default for non-MacOS

A sophisticated cryptocurrency automated trading application powered by ensemble AI analysis, implementing 10 cutting-edge machine learning models for superior trading performance. This system is designed for real-money trading with enterprise-grade security, comprehensive risk management, and advanced AI ensemble techniques.

## 🏗️ Architecture

### System Design Philosophy
- **Microservices Architecture**: Scalable, fault-tolerant design with independent components
- **Security-First**: Enterprise-grade security protocols throughout the system
- **AI Ensemble Approach**: Dynamic weighted-majority algorithm with utility-based weighting
- **Real-Time Processing**: Sub-100ms latency for trading decisions and execution

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Pipeline  │    │   AI Ensemble   │    │ Trading Engine  │
│                 │    │                 │    │                 │
│ • Market Data   │───▶│ • 10 ML Models  │───▶│ • Risk Manager  │
│ • Sentiment     │    │ • Dynamic Weights│    │ • Position Sizer│
│ • On-chain      │    │ • Orchestrator  │    │ • Order Router  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Security      │    │   Monitoring    │    │   Compliance    │
│                 │    │                 │    │                 │
│ • Auth/AuthZ    │    │ • Performance   │    │ • Audit Logs    │
│ • Encryption    │    │ • Health Checks │    │ • Risk Reports  │
│ • Key Rotation  │    │ • Alerting      │    │ • Regulatory    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🤖 AI Ensemble Models

### Priority 1 Models (Core Performance)
1. **Executive-Auxiliary Agent Dual Architecture**: Hierarchical RL addressing sparse rewards (6.3%+ improvement)
2. **Cross-Modal Temporal Fusion**: Transformer architecture with 20% improvement through multimodal integration
3. **Progressive Denoising VAE**: Three-stage denoising for financial signal preservation
4. **Functional Data-Driven Quantile Ensemble**: Mathematically proven optimal VaR prediction

### Priority 2 Models (Enhanced Capabilities)
5. **CryptoBERT-Enhanced Sentiment Fusion**: Advanced NLP for crypto-specific sentiment analysis
6. **Temporal Fusion Transformer**: Multi-scale attention for regime-aware predictions
7. **CNN-GAN-Autoencoder Ensemble**: Pattern generation and anomaly detection
8. **Generalized Random Forest VaR**: Research-validated risk prediction during instability

### Priority 3 Models (Advanced Features)
9. **Dynamic Portfolio Optimization**: Statistical arbitrage with CNN-MHA architecture
10. **Multi-Modal Volatility Prediction**: Chaos theory integration achieving 19.29% improvement

## 📊 Performance Targets

- **Annual ROI**: 25-50% (conservative backtesting estimate)
- **Sharpe Ratio**: >1.5 (crypto-adjusted)
- **Maximum Drawdown**: <15% (95% confidence)
- **System Uptime**: >99.5%
- **Trade Execution**: <50ms latency

## 🔒 Security Features

- **Multi-Factor Authentication**: 2FA for all system access
- **Encrypted Credential Storage**: Fernet symmetric encryption with key rotation
- **Comprehensive Audit Logging**: All trading decisions and system access logged
- **Network Security**: TLS 1.3, VPN access, firewall protection
- **Automated Security Monitoring**: Real-time threat detection and response

## 💰 Resource Integration

- **Helius RPC** ($50/month): Optimized Solana blockchain data access
- **TradingView Premium**: Advanced technical analysis and market data
- **Exchange APIs**: Coinbase One & Kraken Pro integration
- **Free Data Sources**: CoinGecko, Alpha Vantage, social media APIs

## 🚦 Risk Management

### Position Sizing
- Kelly Criterion with ensemble confidence weighting
- Maximum 2% risk per trade
- Portfolio heat limit: 10%
- Dynamic volatility adjustment

### Risk Controls
- Real-time P&L monitoring
- Circuit breakers for extreme conditions
- Automated position reduction
- Emergency stop-loss mechanisms

## 📈 Getting Started

### Prerequisites
```bash
Python 3.9+
PostgreSQL 13+
Redis 6+
Git
```

### Installation
```bash
git clone <repository-url>
cd crypto-ai-trading-system
pip install -r requirements.txt
python setup.py install
```

### Configuration
1. Copy `config/config.example.yaml` to `config/config.yaml`
2. Configure API credentials (see Security section)
3. Set up database connections
4. Initialize the system: `python scripts/initialize_system.py`

### Running the System
```bash
# Start data pipeline
python -m src.data.market_data_service

# Start AI ensemble
python -m src.ai_models.ensemble_orchestrator

# Start trading engine
python -m src.trading.trading_engine

# Monitor system
python -m src.utils.system_monitor

# HTTP prediction bridge for external agents
# (set ENSEMBLE_STUB=true to use the lightweight stub)
ENSEMBLE_STUB=true API_PORT=8288 python -m src.api.fastapi_server
```

### FastAPI Prediction Bridge

`project/src/api/fastapi_server.py` wraps the ensemble orchestrator in a
lightweight FastAPI service so external agents (Rust trader, MCP tools, etc.)
can call `/predict` over HTTP.

- Default host/port: `0.0.0.0:8288` (override with `API_HOST` / `API_PORT`).
- Request body example:
  ```json
  {
    "token": "BONK",
    "features": { "momentum": 0.42, "volume": 1.8 },
    "model": "ensemble"  // optional override
  }
  ```
- Response includes `prediction`, `confidence`, `expected_return`,
  `latency_ms`, and the raw result structure.
- `/health` returns model counts + telemetry; `/telemetry` exposes aggregate
  call counts and average latency for dashboards.
- WebSocket guidance stream: connect to `/ws/guidance` to receive low-latency
  guidance events (and optionally publish via `POST /guidance/publish`).
  ```bash
  curl -fsS -X POST http://localhost:8288/guidance/publish \
    -H 'Content-Type: application/json' \
    --data '{"symbol":"SOL/USDC","confidence":0.77,"score":0.66,"notes":"manual-publish"}'
  ```

### Sidecar Launch (uv)

From the `project/` directory, the easiest way to bring the FastAPI sidecar up is:

```bash
# Creates/uses .venv and installs requirements via uv (pip fallback)
scripts/sidecar_run.sh

# Full ensemble mode (heavier deps, slower startup)
ENSEMBLE_STUB=false scripts/sidecar_run.sh

# Override bind address/port
API_HOST=127.0.0.1 API_PORT=8288 scripts/sidecar_run.sh
```

### Sidecar Smoke Test

Use `scripts/sidecar_smoke.sh` to confirm the bridge is online and returning
predictions before pointing external agents at it:

```bash
# From project root
scripts/sidecar_smoke.sh

# Override the target endpoint if needed
SIDECAR_URL=http://localhost:8288 scripts/sidecar_smoke.sh
```

The script pings `/health`, prints the current mode (`stub` vs `full`), and sends
a sample `/predict` payload so you can catch regressions quickly.

## 🧪 Testing

### Backtesting
```bash
python -m tests.backtesting.comprehensive_backtest --start-date 2022-01-01 --end-date 2024-01-01
```

### Paper Trading
```bash
python -m src.trading.paper_trading --duration 30d
```

### Unit Tests
```bash
pytest tests/unit/ -v --cov=src --cov-report=html
```

## 📚 Documentation

- [Technical Architecture](docs/architecture.md)
- [AI Model Documentation](docs/ai_models.md)
- [Security Implementation](docs/security.md)
- [Risk Management](docs/risk_management.md)
- [API Documentation](docs/api.md)
- [Deployment Guide](docs/deployment.md)

## 🔧 Development

### Project Structure
```
src/
├── ai_models/          # AI ensemble models and orchestrator
├── data/              # Data ingestion and processing
├── trading/           # Trading execution and order management
├── risk/              # Risk management and monitoring
├── security/          # Authentication and encryption
└── utils/             # Shared utilities and helpers

tests/
├── unit/              # Unit tests for individual components
├── integration/       # Integration tests
└── backtesting/       # Comprehensive backtesting framework

config/                # Configuration files
docs/                  # Documentation
scripts/               # Deployment and utility scripts
logs/                  # System logs
data/                  # Historical and cache data
```

### Development Workflow
1. Create feature branch from `main`
2. Implement changes with comprehensive tests
3. Run security audit: `python scripts/security_audit.py`
4. Submit pull request with performance validation
5. Code review and automated testing
6. Deploy to staging for validation
7. Production deployment after approval

## 📊 Monitoring and Alerting

### Key Metrics
- Trading performance (P&L, Sharpe ratio, drawdown)
- System health (uptime, latency, errors)
- Model performance (accuracy, confidence, drift)
- Risk metrics (VaR, portfolio heat, correlation)

### Alert Channels
- Slack integration for system alerts
- Email notifications for performance issues
- SMS alerts for critical system failures
- Dashboard visualizations with Grafana

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run the full test suite
5. Submit a pull request

## 📄 License

This project is proprietary software. All rights reserved.

## ⚠️ Disclaimer

This software is designed for sophisticated users who understand the risks of algorithmic trading. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.

## 📞 Support

For technical support and questions:
- Email: sheawinkler@gmail.com
- Documentation: [Technical Docs](docs/)
- Issue Tracker: GitHub Issues

---

**Built with 🔥 for serious crypto traders**
