"""
Crypto AI Ensemble Trading System

A sophisticated cryptocurrency automated trading application powered by 
ensemble AI analysis with enterprise-grade security and risk management.

Main Components:
- AI Ensemble Models (10 advanced ML models)
- Real-time Data Pipeline
- Risk Management System
- Trading Execution Engine
- Security Framework
- Monitoring and Alerting

Author: AI Trading System Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "AI Trading System Team"
__description__ = "Crypto AI Ensemble Trading System"

# Import key components lazily so lightweight tooling (e.g. FastAPI sidecar)
# can import the package without pulling in heavy ML dependencies upfront.
from importlib import import_module
from typing import Any, Callable, Dict


_LAZY_TARGETS: Dict[str, str] = {
    "get_config": "project.src.utils.config.get_config",
    "get_config_manager": "project.src.utils.config.get_config_manager",
    "get_logger": "project.src.utils.logger.get_logger",
    "get_structured_logger": "project.src.utils.logger.get_structured_logger",
    "log_trade": "project.src.utils.logger.log_trade",
    "log_security_event": "project.src.utils.logger.log_security_event",
    "get_credential_manager": "project.src.security.credential_manager.get_credential_manager",
    "initialize_credentials_from_env": "project.src.security.credential_manager.initialize_credentials_from_env",
}


def _lazy_import(name: str) -> Callable[..., Any]:
    module_path, attr = name.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, attr)


def __getattr__(item: str) -> Any:  # pragma: no cover - passthrough helper
    target = _LAZY_TARGETS.get(item)
    if not target:
        raise AttributeError(f"module 'project' has no attribute '{item}'")
    return _lazy_import(target)


__all__ = list(_LAZY_TARGETS.keys())
