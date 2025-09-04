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

# Import key components for easy access
from .utils.config import get_config, get_config_manager
from .utils.logger import get_logger, get_structured_logger, log_trade, log_security_event
from .security.credential_manager import get_credential_manager, initialize_credentials_from_env

__all__ = [
    "get_config",
    "get_config_manager", 
    "get_logger",
    "get_structured_logger",
    "log_trade",
    "log_security_event",
    "get_credential_manager",
    "initialize_credentials_from_env"
]