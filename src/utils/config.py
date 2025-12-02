"""
Configuration Management System for Crypto AI Trading Platform

This module provides centralized configuration management with environment 
variable substitution, validation, and hot reloading capabilities.
"""

import os
import yaml
import asyncio
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from pydantic import BaseModel, Field, validator
from dataclasses import dataclass
try:
    import structlog  # type: ignore
except ImportError:  # pragma: no cover - fallback when optional dep missing
    import logging

    class _StructlogShim:
        """Minimal shim so config manager works without structlog installed."""

        @staticmethod
        def get_logger(name: Optional[str] = None):  # type: ignore[override]
            return logging.getLogger(name or __name__)

    structlog = _StructlogShim()  # type: ignore
from datetime import datetime
import re

logger = structlog.get_logger(__name__)


class DatabaseConfig(BaseModel):
    """Database configuration settings."""
    
    postgresql: Dict[str, Any] = Field(default_factory=dict)
    influxdb: Dict[str, Any] = Field(default_factory=dict) 
    redis: Dict[str, Any] = Field(default_factory=dict)


class APIConfig(BaseModel):
    """API configuration settings."""
    
    helius: Dict[str, Any] = Field(default_factory=dict)
    tradingview: Dict[str, Any] = Field(default_factory=dict)
    exchanges: Dict[str, Any] = Field(default_factory=dict)
    social: Dict[str, Any] = Field(default_factory=dict)


class AIModelConfig(BaseModel):
    """AI model configuration settings."""
    
    ensemble: Dict[str, Any] = Field(default_factory=dict)
    models: Dict[str, Any] = Field(default_factory=dict)


class TradingConfig(BaseModel):
    """Trading configuration settings."""
    
    mode: str = Field(default="paper", pattern="^(paper|live)$")
    position_sizing: Dict[str, Any] = Field(default_factory=dict)
    risk_management: Dict[str, Any] = Field(default_factory=dict)
    execution: Dict[str, Any] = Field(default_factory=dict)
    symbols: List[str] = Field(default_factory=list)


class SecurityConfig(BaseModel):
    """Security configuration settings."""
    
    encryption: Dict[str, Any] = Field(default_factory=dict)
    authentication: Dict[str, Any] = Field(default_factory=dict)
    api_keys: Dict[str, Any] = Field(default_factory=dict)
    audit_logging: Dict[str, Any] = Field(default_factory=dict)


class MonitoringConfig(BaseModel):
    """Monitoring and alerting configuration."""
    
    prometheus: Dict[str, Any] = Field(default_factory=dict)
    grafana: Dict[str, Any] = Field(default_factory=dict)
    alerts: Dict[str, Any] = Field(default_factory=dict)
    health_checks: Dict[str, Any] = Field(default_factory=dict)
    performance_tracking: Dict[str, Any] = Field(default_factory=dict)


class BacktestingConfig(BaseModel):
    """Backtesting configuration settings."""
    
    start_date: str = Field(default="2022-01-01")
    end_date: str = Field(default="2024-01-01")
    initial_capital: float = Field(default=100000, gt=0)
    timeframe: str = Field(default="5m")
    data_sources: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)


class SystemConfig(BaseModel):
    """System-level configuration settings."""
    
    environment: str = Field(default="development", pattern="^(development|staging|production)$")
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    timezone: str = Field(default="UTC")
    max_workers: int = Field(default=4, gt=0)


class TradingSystemConfig(BaseModel):
    """Main configuration model for the entire trading system."""
    
    system: SystemConfig = Field(default_factory=SystemConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    apis: APIConfig = Field(default_factory=APIConfig)
    ai_models: AIModelConfig = Field(default_factory=AIModelConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    backtesting: BacktestingConfig = Field(default_factory=BacktestingConfig)
    development: Dict[str, Any] = Field(default_factory=dict)


class ConfigManager:
    """
    Centralized configuration management with environment variable substitution,
    validation, and hot reloading capabilities.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path or self._find_config_file()
        self.config: Optional[TradingSystemConfig] = None
        self._last_modified: Optional[datetime] = None
        self._watchers: List[asyncio.Task] = []
        
        logger.info("Configuration manager initialized", config_path=self.config_path)
    
    def _find_config_file(self) -> str:
        """Find the configuration file in common locations."""
        possible_paths = [
            os.path.join(os.getcwd(), "config", "config.yaml"),
            os.path.join(os.getcwd(), "config.yaml"),
            os.path.join(os.path.dirname(__file__), "..", "..", "config", "config.yaml"),
            "/etc/crypto-trading/config.yaml"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Use example config as fallback
        example_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "config.example.yaml")
        if os.path.exists(example_path):
            logger.warning("Using example configuration file", path=example_path)
            return example_path
        
        raise FileNotFoundError("No configuration file found. Please create config/config.yaml")
    
    def _substitute_environment_variables(self, obj: Any) -> Any:
        """
        Recursively substitute environment variables in configuration values.
        
        Args:
            obj: Configuration object (dict, list, or string)
        
        Returns:
            Object with environment variables substituted
        """
        if isinstance(obj, dict):
            return {key: self._substitute_environment_variables(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_environment_variables(item) for item in obj]
        elif isinstance(obj, str):
            # Match ${VARIABLE_NAME} pattern
            pattern = r'\$\{([^}]+)\}'
            
            def replace_env_var(match):
                var_name = match.group(1)
                env_value = os.getenv(var_name)
                
                if env_value is None:
                    logger.warning("Environment variable not found", variable=var_name)
                    return match.group(0)  # Return original if not found
                
                return env_value
            
            return re.sub(pattern, replace_env_var, obj)
        else:
            return obj
    
    def load_config(self) -> TradingSystemConfig:
        """
        Load configuration from file with environment variable substitution.
        
        Returns:
            Validated configuration object
        """
        try:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r') as file:
                raw_config = yaml.safe_load(file)
            
            # Substitute environment variables
            processed_config = self._substitute_environment_variables(raw_config)
            
            # Validate configuration
            self.config = TradingSystemConfig(**processed_config)
            self._last_modified = datetime.fromtimestamp(os.path.getmtime(self.config_path))
            
            logger.info("Configuration loaded successfully", 
                       environment=self.config.system.environment,
                       trading_mode=self.config.trading.mode)
            
            return self.config
            
        except Exception as e:
            logger.error("Failed to load configuration", 
                        config_path=self.config_path, 
                        error=str(e))
            raise
    
    def get_config(self) -> TradingSystemConfig:
        """
        Get the current configuration, loading it if not already loaded.
        
        Returns:
            Current configuration object
        """
        if self.config is None:
            self.load_config()
        
        return self.config
    
    def reload_if_changed(self) -> bool:
        """
        Reload configuration if the file has been modified.
        
        Returns:
            True if configuration was reloaded
        """
        try:
            current_modified = datetime.fromtimestamp(os.path.getmtime(self.config_path))
            
            if self._last_modified is None or current_modified > self._last_modified:
                logger.info("Configuration file changed, reloading...")
                self.load_config()
                return True
            
            return False
            
        except Exception as e:
            logger.error("Failed to check configuration file modification", error=str(e))
            return False
    
    async def watch_for_changes(self, callback=None):
        """
        Watch for configuration file changes and reload automatically.
        
        Args:
            callback: Optional callback function called when config changes
        """
        while True:
            try:
                if self.reload_if_changed():
                    logger.info("Configuration reloaded automatically")
                    
                    if callback:
                        await callback(self.config)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error("Error in configuration watcher", error=str(e))
                await asyncio.sleep(30)  # Wait longer on error
    
    def start_watching(self, callback=None) -> asyncio.Task:
        """
        Start watching for configuration changes in the background.
        
        Args:
            callback: Optional callback function
        
        Returns:
            Background task handle
        """
        task = asyncio.create_task(self.watch_for_changes(callback))
        self._watchers.append(task)
        logger.info("Configuration file watcher started")
        return task
    
    def stop_watching(self):
        """Stop all configuration file watchers."""
        for task in self._watchers:
            task.cancel()
        self._watchers.clear()
        logger.info("Configuration file watchers stopped")
    
    def validate_environment(self) -> Dict[str, bool]:
        """
        Validate that required environment variables are set.
        
        Returns:
            Dictionary mapping environment variable names to availability status
        """
        required_env_vars = [
            'MASTER_PASSWORD',
            'DB_USERNAME',
            'DB_PASSWORD',
            'HELIUS_API_KEY',
            'JWT_SECRET'
        ]
        
        optional_env_vars = [
            'COINBASE_API_KEY',
            'KRAKEN_API_KEY',
            'TWITTER_BEARER_TOKEN',
            'SLACK_WEBHOOK_URL',
            'REDIS_PASSWORD',
            'INFLUXDB_TOKEN'
        ]
        
        env_status = {}
        
        for var in required_env_vars:
            env_status[var] = os.getenv(var) is not None
            if not env_status[var]:
                logger.error(f"Required environment variable not set: {var}")
        
        for var in optional_env_vars:
            env_status[var] = os.getenv(var) is not None
            if not env_status[var]:
                logger.warning(f"Optional environment variable not set: {var}")
        
        return env_status
    
    def get_database_url(self, db_type: str = "postgresql") -> str:
        """
        Construct database connection URL from configuration.
        
        Args:
            db_type: Type of database (postgresql, influxdb, redis)
        
        Returns:
            Database connection URL
        """
        config = self.get_config()
        
        if db_type == "postgresql":
            db_config = config.database.postgresql
            return (f"postgresql://{db_config.get('username')}:"
                   f"{db_config.get('password')}@{db_config.get('host')}:"
                   f"{db_config.get('port')}/{db_config.get('database')}")
        
        elif db_type == "influxdb":
            db_config = config.database.influxdb
            return db_config.get('url')
        
        elif db_type == "redis":
            db_config = config.database.redis
            password = f":{db_config.get('password')}@" if db_config.get('password') else ""
            return f"redis://{password}{db_config.get('host')}:{db_config.get('port')}/{db_config.get('db', 0)}"
        
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    def get_api_config(self, api_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific API.
        
        Args:
            api_name: Name of the API (helius, coinbase, kraken, etc.)
        
        Returns:
            API configuration dictionary
        """
        config = self.get_config()
        
        if api_name in ["helius", "tradingview"]:
            return getattr(config.apis, api_name, {})
        elif api_name in config.apis.exchanges:
            return config.apis.exchanges.get(api_name, {})
        elif api_name in config.apis.social:
            return config.apis.social.get(api_name, {})
        else:
            raise ValueError(f"Unknown API: {api_name}")
    
    def export_config_summary(self) -> Dict[str, Any]:
        """
        Export a sanitized configuration summary for logging/debugging.
        
        Returns:
            Configuration summary with sensitive data removed
        """
        config = self.get_config()
        
        return {
            'system': {
                'environment': config.system.environment,
                'log_level': config.system.log_level,
                'timezone': config.system.timezone,
                'max_workers': config.system.max_workers
            },
            'trading': {
                'mode': config.trading.mode,
                'symbols_count': len(config.trading.symbols),
                'max_risk_per_trade': config.trading.position_sizing.get('max_risk_per_trade'),
                'max_portfolio_heat': config.trading.position_sizing.get('max_portfolio_heat')
            },
            'ai_models': {
                'window_size': config.ai_models.ensemble.get('window_size'),
                'max_models': config.ai_models.ensemble.get('max_models'),
                'enabled_models': [name for name, conf in config.ai_models.models.items() 
                                 if conf.get('enabled', False)]
            },
            'security': {
                'audit_logging_enabled': config.security.audit_logging.get('enabled', False),
                'key_rotation_enabled': config.security.api_keys.get('rotation_enabled', False)
            }
        }


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    
    return _config_manager


def get_config() -> TradingSystemConfig:
    """Get the current system configuration."""
    return get_config_manager().get_config()


if __name__ == "__main__":
    # Example usage and testing
    async def test_config_manager():
        """Test the configuration manager functionality."""
        
        print("Testing configuration manager...")
        
        # Initialize config manager
        cm = ConfigManager()
        
        print("Loading configuration...")
        config = cm.load_config()
        
        print(f"Environment: {config.system.environment}")
        print(f"Trading mode: {config.trading.mode}")
        print(f"Log level: {config.system.log_level}")
        
        print("Validating environment variables...")
        env_status = cm.validate_environment()
        
        missing_required = [var for var, status in env_status.items() 
                           if not status and var in ['MASTER_PASSWORD', 'DB_USERNAME', 'DB_PASSWORD']]
        
        if missing_required:
            print(f"Missing required environment variables: {missing_required}")
        else:
            print("All required environment variables are set!")
        
        print("Testing database URL generation...")
        try:
            # This will fail if credentials aren't set, but that's expected in testing
            db_url = cm.get_database_url("postgresql")
            print(f"PostgreSQL URL: {db_url[:30]}...")
        except Exception as e:
            print(f"Database URL generation failed (expected): {e}")
        
        print("Configuration summary:")
        summary = cm.export_config_summary()
        for section, data in summary.items():
            print(f"  {section}: {data}")
        
        print("All configuration tests completed! 🎉")
    
    # Run tests
    asyncio.run(test_config_manager())
