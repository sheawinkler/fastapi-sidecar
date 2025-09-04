"""
Comprehensive Logging Infrastructure for Crypto AI Trading Platform

This module provides structured logging with multiple outputs, audit trails,
performance tracking, and compliance-ready log formatting.
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import structlog
from structlog import stdlib
from structlog.processors import JSONRenderer, TimeStamper
import colorlog
from pythonjsonlogger import jsonlogger
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import traceback

# Import configuration
from .config import get_config


class TradingSystemLogger:
    """
    Comprehensive logging system with structured logging, audit trails,
    and multiple output formats for different use cases.
    """
    
    def __init__(self):
        """Initialize the logging system."""
        self.loggers: Dict[str, logging.Logger] = {}
        self.structured_logger = None
        self.audit_logger = None
        self.performance_logger = None
        self.trade_logger = None
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up all logging components."""
        try:
            config = get_config()
            log_level = getattr(logging, config.system.log_level.upper())
            environment = config.system.environment
            
            # Create logs directory
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            # Configure structlog
            self._setup_structlog(log_level, environment)
            
            # Set up specialized loggers
            self._setup_audit_logger(log_level)
            self._setup_performance_logger(log_level)
            self._setup_trade_logger(log_level)
            self._setup_system_logger(log_level, environment)
            
            print(f"✅ Logging system initialized (level: {config.system.log_level})")
            
        except Exception as e:
            print(f"❌ Failed to initialize logging system: {e}")
            # Fallback to basic logging
            self._setup_fallback_logging()
    
    def _setup_structlog(self, log_level: int, environment: str):
        """Set up structured logging with structlog."""
        
        # Configure processors based on environment
        if environment == "development":
            processors = [
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.dev.ConsoleRenderer(colors=True)
            ]
        else:
            processors = [
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                JSONRenderer()
            ]
        
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        self.structured_logger = structlog.get_logger("crypto_trading")
    
    def _setup_audit_logger(self, log_level: int):
        """Set up audit logging for compliance and security monitoring."""
        
        # Create audit logger
        audit_logger = logging.getLogger("audit")
        audit_logger.setLevel(logging.INFO)  # Always log audit events
        
        # JSON formatter for audit logs
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S'
        )
        
        # File handler with rotation
        audit_file_handler = TimedRotatingFileHandler(
            filename="logs/audit.log",
            when="midnight",
            interval=1,
            backupCount=365,  # Keep 1 year of audit logs
            encoding='utf-8'
        )
        audit_file_handler.setFormatter(formatter)
        audit_logger.addHandler(audit_file_handler)
        
        # Console handler for development
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '🔍 AUDIT: %(asctime)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        audit_logger.addHandler(console_handler)
        
        self.audit_logger = audit_logger
        self.loggers["audit"] = audit_logger
    
    def _setup_performance_logger(self, log_level: int):
        """Set up performance logging for system monitoring."""
        
        # Create performance logger
        perf_logger = logging.getLogger("performance")
        perf_logger.setLevel(logging.INFO)
        
        # JSON formatter for structured performance data
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S'
        )
        
        # File handler with size-based rotation
        perf_file_handler = RotatingFileHandler(
            filename="logs/performance.log",
            maxBytes=50*1024*1024,  # 50MB per file
            backupCount=10,
            encoding='utf-8'
        )
        perf_file_handler.setFormatter(formatter)
        perf_logger.addHandler(perf_file_handler)
        
        self.performance_logger = perf_logger
        self.loggers["performance"] = perf_logger
    
    def _setup_trade_logger(self, log_level: int):
        """Set up trade-specific logging for execution tracking."""
        
        # Create trade logger
        trade_logger = logging.getLogger("trades")
        trade_logger.setLevel(logging.INFO)
        
        # JSON formatter for trade data
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S'
        )
        
        # File handler with daily rotation
        trade_file_handler = TimedRotatingFileHandler(
            filename="logs/trades.log",
            when="midnight",
            interval=1,
            backupCount=90,  # Keep 3 months of trade logs
            encoding='utf-8'
        )
        trade_file_handler.setFormatter(formatter)
        trade_logger.addHandler(trade_file_handler)
        
        # Console handler for real-time monitoring
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '💰 TRADE: %(asctime)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        trade_logger.addHandler(console_handler)
        
        self.trade_logger = trade_logger
        self.loggers["trades"] = trade_logger
    
    def _setup_system_logger(self, log_level: int, environment: str):
        """Set up general system logging."""
        
        # Create system logger
        system_logger = logging.getLogger("system")
        system_logger.setLevel(log_level)
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            filename="logs/system.log",
            maxBytes=20*1024*1024,  # 20MB per file
            backupCount=5,
            encoding='utf-8'
        )
        
        if environment == "development":
            # Human-readable format for development
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            # JSON format for production
            file_formatter = jsonlogger.JsonFormatter(
                '%(asctime)s %(name)s %(levelname)s %(message)s',
                datefmt='%Y-%m-%dT%H:%M:%S'
            )
        
        file_handler.setFormatter(file_formatter)
        system_logger.addHandler(file_handler)
        
        # Console handler with colors for development
        if environment == "development":
            console_handler = colorlog.StreamHandler()
            console_handler.setFormatter(colorlog.ColoredFormatter(
                '%(log_color)s%(levelname)-8s%(reset)s '
                '%(blue)s%(name)s%(reset)s: %(message)s',
                datefmt='%H:%M:%S',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            ))
            system_logger.addHandler(console_handler)
        
        self.loggers["system"] = system_logger
    
    def _setup_fallback_logging(self):
        """Set up basic logging if configuration fails."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/fallback.log')
            ]
        )
        
        self.structured_logger = structlog.get_logger("fallback")
        print("⚠️ Using fallback logging configuration")
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger by name.
        
        Args:
            name: Logger name (audit, performance, trades, system)
        
        Returns:
            Logger instance
        """
        return self.loggers.get(name, self.loggers.get("system"))
    
    def get_structured_logger(self, name: str = None):
        """
        Get a structured logger instance.
        
        Args:
            name: Optional logger name for context
        
        Returns:
            Structured logger with bound context
        """
        if name:
            return self.structured_logger.bind(component=name)
        return self.structured_logger
    
    def log_trade_execution(self, trade_data: Dict[str, Any]):
        """
        Log trade execution with structured data.
        
        Args:
            trade_data: Trading execution data
        """
        try:
            # Sanitize sensitive data
            sanitized_data = self._sanitize_trade_data(trade_data)
            
            self.trade_logger.info(
                "Trade executed",
                extra={
                    "trade_id": sanitized_data.get("trade_id"),
                    "symbol": sanitized_data.get("symbol"),
                    "side": sanitized_data.get("side"),
                    "quantity": sanitized_data.get("quantity"),
                    "price": sanitized_data.get("price"),
                    "timestamp": datetime.utcnow().isoformat(),
                    "exchange": sanitized_data.get("exchange"),
                    "order_type": sanitized_data.get("order_type"),
                    "fees": sanitized_data.get("fees"),
                    "slippage": sanitized_data.get("slippage")
                }
            )
            
        except Exception as e:
            self.structured_logger.error("Failed to log trade execution", error=str(e))
    
    def log_model_prediction(self, model_name: str, prediction_data: Dict[str, Any]):
        """
        Log AI model predictions for analysis.
        
        Args:
            model_name: Name of the AI model
            prediction_data: Prediction results and metadata
        """
        try:
            self.performance_logger.info(
                "Model prediction generated",
                extra={
                    "model_name": model_name,
                    "timestamp": datetime.utcnow().isoformat(),
                    "prediction": prediction_data.get("prediction"),
                    "confidence": prediction_data.get("confidence"),
                    "features_used": prediction_data.get("features_count"),
                    "processing_time_ms": prediction_data.get("processing_time_ms"),
                    "market_conditions": prediction_data.get("market_conditions")
                }
            )
            
        except Exception as e:
            self.structured_logger.error("Failed to log model prediction", 
                                       model=model_name, error=str(e))
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """
        Log security-related events for audit and monitoring.
        
        Args:
            event_type: Type of security event
            details: Event details and context
        """
        try:
            self.audit_logger.critical(
                f"Security event: {event_type}",
                extra={
                    "event_type": event_type,
                    "timestamp": datetime.utcnow().isoformat(),
                    "user_id": details.get("user_id"),
                    "ip_address": details.get("ip_address"),
                    "action": details.get("action"),
                    "resource": details.get("resource"),
                    "success": details.get("success", False),
                    "error_message": details.get("error_message"),
                    "session_id": details.get("session_id")
                }
            )
            
        except Exception as e:
            # Use fallback logging for security events
            logging.critical(f"Security event logging failed: {e}")
    
    def log_performance_metric(self, metric_name: str, value: float, 
                             context: Optional[Dict[str, Any]] = None):
        """
        Log performance metrics for monitoring.
        
        Args:
            metric_name: Name of the performance metric
            value: Metric value
            context: Additional context data
        """
        try:
            metric_data = {
                "metric_name": metric_name,
                "value": value,
                "timestamp": datetime.utcnow().isoformat(),
                "unit": context.get("unit") if context else None
            }
            
            if context:
                metric_data.update(context)
            
            self.performance_logger.info(
                f"Performance metric: {metric_name}",
                extra=metric_data
            )
            
        except Exception as e:
            self.structured_logger.error("Failed to log performance metric",
                                       metric=metric_name, error=str(e))
    
    def _sanitize_trade_data(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove or mask sensitive data from trade logs.
        
        Args:
            trade_data: Raw trade data
        
        Returns:
            Sanitized trade data
        """
        sensitive_fields = ["api_key", "secret", "private_key", "password"]
        sanitized = trade_data.copy()
        
        for field in sensitive_fields:
            if field in sanitized:
                sanitized[field] = "***MASKED***"
        
        return sanitized
    
    async def log_system_health(self):
        """Continuously log system health metrics."""
        while True:
            try:
                import psutil
                
                # System metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                self.log_performance_metric("cpu_usage_percent", cpu_percent)
                self.log_performance_metric("memory_usage_percent", memory.percent)
                self.log_performance_metric("disk_usage_percent", disk.percent)
                self.log_performance_metric("memory_available_gb", memory.available / (1024**3))
                
                # Wait 60 seconds between health checks
                await asyncio.sleep(60)
                
            except Exception as e:
                self.structured_logger.error("System health logging failed", error=str(e))
                await asyncio.sleep(60)
    
    def start_health_monitoring(self) -> asyncio.Task:
        """
        Start background system health monitoring.
        
        Returns:
            Background task handle
        """
        task = asyncio.create_task(self.log_system_health())
        self.structured_logger.info("System health monitoring started")
        return task
    
    def export_logs(self, start_time: datetime, end_time: datetime, 
                   log_types: List[str] = None) -> Dict[str, List[Dict]]:
        """
        Export logs for analysis or compliance reporting.
        
        Args:
            start_time: Start time for log export
            end_time: End time for log export
            log_types: Types of logs to export (audit, trades, performance, system)
        
        Returns:
            Dictionary containing exported log entries by type
        """
        if log_types is None:
            log_types = ["audit", "trades", "performance", "system"]
        
        exported_logs = {}
        
        for log_type in log_types:
            try:
                log_file = f"logs/{log_type}.log"
                if os.path.exists(log_file):
                    exported_logs[log_type] = self._parse_log_file(
                        log_file, start_time, end_time
                    )
                else:
                    exported_logs[log_type] = []
                    
            except Exception as e:
                self.structured_logger.error("Failed to export logs", 
                                           log_type=log_type, error=str(e))
                exported_logs[log_type] = []
        
        return exported_logs
    
    def _parse_log_file(self, log_file: str, start_time: datetime, 
                       end_time: datetime) -> List[Dict]:
        """
        Parse log file and filter by time range.
        
        Args:
            log_file: Path to log file
            start_time: Start time filter
            end_time: End time filter
        
        Returns:
            List of log entries within time range
        """
        entries = []
        
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        entry_time = datetime.fromisoformat(entry.get('asctime', ''))
                        
                        if start_time <= entry_time <= end_time:
                            entries.append(entry)
                            
                    except (json.JSONDecodeError, ValueError, KeyError):
                        # Skip malformed lines
                        continue
                        
        except Exception as e:
            self.structured_logger.error("Failed to parse log file", 
                                       file=log_file, error=str(e))
        
        return entries


# Global logger instance
_logger_system: Optional[TradingSystemLogger] = None


def get_logger_system() -> TradingSystemLogger:
    """Get the global logger system instance."""
    global _logger_system
    
    if _logger_system is None:
        _logger_system = TradingSystemLogger()
    
    return _logger_system


def get_logger(name: str = "system") -> logging.Logger:
    """Get a logger by name."""
    return get_logger_system().get_logger(name)


def get_structured_logger(name: str = None):
    """Get a structured logger instance."""
    return get_logger_system().get_structured_logger(name)


# Convenience functions for common logging operations
def log_trade(trade_data: Dict[str, Any]):
    """Log a trade execution."""
    get_logger_system().log_trade_execution(trade_data)


def log_model_prediction(model_name: str, prediction_data: Dict[str, Any]):
    """Log an AI model prediction."""
    get_logger_system().log_model_prediction(model_name, prediction_data)


def log_security_event(event_type: str, details: Dict[str, Any]):
    """Log a security event."""
    get_logger_system().log_security_event(event_type, details)


def log_performance_metric(metric_name: str, value: float, 
                         context: Optional[Dict[str, Any]] = None):
    """Log a performance metric."""
    get_logger_system().log_performance_metric(metric_name, value, context)


if __name__ == "__main__":
    # Example usage and testing
    async def test_logging_system():
        """Test the logging system functionality."""
        
        print("Testing logging system...")
        
        # Initialize logging system
        logger_system = TradingSystemLogger()
        
        # Test structured logging
        struct_logger = logger_system.get_structured_logger("test")
        struct_logger.info("Testing structured logging", test_value=123, status="active")
        
        # Test trade logging
        sample_trade = {
            "trade_id": "TEST_001",
            "symbol": "BTC/USD",
            "side": "buy",
            "quantity": 0.1,
            "price": 45000.0,
            "exchange": "coinbase",
            "order_type": "market",
            "fees": 4.50,
            "slippage": 0.001
        }
        
        logger_system.log_trade_execution(sample_trade)
        
        # Test model prediction logging
        sample_prediction = {
            "prediction": "buy",
            "confidence": 0.85,
            "features_count": 42,
            "processing_time_ms": 15.2,
            "market_conditions": "volatile"
        }
        
        logger_system.log_model_prediction("executive_auxiliary_agent", sample_prediction)
        
        # Test security event logging
        security_event = {
            "user_id": "system",
            "ip_address": "127.0.0.1",
            "action": "api_key_rotation",
            "resource": "helius_api",
            "success": True
        }
        
        logger_system.log_security_event("api_key_rotation", security_event)
        
        # Test performance metric logging
        logger_system.log_performance_metric("latency_ms", 23.5, {"endpoint": "/api/predict"})
        
        print("Testing system health monitoring...")
        
        # Start health monitoring for a few seconds
        health_task = logger_system.start_health_monitoring()
        await asyncio.sleep(3)  # Let it run for 3 seconds
        health_task.cancel()
        
        print("All logging tests completed! 🎉")
        print("Check the 'logs/' directory for generated log files.")
    
    # Run tests
    asyncio.run(test_logging_system())