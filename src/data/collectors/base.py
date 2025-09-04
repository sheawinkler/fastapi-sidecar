"""
Base classes for data collectors.
Enterprise-grade abstract interfaces for all data collection components.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, AsyncGenerator, Any, Union
from dataclasses import dataclass
import time

from aiolimiter import AsyncLimiter
from ..models import (
    DataSource, MarketDataType, OHLCV, OrderBook, 
    SentimentData, OnChainData, TechnicalIndicators,
    DataQualityMetrics, ProcessedData
)
from ...utils.logger import get_logger
from ...security.credential_manager import CredentialManager


@dataclass
class CollectorConfig:
    """Configuration for data collectors."""
    source: DataSource
    symbols: List[str]
    timeframes: List[str]
    rate_limit: int  # requests per second
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    batch_size: int = 100
    cache_ttl: int = 60  # seconds
    enable_websocket: bool = True
    enable_rest: bool = True


@dataclass
class DataCollectionResult:
    """Result from data collection operation."""
    success: bool
    data: List[Any]
    errors: List[str]
    metrics: Dict[str, Any]
    timestamp: datetime
    collection_time_ms: float
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        total_attempts = len(self.data) + len(self.errors)
        return len(self.errors) / total_attempts if total_attempts > 0 else 0.0


class BaseDataCollector(ABC):
    """
    Abstract base class for all data collectors.
    Provides common functionality for rate limiting, error handling, and logging.
    """
    
    def __init__(self, config: CollectorConfig, credential_manager: CredentialManager):
        self.config = config
        self.credential_manager = credential_manager
        self.logger = get_logger(f"collector.{config.source.value}")
        
        # Rate limiting
        self.rate_limiter = AsyncLimiter(config.rate_limit, 1.0)
        
        # Connection management
        self._session = None
        self._websocket_connections = {}
        self._last_error_time = {}
        
        # Metrics tracking
        self.metrics = {
            'requests_made': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_data_points': 0,
            'average_latency_ms': 0.0,
            'last_collection_time': None
        }
        
        # Data quality tracking
        self.quality_metrics = DataQualityMetrics(
            timestamp=datetime.utcnow(),
            source=config.source,
            symbol="",
            completeness_score=1.0,
            accuracy_score=1.0,
            freshness_score=1.0,
            consistency_score=1.0,
            overall_score=1.0
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the collector (connections, authentication, etc.)."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources (close connections, etc.)."""
        pass
    
    @abstractmethod
    async def collect_ohlcv(
        self, 
        symbol: str, 
        timeframe: str, 
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[OHLCV]:
        """Collect OHLCV data for a symbol."""
        pass
    
    @abstractmethod
    async def collect_orderbook(self, symbol: str, limit: Optional[int] = None) -> Optional[OrderBook]:
        """Collect order book data for a symbol."""
        pass
    
    @abstractmethod
    async def get_supported_symbols(self) -> List[str]:
        """Get list of supported trading symbols."""
        pass
    
    @abstractmethod
    async def get_supported_timeframes(self) -> List[str]:
        """Get list of supported timeframes."""
        pass
    
    async def collect_with_retry(self, operation, *args, **kwargs) -> Any:
        """Execute operation with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                async with self.rate_limiter:
                    start_time = time.time()
                    result = await operation(*args, **kwargs)
                    
                    # Update metrics
                    latency_ms = (time.time() - start_time) * 1000
                    self._update_success_metrics(latency_ms)
                    
                    return result
                    
            except Exception as e:
                last_exception = e
                self._update_error_metrics(str(e))
                
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"All {self.config.max_retries + 1} attempts failed: {e}")
                    break
        
        raise last_exception if last_exception else Exception("Unknown error in retry logic")
    
    def _update_success_metrics(self, latency_ms: float) -> None:
        """Update success metrics."""
        self.metrics['requests_made'] += 1
        self.metrics['successful_requests'] += 1
        
        # Update average latency using exponential moving average
        alpha = 0.1
        if self.metrics['average_latency_ms'] == 0:
            self.metrics['average_latency_ms'] = latency_ms
        else:
            self.metrics['average_latency_ms'] = (
                alpha * latency_ms + 
                (1 - alpha) * self.metrics['average_latency_ms']
            )
        
        self.metrics['last_collection_time'] = datetime.utcnow()
    
    def _update_error_metrics(self, error_message: str) -> None:
        """Update error metrics."""
        self.metrics['requests_made'] += 1
        self.metrics['failed_requests'] += 1
        
        error_time = datetime.utcnow()
        self._last_error_time[error_message] = error_time
        
        # Log structured error information
        self.logger.error(
            "Data collection error",
            extra={
                'source': self.config.source.value,
                'error': error_message,
                'error_rate': self.get_error_rate(),
                'timestamp': error_time.isoformat()
            }
        )
    
    def get_error_rate(self) -> float:
        """Calculate current error rate."""
        total_requests = self.metrics['requests_made']
        failed_requests = self.metrics['failed_requests']
        return failed_requests / total_requests if total_requests > 0 else 0.0
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get collector health status."""
        error_rate = self.get_error_rate()
        avg_latency = self.metrics['average_latency_ms']
        
        # Determine health status
        if error_rate > 0.2 or avg_latency > 5000:  # 20% error rate or >5s latency
            status = "unhealthy"
        elif error_rate > 0.1 or avg_latency > 2000:  # 10% error rate or >2s latency
            status = "degraded"
        else:
            status = "healthy"
        
        return {
            'status': status,
            'error_rate': error_rate,
            'average_latency_ms': avg_latency,
            'total_requests': self.metrics['requests_made'],
            'successful_requests': self.metrics['successful_requests'],
            'failed_requests': self.metrics['failed_requests'],
            'last_collection_time': self.metrics['last_collection_time'],
            'data_quality_score': self.quality_metrics.overall_score
        }
    
    async def validate_data_quality(self, data: List[Any]) -> DataQualityMetrics:
        """Validate data quality and return metrics."""
        timestamp = datetime.utcnow()
        
        if not data:
            return DataQualityMetrics(
                timestamp=timestamp,
                source=self.config.source,
                symbol="",
                completeness_score=0.0,
                accuracy_score=0.0,
                freshness_score=0.0,
                consistency_score=0.0,
                overall_score=0.0,
                missing_fields=["all_data"]
            )
        
        # Completeness check
        expected_fields = self._get_expected_fields(data[0])
        missing_fields = []
        complete_records = 0
        
        for record in data:
            record_missing = []
            for field in expected_fields:
                if not hasattr(record, field) or getattr(record, field) is None:
                    record_missing.append(field)
            
            if not record_missing:
                complete_records += 1
            else:
                missing_fields.extend(record_missing)
        
        completeness_score = complete_records / len(data)
        
        # Freshness check (data should be recent)
        freshness_scores = []
        for record in data:
            if hasattr(record, 'timestamp'):
                age_minutes = (timestamp - record.timestamp).total_seconds() / 60
                # Data older than 1 hour gets lower freshness score
                freshness = max(0.0, 1.0 - (age_minutes / 60))
                freshness_scores.append(freshness)
        
        freshness_score = sum(freshness_scores) / len(freshness_scores) if freshness_scores else 1.0
        
        # Accuracy check (basic data validation)
        valid_records = 0
        for record in data:
            try:
                # This will raise an exception if data validation fails
                if hasattr(record, '__post_init__'):
                    record.__post_init__()
                valid_records += 1
            except Exception as e:
                self.logger.warning(f"Data validation failed for record: {e}")
        
        accuracy_score = valid_records / len(data)
        
        # Consistency check (data should be logically consistent)
        consistency_score = self._check_data_consistency(data)
        
        quality_metrics = DataQualityMetrics(
            timestamp=timestamp,
            source=self.config.source,
            symbol=getattr(data[0], 'symbol', ''),
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            freshness_score=freshness_score,
            consistency_score=consistency_score,
            overall_score=0.0,  # Will be calculated in __post_init__
            missing_fields=list(set(missing_fields)),
            outliers_detected=self._detect_outliers(data)
        )
        
        self.quality_metrics = quality_metrics
        return quality_metrics
    
    def _get_expected_fields(self, record: Any) -> List[str]:
        """Get expected fields for a record type."""
        if hasattr(record, '__dataclass_fields__'):
            return list(record.__dataclass_fields__.keys())
        return []
    
    def _check_data_consistency(self, data: List[Any]) -> float:
        """Check logical consistency of data."""
        if len(data) < 2:
            return 1.0
        
        consistency_issues = 0
        total_checks = 0
        
        # For OHLCV data, check that timestamps are in order
        if data and hasattr(data[0], 'timestamp'):
            for i in range(1, len(data)):
                total_checks += 1
                if data[i].timestamp < data[i-1].timestamp:
                    consistency_issues += 1
        
        # For price data, check for extreme price movements (>50% in one candle)
        if data and hasattr(data[0], 'close'):
            for i in range(1, len(data)):
                total_checks += 1
                prev_close = data[i-1].close
                curr_close = data[i].close
                if abs(curr_close - prev_close) / prev_close > 0.5:
                    consistency_issues += 1
        
        return max(0.0, 1.0 - (consistency_issues / total_checks)) if total_checks > 0 else 1.0
    
    def _detect_outliers(self, data: List[Any]) -> int:
        """Detect outliers in data using simple statistical methods."""
        if len(data) < 10:  # Need sufficient data for outlier detection
            return 0
        
        outliers = 0
        
        # For numeric fields, use IQR method
        numeric_fields = ['close', 'volume', 'sentiment_score']
        
        for field in numeric_fields:
            if not hasattr(data[0], field):
                continue
                
            values = [getattr(record, field) for record in data 
                     if hasattr(record, field) and getattr(record, field) is not None]
            
            if len(values) < 10:
                continue
            
            # Calculate IQR
            sorted_values = sorted(values)
            q1_idx = len(sorted_values) // 4
            q3_idx = 3 * len(sorted_values) // 4
            
            q1 = sorted_values[q1_idx]
            q3 = sorted_values[q3_idx]
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            for value in values:
                if value < lower_bound or value > upper_bound:
                    outliers += 1
        
        return outliers


class BaseMarketDataCollector(BaseDataCollector):
    """Base class for market data collectors (exchanges)."""
    
    @abstractmethod
    async def collect_ticker(self, symbol: str) -> Dict[str, Any]:
        """Collect ticker data for a symbol."""
        pass
    
    @abstractmethod
    async def collect_trades(
        self, 
        symbol: str, 
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Collect recent trades for a symbol."""
        pass
    
    async def collect_all_symbols(self, timeframe: str) -> DataCollectionResult:
        """Collect OHLCV data for all configured symbols."""
        start_time = time.time()
        all_data = []
        errors = []
        
        for symbol in self.config.symbols:
            try:
                ohlcv_data = await self.collect_ohlcv(symbol, timeframe, limit=100)
                all_data.extend(ohlcv_data)
                self.logger.debug(f"Collected {len(ohlcv_data)} candles for {symbol}")
            except Exception as e:
                error_msg = f"Failed to collect {symbol} data: {str(e)}"
                errors.append(error_msg)
                self.logger.error(error_msg)
        
        collection_time = (time.time() - start_time) * 1000
        
        return DataCollectionResult(
            success=len(errors) == 0,
            data=all_data,
            errors=errors,
            metrics={
                'symbols_processed': len(self.config.symbols),
                'successful_symbols': len(self.config.symbols) - len(errors),
                'total_data_points': len(all_data),
                'error_rate': len(errors) / len(self.config.symbols) if self.config.symbols else 0
            },
            timestamp=datetime.utcnow(),
            collection_time_ms=collection_time
        )


class BaseSentimentCollector(BaseDataCollector):
    """Base class for sentiment data collectors."""
    
    @abstractmethod
    async def collect_sentiment(
        self, 
        symbol: str, 
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[SentimentData]:
        """Collect sentiment data for a symbol."""
        pass
    
    @abstractmethod
    async def get_trending_topics(self, limit: int = 10) -> List[str]:
        """Get trending topics related to crypto."""
        pass


class BaseOnChainCollector(BaseDataCollector):
    """Base class for on-chain data collectors."""
    
    @abstractmethod
    async def collect_onchain_data(
        self, 
        symbol: str, 
        network: str,
        since: Optional[datetime] = None
    ) -> Optional[OnChainData]:
        """Collect on-chain analytics data."""
        pass
    
    @abstractmethod
    async def collect_whale_movements(
        self, 
        network: str,
        min_value: float = 1000000  # $1M minimum
    ) -> List[Dict[str, Any]]:
        """Collect large transaction movements."""
        pass