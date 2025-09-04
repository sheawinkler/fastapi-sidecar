"""
Data processing pipeline for the crypto trading system.
Orchestrates data collection, processing, and storage across all sources.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
import time
from dataclasses import dataclass

from ..collectors.base import BaseDataCollector, DataCollectionResult
from ..collectors.exchange import ExchangeCollector, TechnicalIndicatorCalculator
from ..collectors.helius import HeliusRPCCollector, SolanaNetworkMonitor
from ..collectors.sentiment import SentimentCollector
from ..storage.database import DatabaseManager
from ..models import (
    DataSource, OHLCV, ProcessedData, TechnicalIndicators,
    SentimentData, OnChainData, MarketMetrics, DataQualityMetrics
)
from ...utils.logger import get_logger
from ...security.credential_manager import CredentialManager


@dataclass
class PipelineConfig:
    """Configuration for the data processing pipeline."""
    # Collection intervals (in seconds)
    ohlcv_interval: int = 60        # 1 minute
    orderbook_interval: int = 5     # 5 seconds
    sentiment_interval: int = 300   # 5 minutes
    onchain_interval: int = 600     # 10 minutes
    
    # Data processing settings
    max_concurrent_tasks: int = 10
    batch_size: int = 100
    error_threshold: float = 0.1    # 10% error rate threshold
    
    # Enabled data sources
    enabled_exchanges: List[DataSource] = None
    enabled_sentiment_sources: bool = True
    enabled_onchain_sources: bool = True
    
    # Symbol configuration
    primary_symbols: List[str] = None
    timeframes: List[str] = None
    
    def __post_init__(self):
        if self.enabled_exchanges is None:
            self.enabled_exchanges = [DataSource.COINBASE, DataSource.KRAKEN]
        
        if self.primary_symbols is None:
            self.primary_symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD']
        
        if self.timeframes is None:
            self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']


class DataProcessingPipeline:
    """
    Comprehensive data processing pipeline orchestrator.
    Manages all data collection, processing, and storage operations.
    """
    
    def __init__(self, config: PipelineConfig, credential_manager: CredentialManager):
        self.config = config
        self.credential_manager = credential_manager
        self.logger = get_logger("data_pipeline")
        
        # Core components
        self.database_manager = None
        self.collectors: Dict[DataSource, BaseDataCollector] = {}
        self.technical_calculator = TechnicalIndicatorCalculator()
        
        # Pipeline state
        self.running = False
        self.tasks: Set[asyncio.Task] = set()
        self.last_collection_times: Dict[str, datetime] = {}
        self.collection_stats: Dict[str, Dict[str, Any]] = {}
        
        # Performance monitoring
        self.performance_metrics = {
            'total_collections': 0,
            'successful_collections': 0,
            'failed_collections': 0,
            'average_processing_time': 0.0,
            'data_quality_score': 1.0,
            'last_update': datetime.utcnow()
        }
    
    async def initialize(self) -> None:
        """Initialize the data processing pipeline."""
        try:
            self.logger.info("Initializing data processing pipeline...")
            
            # Initialize database manager
            self.database_manager = DatabaseManager(self.credential_manager)
            await self.database_manager.initialize()
            
            # Initialize data collectors
            await self._initialize_collectors()
            
            # Initialize collection stats
            self._initialize_collection_stats()
            
            self.logger.info("Data processing pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize data pipeline: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up pipeline resources."""
        try:
            # Stop all running tasks
            await self.stop()
            
            # Clean up collectors
            for collector in self.collectors.values():
                await collector.cleanup()
            
            # Clean up database manager
            if self.database_manager:
                await self.database_manager.cleanup()
            
            self.logger.info("Data pipeline cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Pipeline cleanup error: {e}")
    
    async def _initialize_collectors(self) -> None:
        """Initialize all data collectors."""
        # Initialize exchange collectors
        for exchange in self.config.enabled_exchanges:
            try:
                collector_config = self._create_collector_config(exchange)
                collector = ExchangeCollector(collector_config, self.credential_manager)
                await collector.initialize()
                self.collectors[exchange] = collector
                
                self.logger.info(f"Initialized {exchange.value} collector")
                
            except Exception as e:
                self.logger.warning(f"Failed to initialize {exchange.value} collector: {e}")
        
        # Initialize Helius (Solana on-chain) collector
        if self.config.enabled_onchain_sources:
            try:
                helius_config = self._create_collector_config(DataSource.HELIUS)
                helius_collector = HeliusRPCCollector(helius_config, self.credential_manager)
                await helius_collector.initialize()
                self.collectors[DataSource.HELIUS] = helius_collector
                
                self.logger.info("Initialized Helius RPC collector")
                
            except Exception as e:
                self.logger.warning(f"Failed to initialize Helius collector: {e}")
        
        # Initialize sentiment collector
        if self.config.enabled_sentiment_sources:
            try:
                sentiment_config = self._create_collector_config(DataSource.TWITTER)  # Generic config
                sentiment_collector = SentimentCollector(sentiment_config, self.credential_manager)
                await sentiment_collector.initialize()
                self.collectors[DataSource.TWITTER] = sentiment_collector  # Use as key for sentiment
                
                self.logger.info("Initialized sentiment collector")
                
            except Exception as e:
                self.logger.warning(f"Failed to initialize sentiment collector: {e}")
    
    def _create_collector_config(self, source: DataSource):
        """Create collector configuration for a data source."""
        from ..collectors.base import CollectorConfig
        
        # Base configuration
        config = CollectorConfig(
            source=source,
            symbols=self.config.primary_symbols,
            timeframes=self.config.timeframes,
            rate_limit=100,  # Will be adjusted per source
            max_retries=3,
            retry_delay=1.0,
            timeout=30.0,
            batch_size=self.config.batch_size
        )
        
        # Source-specific adjustments
        if source == DataSource.COINBASE:
            config.rate_limit = 10  # Coinbase has stricter limits
        elif source == DataSource.KRAKEN:
            config.rate_limit = 20
        elif source == DataSource.HELIUS:
            config.rate_limit = 100  # Helius $50 plan
            config.symbols = ['SOL', 'USDC', 'USDT', 'RAY', 'SRM', 'ORCA']
        elif source == DataSource.TWITTER:
            config.rate_limit = 300  # Twitter API v2 limits
        
        return config
    
    def _initialize_collection_stats(self) -> None:
        """Initialize collection statistics tracking."""
        for source in self.collectors:
            self.collection_stats[source.value] = {
                'last_collection': None,
                'successful_collections': 0,
                'failed_collections': 0,
                'total_data_points': 0,
                'average_latency': 0.0,
                'error_rate': 0.0,
                'data_quality_score': 1.0
            }
    
    async def start(self) -> None:
        """Start the data processing pipeline."""
        if self.running:
            self.logger.warning("Pipeline is already running")
            return
        
        self.running = True
        self.logger.info("Starting data processing pipeline...")
        
        try:
            # Start collection tasks
            self.tasks.add(asyncio.create_task(self._ohlcv_collection_loop()))
            self.tasks.add(asyncio.create_task(self._sentiment_collection_loop()))
            
            if DataSource.HELIUS in self.collectors:
                self.tasks.add(asyncio.create_task(self._onchain_collection_loop()))
            
            # Start monitoring and maintenance tasks
            self.tasks.add(asyncio.create_task(self._health_monitoring_loop()))
            self.tasks.add(asyncio.create_task(self._data_cleanup_loop()))
            
            self.logger.info(f"Started {len(self.tasks)} pipeline tasks")
            
            # Wait for all tasks to complete (or until stopped)
            await asyncio.gather(*self.tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"Pipeline execution error: {e}")
            await self.stop()
    
    async def stop(self) -> None:
        """Stop the data processing pipeline."""
        if not self.running:
            return
        
        self.running = False
        self.logger.info("Stopping data processing pipeline...")
        
        # Cancel all running tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete cancellation
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        self.tasks.clear()
        self.logger.info("Data processing pipeline stopped")
    
    async def _ohlcv_collection_loop(self) -> None:
        """Main OHLCV data collection loop."""
        self.logger.info("Started OHLCV collection loop")
        
        while self.running:
            try:
                start_time = time.time()
                
                # Collect from all exchange sources
                collection_tasks = []
                
                for source, collector in self.collectors.items():
                    if source in self.config.enabled_exchanges:
                        for timeframe in self.config.timeframes:
                            task = self._collect_ohlcv_for_timeframe(
                                collector, source, timeframe
                            )
                            collection_tasks.append(task)
                
                # Execute all collection tasks concurrently
                results = await asyncio.gather(*collection_tasks, return_exceptions=True)
                
                # Process results and update statistics
                await self._process_collection_results(results, 'ohlcv')
                
                processing_time = time.time() - start_time
                self._update_performance_metrics(processing_time, len(results))
                
                # Wait until next collection interval
                await asyncio.sleep(max(0, self.config.ohlcv_interval - processing_time))
                
            except Exception as e:
                self.logger.error(f"OHLCV collection loop error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _collect_ohlcv_for_timeframe(
        self, 
        collector: BaseDataCollector, 
        source: DataSource, 
        timeframe: str
    ) -> Dict[str, Any]:
        """Collect OHLCV data for a specific timeframe."""
        try:
            all_data = []
            
            for symbol in self.config.primary_symbols:
                try:
                    # Collect OHLCV data
                    ohlcv_data = await collector.collect_ohlcv(
                        symbol, timeframe, limit=100
                    )
                    
                    if ohlcv_data:
                        # Calculate technical indicators
                        technical_indicators = self.technical_calculator.calculate_all_indicators(
                            ohlcv_data, symbol, timeframe
                        )
                        
                        # Store OHLCV data
                        await self.database_manager.store_ohlcv_data(ohlcv_data)
                        
                        # Store technical indicators
                        if technical_indicators:
                            await self.database_manager.store_technical_indicators(
                                technical_indicators
                            )
                        
                        all_data.extend(ohlcv_data)
                
                except Exception as e:
                    self.logger.error(f"Error collecting {symbol} {timeframe} from {source.value}: {e}")
                    continue
            
            return {
                'success': True,
                'source': source.value,
                'timeframe': timeframe,
                'data_points': len(all_data),
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            return {
                'success': False,
                'source': source.value,
                'timeframe': timeframe,
                'error': str(e),
                'timestamp': datetime.utcnow()
            }
    
    async def _sentiment_collection_loop(self) -> None:
        """Sentiment data collection loop."""
        if DataSource.TWITTER not in self.collectors:
            return
        
        self.logger.info("Started sentiment collection loop")
        sentiment_collector = self.collectors[DataSource.TWITTER]
        
        while self.running:
            try:
                start_time = time.time()
                
                # Collect sentiment for all symbols
                for symbol in ['BTC', 'ETH', 'SOL', 'ADA']:  # Use base symbols for sentiment
                    try:
                        sentiment_data = await sentiment_collector.collect_sentiment(
                            symbol, limit=50
                        )
                        
                        if sentiment_data:
                            await self.database_manager.store_sentiment_data(sentiment_data)
                            
                            self.logger.debug(
                                f"Collected {len(sentiment_data)} sentiment points for {symbol}"
                            )
                    
                    except Exception as e:
                        self.logger.error(f"Sentiment collection error for {symbol}: {e}")
                        continue
                
                # Collect Fear & Greed Index
                try:
                    fear_greed = await sentiment_collector.get_fear_greed_index()
                    if fear_greed:
                        # Store in market metrics (implement if needed)
                        pass
                except Exception as e:
                    self.logger.warning(f"Fear & Greed collection error: {e}")
                
                processing_time = time.time() - start_time
                
                # Wait until next collection interval
                await asyncio.sleep(max(0, self.config.sentiment_interval - processing_time))
                
            except Exception as e:
                self.logger.error(f"Sentiment collection loop error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    async def _onchain_collection_loop(self) -> None:
        """On-chain data collection loop."""
        if DataSource.HELIUS not in self.collectors:
            return
        
        self.logger.info("Started on-chain collection loop")
        helius_collector = self.collectors[DataSource.HELIUS]
        
        while self.running:
            try:
                start_time = time.time()
                
                # Collect on-chain data for Solana tokens
                onchain_data_list = []
                
                for symbol in ['SOL', 'USDC', 'USDT', 'RAY']:
                    try:
                        onchain_data = await helius_collector.collect_onchain_data(
                            symbol, 'solana'
                        )
                        
                        if onchain_data:
                            onchain_data_list.append(onchain_data)
                    
                    except Exception as e:
                        self.logger.error(f"On-chain collection error for {symbol}: {e}")
                        continue
                
                # Store on-chain data
                if onchain_data_list:
                    await self.database_manager.store_onchain_data(onchain_data_list)
                    
                    self.logger.debug(f"Collected on-chain data for {len(onchain_data_list)} tokens")
                
                # Collect whale movements
                try:
                    whale_movements = await helius_collector.collect_whale_movements(
                        'solana', min_value=500000  # $500K threshold
                    )
                    
                    if whale_movements:
                        self.logger.info(f"Detected {len(whale_movements)} whale movements")
                        
                        # Store whale movements (implement specific storage if needed)
                        # For now, they're included in onchain_data.whale_movements
                
                except Exception as e:
                    self.logger.warning(f"Whale movement detection error: {e}")
                
                processing_time = time.time() - start_time
                
                # Wait until next collection interval
                await asyncio.sleep(max(0, self.config.onchain_interval - processing_time))
                
            except Exception as e:
                self.logger.error(f"On-chain collection loop error: {e}")
                await asyncio.sleep(600)  # Wait 10 minutes before retrying
    
    async def _health_monitoring_loop(self) -> None:
        """Health monitoring and alerting loop."""
        self.logger.info("Started health monitoring loop")
        
        while self.running:
            try:
                # Check collector health
                for source, collector in self.collectors.items():
                    health_status = collector.get_health_status()
                    
                    # Update collection statistics
                    stats = self.collection_stats[source.value]
                    stats.update({
                        'error_rate': health_status['error_rate'],
                        'average_latency': health_status['average_latency_ms'],
                        'data_quality_score': health_status['data_quality_score']
                    })
                    
                    # Alert on unhealthy collectors
                    if health_status['status'] == 'unhealthy':
                        self.logger.warning(
                            f"{source.value} collector unhealthy: "
                            f"error_rate={health_status['error_rate']:.2%}, "
                            f"latency={health_status['average_latency_ms']:.1f}ms"
                        )
                
                # Check database health
                db_health = await self.database_manager.get_database_health()
                
                for component, status in db_health.items():
                    if isinstance(status, str) and status != 'healthy':
                        self.logger.warning(f"Database component {component} status: {status}")
                
                # Log overall pipeline health
                self._log_pipeline_health()
                
                # Wait 5 minutes between health checks
                await asyncio.sleep(300)
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(300)
    
    async def _data_cleanup_loop(self) -> None:
        """Data cleanup and maintenance loop."""
        self.logger.info("Started data cleanup loop")
        
        while self.running:
            try:
                # Run cleanup every 24 hours
                await asyncio.sleep(86400)
                
                if not self.running:
                    break
                
                self.logger.info("Running data cleanup...")
                await self.database_manager.cleanup_old_data()
                
                # Optimize Helius polling intervals
                if DataSource.HELIUS in self.collectors:
                    helius_collector = self.collectors[DataSource.HELIUS]
                    await helius_collector.optimize_polling_intervals()
                
                self.logger.info("Data cleanup completed")
                
            except Exception as e:
                self.logger.error(f"Data cleanup error: {e}")
    
    async def _process_collection_results(
        self, 
        results: List[Any], 
        collection_type: str
    ) -> None:
        """Process and analyze collection results."""
        successful = 0
        failed = 0
        total_data_points = 0
        
        for result in results:
            if isinstance(result, dict):
                if result.get('success', False):
                    successful += 1
                    total_data_points += result.get('data_points', 0)
                else:
                    failed += 1
                    self.logger.error(f"Collection failed: {result.get('error', 'Unknown error')}")
            elif isinstance(result, Exception):
                failed += 1
                self.logger.error(f"Collection exception: {result}")
            else:
                failed += 1
        
        # Update statistics
        total = successful + failed
        if total > 0:
            error_rate = failed / total
            
            if error_rate > self.config.error_threshold:
                self.logger.warning(
                    f"High error rate in {collection_type} collection: "
                    f"{error_rate:.2%} ({failed}/{total})"
                )
        
        # Update performance metrics
        self.performance_metrics.update({
            'total_collections': self.performance_metrics['total_collections'] + total,
            'successful_collections': self.performance_metrics['successful_collections'] + successful,
            'failed_collections': self.performance_metrics['failed_collections'] + failed,
            'last_update': datetime.utcnow()
        })
    
    def _update_performance_metrics(self, processing_time: float, task_count: int) -> None:
        """Update pipeline performance metrics."""
        # Update average processing time using exponential moving average
        alpha = 0.1
        current_avg = self.performance_metrics['average_processing_time']
        
        if current_avg == 0:
            self.performance_metrics['average_processing_time'] = processing_time
        else:
            self.performance_metrics['average_processing_time'] = (
                alpha * processing_time + (1 - alpha) * current_avg
            )
        
        # Calculate overall data quality score
        total_quality_score = 0
        quality_count = 0
        
        for stats in self.collection_stats.values():
            if stats['data_quality_score'] > 0:
                total_quality_score += stats['data_quality_score']
                quality_count += 1
        
        if quality_count > 0:
            self.performance_metrics['data_quality_score'] = total_quality_score / quality_count
    
    def _log_pipeline_health(self) -> None:
        """Log comprehensive pipeline health information."""
        metrics = self.performance_metrics
        
        # Calculate success rate
        total_collections = metrics['total_collections']
        success_rate = (
            metrics['successful_collections'] / total_collections
            if total_collections > 0 else 1.0
        )
        
        self.logger.info(
            f"Pipeline Health - Success Rate: {success_rate:.2%}, "
            f"Avg Processing Time: {metrics['average_processing_time']:.2f}s, "
            f"Data Quality: {metrics['data_quality_score']:.3f}, "
            f"Total Collections: {total_collections}"
        )
        
        # Log individual collector status
        for source, stats in self.collection_stats.items():
            if stats['last_collection']:
                time_since = (datetime.utcnow() - stats['last_collection']).total_seconds()
                self.logger.debug(
                    f"{source} - Error Rate: {stats['error_rate']:.2%}, "
                    f"Latency: {stats['average_latency']:.1f}ms, "
                    f"Last Collection: {time_since:.0f}s ago"
                )
    
    # Public API methods
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status."""
        return {
            'running': self.running,
            'active_tasks': len(self.tasks),
            'performance_metrics': self.performance_metrics,
            'collection_stats': self.collection_stats,
            'collector_count': len(self.collectors),
            'enabled_sources': [source.value for source in self.collectors.keys()],
            'last_health_check': datetime.utcnow().isoformat()
        }
    
    async def force_collection(self, source: str, data_type: str = 'ohlcv') -> Dict[str, Any]:
        """Force immediate data collection from a specific source."""
        try:
            source_enum = DataSource(source)
            
            if source_enum not in self.collectors:
                return {'success': False, 'error': f'Source {source} not available'}
            
            collector = self.collectors[source_enum]
            
            if data_type == 'ohlcv':
                result = await self._collect_ohlcv_for_timeframe(
                    collector, source_enum, '5m'
                )
            else:
                return {'success': False, 'error': f'Data type {data_type} not supported'}
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def get_data_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of collected data for the last N hours."""
        try:
            # This would query the database for recent data statistics
            # Implementation depends on specific database queries needed
            
            summary = {
                'timeframe_hours': hours,
                'total_ohlcv_records': 0,
                'total_sentiment_records': 0,
                'total_onchain_records': 0,
                'symbols_covered': set(),
                'sources_active': len(self.collectors),
                'data_quality_average': self.performance_metrics['data_quality_score']
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating data summary: {e}")
            return {'error': str(e)}