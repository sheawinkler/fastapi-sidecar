"""
Data orchestrator for the crypto trading system.
Main entry point for all data operations and management.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import signal
import sys

from .processors.pipeline import DataProcessingPipeline, PipelineConfig
from .collectors.exchange import ExchangeCollector
from .collectors.helius import HeliusRPCCollector
from .collectors.sentiment import SentimentCollector
from .storage.database import DatabaseManager
from .models import DataSource, OHLCV, ProcessedData
from ..utils.logger import get_logger
from ..security.credential_manager import CredentialManager
from ..utils.config import ConfigManager


class DataOrchestrator:
    """
    Main data orchestration class.
    Coordinates all data collection, processing, and storage operations.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_manager = ConfigManager(config_path)
        self.credential_manager = CredentialManager()
        self.logger = get_logger("data_orchestrator")
        
        # Core components
        self.pipeline = None
        self.database_manager = None
        
        # State management
        self.running = False
        self.shutdown_event = asyncio.Event()
        
        # Configuration
        self.pipeline_config = None
        
    async def initialize(self) -> None:
        """Initialize the data orchestrator."""
        try:
            self.logger.info("Initializing Data Orchestrator...")
            
            # Load configuration
            await self._load_configuration()
            
            # Initialize credential manager
            await self.credential_manager.initialize()
            
            # Create pipeline configuration
            self.pipeline_config = PipelineConfig(
                ohlcv_interval=self.config_manager.get('data.collection.ohlcv_interval', 60),
                sentiment_interval=self.config_manager.get('data.collection.sentiment_interval', 300),
                onchain_interval=self.config_manager.get('data.collection.onchain_interval', 600),
                max_concurrent_tasks=self.config_manager.get('data.processing.max_concurrent_tasks', 10),
                batch_size=self.config_manager.get('data.processing.batch_size', 100),
                primary_symbols=self.config_manager.get('trading.symbols', ['BTC/USD', 'ETH/USD', 'SOL/USD']),
                timeframes=self.config_manager.get('trading.timeframes', ['1m', '5m', '15m', '1h', '4h', '1d']),
                enabled_exchanges=[
                    DataSource(source) for source in 
                    self.config_manager.get('data.sources.exchanges', ['coinbase', 'kraken'])
                ],
                enabled_sentiment_sources=self.config_manager.get('data.sources.sentiment_enabled', True),
                enabled_onchain_sources=self.config_manager.get('data.sources.onchain_enabled', True)
            )
            
            # Initialize data processing pipeline
            self.pipeline = DataProcessingPipeline(self.pipeline_config, self.credential_manager)
            await self.pipeline.initialize()
            
            # Initialize database manager separately for API access
            self.database_manager = DatabaseManager(self.credential_manager)
            await self.database_manager.initialize()
            
            # Set up signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            self.logger.info("Data Orchestrator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Data Orchestrator: {e}")
            raise
    
    async def _load_configuration(self) -> None:
        """Load and validate configuration."""
        try:
            # Load main configuration
            await self.config_manager.load_config()
            
            # Validate required configuration sections
            required_sections = ['data', 'trading', 'security']
            for section in required_sections:
                if not self.config_manager.has_section(section):
                    self.logger.warning(f"Missing configuration section: {section}")
            
            # Set default values for missing configurations
            self._set_default_configuration()
            
        except Exception as e:
            self.logger.error(f"Configuration loading failed: {e}")
            raise
    
    def _set_default_configuration(self) -> None:
        """Set default configuration values."""
        defaults = {
            'data.collection.ohlcv_interval': 60,
            'data.collection.sentiment_interval': 300,
            'data.collection.onchain_interval': 600,
            'data.processing.max_concurrent_tasks': 10,
            'data.processing.batch_size': 100,
            'data.storage.retention_days': 365,
            'trading.symbols': ['BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD'],
            'trading.timeframes': ['1m', '5m', '15m', '1h', '4h', '1d'],
            'data.sources.exchanges': ['coinbase', 'kraken'],
            'data.sources.sentiment_enabled': True,
            'data.sources.onchain_enabled': True
        }
        
        for key, value in defaults.items():
            if not self.config_manager.has_key(key):
                self.config_manager.set(key, value)
    
    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        if sys.platform != 'win32':
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
    
    async def start(self) -> None:
        """Start the data orchestrator."""
        if self.running:
            self.logger.warning("Data Orchestrator is already running")
            return
        
        try:
            self.running = True
            self.logger.info("Starting Data Orchestrator...")
            
            # Start the data processing pipeline
            pipeline_task = asyncio.create_task(self.pipeline.start())
            
            # Start monitoring task
            monitor_task = asyncio.create_task(self._monitoring_loop())
            
            # Wait for shutdown signal or pipeline completion
            await asyncio.gather(
                pipeline_task,
                monitor_task,
                self.shutdown_event.wait(),
                return_exceptions=True
            )
            
        except Exception as e:
            self.logger.error(f"Data Orchestrator error: {e}")
            raise
        finally:
            await self.cleanup()
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the data orchestrator."""
        if not self.running:
            return
        
        self.logger.info("Shutting down Data Orchestrator...")
        self.running = False
        self.shutdown_event.set()
        
        # Stop the pipeline
        if self.pipeline:
            await self.pipeline.stop()
        
        await self.cleanup()
        self.logger.info("Data Orchestrator shutdown complete")
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self.pipeline:
                await self.pipeline.cleanup()
            
            if self.database_manager:
                await self.database_manager.cleanup()
            
            await self.credential_manager.cleanup()
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring and health check loop."""
        self.logger.info("Started monitoring loop")
        
        while self.running:
            try:
                # Get pipeline status
                status = await self.pipeline.get_pipeline_status()
                
                # Log health information every 10 minutes
                if datetime.utcnow().minute % 10 == 0:
                    self.logger.info(
                        f"Data Pipeline Status - Running: {status['running']}, "
                        f"Active Tasks: {status['active_tasks']}, "
                        f"Success Rate: {self._calculate_success_rate(status):.2%}"
                    )
                
                # Check for critical issues
                await self._check_critical_issues(status)
                
                # Wait 1 minute between checks
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)
    
    def _calculate_success_rate(self, status: Dict[str, Any]) -> float:
        """Calculate overall success rate from pipeline status."""
        metrics = status.get('performance_metrics', {})
        total = metrics.get('total_collections', 0)
        successful = metrics.get('successful_collections', 0)
        
        return successful / total if total > 0 else 1.0
    
    async def _check_critical_issues(self, status: Dict[str, Any]) -> None:
        """Check for critical issues that require attention."""
        metrics = status.get('performance_metrics', {})
        collection_stats = status.get('collection_stats', {})
        
        # Check overall success rate
        success_rate = self._calculate_success_rate(status)
        if success_rate < 0.7:  # Less than 70% success rate
            self.logger.critical(
                f"Critical: Low success rate detected: {success_rate:.2%}"
            )
        
        # Check individual collector health
        for source, stats in collection_stats.items():
            error_rate = stats.get('error_rate', 0)
            if error_rate > 0.3:  # More than 30% error rate
                self.logger.critical(
                    f"Critical: High error rate for {source}: {error_rate:.2%}"
                )
            
            # Check if collector hasn't collected data recently
            last_collection = stats.get('last_collection')
            if last_collection:
                time_since = (datetime.utcnow() - last_collection).total_seconds()
                if time_since > 3600:  # More than 1 hour
                    self.logger.warning(
                        f"Warning: {source} hasn't collected data for {time_since/60:.0f} minutes"
                    )
    
    # Public API methods
    async def get_latest_data(
        self, 
        symbol: str, 
        timeframe: str = '5m', 
        limit: int = 100
    ) -> List[OHLCV]:
        """Get latest OHLCV data for a symbol."""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=24)  # Last 24 hours
            
            return await self.database_manager.get_ohlcv_data(
                symbol, timeframe, start_time, end_time
            )
            
        except Exception as e:
            self.logger.error(f"Error retrieving latest data for {symbol}: {e}")
            return []
    
    async def get_data_quality_report(self) -> Dict[str, Any]:
        """Get comprehensive data quality report."""
        try:
            status = await self.pipeline.get_pipeline_status()
            
            report = {
                'timestamp': datetime.utcnow().isoformat(),
                'overall_health': 'healthy',
                'pipeline_running': status['running'],
                'active_collectors': len(status.get('enabled_sources', [])),
                'performance_metrics': status.get('performance_metrics', {}),
                'collection_statistics': status.get('collection_stats', {}),
                'database_health': await self.database_manager.get_database_health(),
                'recommendations': []
            }
            
            # Analyze health and add recommendations
            success_rate = self._calculate_success_rate(status)
            
            if success_rate < 0.8:
                report['overall_health'] = 'degraded'
                report['recommendations'].append(
                    f"Success rate is {success_rate:.2%}. Check collector configurations."
                )
            
            if success_rate < 0.5:
                report['overall_health'] = 'critical'
                report['recommendations'].append(
                    "Critical success rate. Immediate attention required."
                )
            
            # Check database health
            db_health = report['database_health']
            unhealthy_components = [
                comp for comp, status in db_health.items() 
                if isinstance(status, str) and status != 'healthy'
            ]
            
            if unhealthy_components:
                report['overall_health'] = 'degraded'
                report['recommendations'].append(
                    f"Database components need attention: {', '.join(unhealthy_components)}"
                )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating data quality report: {e}")
            return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}
    
    async def force_data_collection(
        self, 
        source: str, 
        data_type: str = 'ohlcv'
    ) -> Dict[str, Any]:
        """Force immediate data collection from a specific source."""
        try:
            return await self.pipeline.force_collection(source, data_type)
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def get_supported_symbols(self) -> List[str]:
        """Get list of supported trading symbols."""
        return self.pipeline_config.primary_symbols
    
    async def get_supported_timeframes(self) -> List[str]:
        """Get list of supported timeframes."""
        return self.pipeline_config.timeframes
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            pipeline_status = await self.pipeline.get_pipeline_status()
            db_health = await self.database_manager.get_database_health()
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'orchestrator_running': self.running,
                'pipeline_status': pipeline_status,
                'database_health': db_health,
                'configuration': {
                    'primary_symbols': self.pipeline_config.primary_symbols,
                    'timeframes': self.pipeline_config.timeframes,
                    'collection_intervals': {
                        'ohlcv': self.pipeline_config.ohlcv_interval,
                        'sentiment': self.pipeline_config.sentiment_interval,
                        'onchain': self.pipeline_config.onchain_interval
                    },
                    'enabled_sources': [
                        source.value for source in self.pipeline_config.enabled_exchanges
                    ]
                }
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    # Data analysis methods
    async def analyze_market_trends(
        self, 
        symbol: str, 
        timeframe: str = '1h', 
        hours: int = 24
    ) -> Dict[str, Any]:
        """Analyze market trends for a symbol."""
        try:
            # Get recent OHLCV data
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)
            
            ohlcv_data = await self.database_manager.get_ohlcv_data(
                symbol, timeframe, start_time, end_time
            )
            
            if not ohlcv_data:
                return {'error': 'No data available'}
            
            # Calculate basic trend analysis
            prices = [float(candle.close) for candle in ohlcv_data]
            volumes = [float(candle.volume) for candle in ohlcv_data]
            
            analysis = {
                'symbol': symbol,
                'timeframe': timeframe,
                'period_hours': hours,
                'data_points': len(ohlcv_data),
                'price_change': {
                    'start_price': prices[0] if prices else 0,
                    'end_price': prices[-1] if prices else 0,
                    'absolute_change': prices[-1] - prices[0] if len(prices) >= 2 else 0,
                    'percentage_change': ((prices[-1] - prices[0]) / prices[0] * 100) if len(prices) >= 2 and prices[0] > 0 else 0
                },
                'volume_analysis': {
                    'average_volume': sum(volumes) / len(volumes) if volumes else 0,
                    'total_volume': sum(volumes),
                    'max_volume': max(volumes) if volumes else 0,
                    'min_volume': min(volumes) if volumes else 0
                },
                'volatility': {
                    'price_range': max(prices) - min(prices) if prices else 0,
                    'high_price': max(prices) if prices else 0,
                    'low_price': min(prices) if prices else 0
                }
            }
            
            # Determine trend direction
            if analysis['price_change']['percentage_change'] > 2:
                analysis['trend'] = 'bullish'
            elif analysis['price_change']['percentage_change'] < -2:
                analysis['trend'] = 'bearish'
            else:
                analysis['trend'] = 'sideways'
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Market trend analysis error for {symbol}: {e}")
            return {'error': str(e)}


# Main execution function
async def main():
    """Main function to run the data orchestrator."""
    orchestrator = DataOrchestrator()
    
    try:
        await orchestrator.initialize()
        await orchestrator.start()
    except KeyboardInterrupt:
        print("\nReceived interrupt signal, shutting down...")
        await orchestrator.shutdown()
    except Exception as e:
        print(f"Fatal error: {e}")
        await orchestrator.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    # Run the data orchestrator
    asyncio.run(main())