"""
Database storage system for the crypto trading platform.
Implements InfluxDB for time-series data, PostgreSQL for transactional data, and Redis for caching.
"""

import asyncio
import asyncpg
import aioredis
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from influxdb_client.client.write_api_async import WriteApiAsync
from influxdb_client.client.query_api_async import QueryApiAsync
from influxdb_client import Point, WritePrecision
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
import json
import pickle
import hashlib
from dataclasses import asdict

from ..models import (
    OHLCV, OrderBook, TechnicalIndicators, SentimentData, 
    OnChainData, MarketMetrics, ProcessedData, DataQualityMetrics,
    DataSource
)
from ...utils.logger import get_logger
from ...security.credential_manager import CredentialManager


class DatabaseManager:
    """
    Comprehensive database manager handling all data storage needs.
    Implements proper data partitioning and retention policies.
    """
    
    def __init__(self, credential_manager: CredentialManager):
        self.credential_manager = credential_manager
        self.logger = get_logger("database_manager")
        
        # Database connections
        self.influx_client = None
        self.influx_write_api = None
        self.influx_query_api = None
        self.postgres_pool = None
        self.redis_client = None
        
        # Database configuration
        self.influx_config = {
            'org': 'crypto-trading-org',
            'bucket': 'market-data',
            'token': None,
            'url': 'http://localhost:8086'
        }
        
        # Data retention policies (in days)
        self.retention_policies = {
            'ohlcv_1m': 30,      # 1-minute data: 30 days
            'ohlcv_5m': 90,      # 5-minute data: 90 days  
            'ohlcv_1h': 365,     # 1-hour data: 1 year
            'ohlcv_1d': 1825,    # Daily data: 5 years
            'orderbook': 7,      # Order book: 7 days
            'sentiment': 180,    # Sentiment: 6 months
            'onchain': 365,      # On-chain: 1 year
            'technical': 365     # Technical indicators: 1 year
        }
        
        # Cache TTL settings (in seconds)
        self.cache_ttl = {
            'ohlcv': 60,         # 1 minute
            'orderbook': 5,      # 5 seconds
            'sentiment': 300,    # 5 minutes
            'technical': 300,    # 5 minutes
            'onchain': 600       # 10 minutes
        }
    
    async def initialize(self) -> None:
        """Initialize all database connections."""
        try:
            await self._initialize_influxdb()
            await self._initialize_postgresql()
            await self._initialize_redis()
            
            # Create database schemas
            await self._create_schemas()
            
            self.logger.info("Database manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database manager: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up all database connections."""
        try:
            if self.influx_client:
                await self.influx_client.close()
            
            if self.postgres_pool:
                await self.postgres_pool.close()
            
            if self.redis_client:
                await self.redis_client.close()
            
            self.logger.info("Database connections closed")
            
        except Exception as e:
            self.logger.error(f"Error during database cleanup: {e}")
    
    async def _initialize_influxdb(self) -> None:
        """Initialize InfluxDB connection for time-series data."""
        try:
            # Get InfluxDB credentials
            self.influx_config['token'] = await self.credential_manager.get_credential(
                'INFLUXDB_TOKEN'
            )
            
            url = await self.credential_manager.get_credential('INFLUXDB_URL', 
                                                             self.influx_config['url'])
            
            self.influx_client = InfluxDBClientAsync(
                url=url,
                token=self.influx_config['token'],
                org=self.influx_config['org']
            )
            
            # Initialize write and query APIs
            self.influx_write_api = self.influx_client.write_api()
            self.influx_query_api = self.influx_client.query_api()
            
            # Test connection
            health = await self.influx_client.health()
            if health.status == "pass":
                self.logger.info("InfluxDB connection established successfully")
            else:
                raise Exception(f"InfluxDB health check failed: {health.status}")
                
        except Exception as e:
            self.logger.warning(f"InfluxDB initialization failed: {e}")
            # Continue without InfluxDB - use PostgreSQL for time-series data
            self.influx_client = None
    
    async def _initialize_postgresql(self) -> None:
        """Initialize PostgreSQL connection for relational data."""
        try:
            # Get PostgreSQL credentials
            postgres_config = {
                'host': await self.credential_manager.get_credential('POSTGRES_HOST', 'localhost'),
                'port': int(await self.credential_manager.get_credential('POSTGRES_PORT', '5432')),
                'database': await self.credential_manager.get_credential('POSTGRES_DB', 'crypto_trading'),
                'user': await self.credential_manager.get_credential('POSTGRES_USER', 'postgres'),
                'password': await self.credential_manager.get_credential('POSTGRES_PASSWORD')
            }
            
            # Create connection pool
            self.postgres_pool = await asyncpg.create_pool(
                min_size=5,
                max_size=20,
                **postgres_config
            )
            
            # Test connection
            async with self.postgres_pool.acquire() as conn:
                result = await conn.fetchval('SELECT version()')
                self.logger.info(f"PostgreSQL connected: {result[:50]}...")
                
        except Exception as e:
            self.logger.error(f"PostgreSQL initialization failed: {e}")
            raise
    
    async def _initialize_redis(self) -> None:
        """Initialize Redis connection for caching."""
        try:
            # Get Redis credentials
            redis_url = await self.credential_manager.get_credential(
                'REDIS_URL', 'redis://localhost:6379'
            )
            
            self.redis_client = await aioredis.from_url(
                redis_url,
                decode_responses=True,
                max_connections=20
            )
            
            # Test connection
            await self.redis_client.ping()
            self.logger.info("Redis connection established successfully")
            
        except Exception as e:
            self.logger.warning(f"Redis initialization failed: {e}")
            # Continue without Redis caching
            self.redis_client = None
    
    async def _create_schemas(self) -> None:
        """Create database schemas and tables."""
        try:
            async with self.postgres_pool.acquire() as conn:
                # Create tables for different data types
                
                # Trading pairs table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS trading_pairs (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL UNIQUE,
                        base_asset VARCHAR(10) NOT NULL,
                        quote_asset VARCHAR(10) NOT NULL,
                        active BOOLEAN DEFAULT true,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Data sources table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS data_sources (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(50) NOT NULL UNIQUE,
                        type VARCHAR(20) NOT NULL,
                        active BOOLEAN DEFAULT true,
                        rate_limit INTEGER DEFAULT 100,
                        last_request TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # OHLCV data table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS ohlcv_data (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        timeframe VARCHAR(10) NOT NULL,
                        open_price DECIMAL(20, 8) NOT NULL,
                        high_price DECIMAL(20, 8) NOT NULL,
                        low_price DECIMAL(20, 8) NOT NULL,
                        close_price DECIMAL(20, 8) NOT NULL,
                        volume DECIMAL(20, 8) NOT NULL,
                        source VARCHAR(20) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(timestamp, symbol, timeframe, source)
                    )
                ''')
                
                # Create indexes for OHLCV data
                await conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time 
                    ON ohlcv_data(symbol, timestamp DESC)
                ''')
                
                await conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_ohlcv_timeframe 
                    ON ohlcv_data(timeframe, timestamp DESC)
                ''')
                
                # Technical indicators table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS technical_indicators (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        timeframe VARCHAR(10) NOT NULL,
                        rsi DECIMAL(10, 4),
                        macd DECIMAL(10, 4),
                        macd_signal DECIMAL(10, 4),
                        macd_histogram DECIMAL(10, 4),
                        bb_upper DECIMAL(20, 8),
                        bb_middle DECIMAL(20, 8),
                        bb_lower DECIMAL(20, 8),
                        sma_20 DECIMAL(20, 8),
                        sma_50 DECIMAL(20, 8),
                        sma_200 DECIMAL(20, 8),
                        ema_12 DECIMAL(20, 8),
                        ema_26 DECIMAL(20, 8),
                        atr DECIMAL(10, 6),
                        adx DECIMAL(10, 4),
                        stoch_k DECIMAL(10, 4),
                        stoch_d DECIMAL(10, 4),
                        williams_r DECIMAL(10, 4),
                        cci DECIMAL(10, 4),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(timestamp, symbol, timeframe)
                    )
                ''')
                
                # Sentiment data table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS sentiment_data (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        source VARCHAR(20) NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        sentiment_score DECIMAL(5, 4) NOT NULL,
                        sentiment_type VARCHAR(20) NOT NULL,
                        confidence DECIMAL(5, 4) NOT NULL,
                        volume INTEGER NOT NULL,
                        keywords TEXT[],
                        raw_text TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # On-chain data table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS onchain_data (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        network VARCHAR(20) NOT NULL,
                        transaction_count INTEGER NOT NULL,
                        active_addresses INTEGER NOT NULL,
                        total_value_transferred DECIMAL(30, 8) NOT NULL,
                        average_transaction_value DECIMAL(20, 8) NOT NULL,
                        whale_movements JSONB,
                        network_fees DECIMAL(10, 6),
                        hash_rate DECIMAL(20, 2),
                        difficulty DECIMAL(30, 2),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(timestamp, symbol, network)
                    )
                ''')
                
                # Market metrics table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS market_metrics (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        fear_greed_index INTEGER,
                        bitcoin_dominance DECIMAL(5, 2),
                        total_market_cap DECIMAL(30, 2),
                        total_volume_24h DECIMAL(30, 2),
                        defi_tvl DECIMAL(30, 2),
                        stablecoin_supply DECIMAL(30, 2),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(timestamp)
                    )
                ''')
                
                # Data quality metrics table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS data_quality_metrics (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        source VARCHAR(20) NOT NULL,
                        symbol VARCHAR(20),
                        completeness_score DECIMAL(5, 4) NOT NULL,
                        accuracy_score DECIMAL(5, 4) NOT NULL,
                        freshness_score DECIMAL(5, 4) NOT NULL,
                        consistency_score DECIMAL(5, 4) NOT NULL,
                        overall_score DECIMAL(5, 4) NOT NULL,
                        missing_fields TEXT[],
                        outliers_detected INTEGER DEFAULT 0,
                        latency_ms DECIMAL(10, 2),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                self.logger.info("Database schemas created successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to create database schemas: {e}")
            raise
    
    # OHLCV Data Storage Methods
    async def store_ohlcv_data(self, ohlcv_list: List[OHLCV]) -> None:
        """Store OHLCV data in both InfluxDB and PostgreSQL."""
        if not ohlcv_list:
            return
        
        try:
            # Store in InfluxDB if available
            if self.influx_client:
                await self._store_ohlcv_influx(ohlcv_list)
            
            # Store in PostgreSQL as backup
            await self._store_ohlcv_postgres(ohlcv_list)
            
            # Cache latest data in Redis
            if self.redis_client:
                await self._cache_latest_ohlcv(ohlcv_list)
            
            self.logger.debug(f"Stored {len(ohlcv_list)} OHLCV records")
            
        except Exception as e:
            self.logger.error(f"Failed to store OHLCV data: {e}")
            raise
    
    async def _store_ohlcv_influx(self, ohlcv_list: List[OHLCV]) -> None:
        """Store OHLCV data in InfluxDB."""
        points = []
        
        for ohlcv in ohlcv_list:
            point = Point("ohlcv") \
                .tag("symbol", ohlcv.symbol) \
                .tag("timeframe", ohlcv.timeframe) \
                .tag("source", ohlcv.source.value) \
                .field("open", float(ohlcv.open)) \
                .field("high", float(ohlcv.high)) \
                .field("low", float(ohlcv.low)) \
                .field("close", float(ohlcv.close)) \
                .field("volume", float(ohlcv.volume)) \
                .time(ohlcv.timestamp, WritePrecision.NS)
            
            points.append(point)
        
        await self.influx_write_api.write(
            bucket=self.influx_config['bucket'],
            org=self.influx_config['org'],
            record=points
        )
    
    async def _store_ohlcv_postgres(self, ohlcv_list: List[OHLCV]) -> None:
        """Store OHLCV data in PostgreSQL."""
        async with self.postgres_pool.acquire() as conn:
            # Prepare data for bulk insert
            data = [
                (
                    ohlcv.timestamp,
                    ohlcv.symbol,
                    ohlcv.timeframe,
                    ohlcv.open,
                    ohlcv.high,
                    ohlcv.low,
                    ohlcv.close,
                    ohlcv.volume,
                    ohlcv.source.value
                )
                for ohlcv in ohlcv_list
            ]
            
            # Use COPY for efficient bulk insert
            await conn.copy_records_to_table(
                'ohlcv_data',
                records=data,
                columns=['timestamp', 'symbol', 'timeframe', 'open_price', 
                        'high_price', 'low_price', 'close_price', 'volume', 'source']
            )
    
    async def _cache_latest_ohlcv(self, ohlcv_list: List[OHLCV]) -> None:
        """Cache latest OHLCV data in Redis."""
        for ohlcv in ohlcv_list:
            cache_key = f"ohlcv:{ohlcv.symbol}:{ohlcv.timeframe}:latest"
            cache_data = {
                'timestamp': ohlcv.timestamp.isoformat(),
                'open': float(ohlcv.open),
                'high': float(ohlcv.high),
                'low': float(ohlcv.low),
                'close': float(ohlcv.close),
                'volume': float(ohlcv.volume),
                'source': ohlcv.source.value
            }
            
            await self.redis_client.setex(
                cache_key,
                self.cache_ttl['ohlcv'],
                json.dumps(cache_data)
            )
    
    async def get_ohlcv_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        source: Optional[DataSource] = None
    ) -> List[OHLCV]:
        """Retrieve OHLCV data from storage."""
        try:
            # Try InfluxDB first
            if self.influx_client:
                return await self._get_ohlcv_influx(symbol, timeframe, start_time, end_time, source)
            
            # Fallback to PostgreSQL
            return await self._get_ohlcv_postgres(symbol, timeframe, start_time, end_time, source)
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve OHLCV data: {e}")
            return []
    
    async def _get_ohlcv_influx(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        source: Optional[DataSource] = None
    ) -> List[OHLCV]:
        """Retrieve OHLCV data from InfluxDB."""
        # Build InfluxDB query
        query = f'''
            from(bucket: "{self.influx_config['bucket']}")
                |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
                |> filter(fn: (r) => r._measurement == "ohlcv")
                |> filter(fn: (r) => r.symbol == "{symbol}")
                |> filter(fn: (r) => r.timeframe == "{timeframe}")
        '''
        
        if source:
            query += f'|> filter(fn: (r) => r.source == "{source.value}")'
        
        query += '|> sort(columns: ["_time"])'
        
        result = await self.influx_query_api.query(query, org=self.influx_config['org'])
        
        # Process results
        ohlcv_data = []
        data_points = {}
        
        for table in result:
            for record in table.records:
                timestamp = record.get_time()
                field = record.get_field()
                value = record.get_value()
                
                if timestamp not in data_points:
                    data_points[timestamp] = {
                        'timestamp': timestamp,
                        'symbol': record.values.get('symbol'),
                        'timeframe': record.values.get('timeframe'),
                        'source': record.values.get('source')
                    }
                
                data_points[timestamp][field] = value
        
        # Convert to OHLCV objects
        for timestamp, data in sorted(data_points.items()):
            if all(field in data for field in ['open', 'high', 'low', 'close', 'volume']):
                ohlcv = OHLCV(
                    timestamp=timestamp,
                    symbol=data['symbol'],
                    timeframe=data['timeframe'],
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    volume=data['volume'],
                    source=DataSource(data['source'])
                )
                ohlcv_data.append(ohlcv)
        
        return ohlcv_data
    
    async def _get_ohlcv_postgres(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        source: Optional[DataSource] = None
    ) -> List[OHLCV]:
        """Retrieve OHLCV data from PostgreSQL."""
        async with self.postgres_pool.acquire() as conn:
            query = '''
                SELECT timestamp, symbol, timeframe, open_price, high_price, 
                       low_price, close_price, volume, source
                FROM ohlcv_data
                WHERE symbol = $1 AND timeframe = $2 
                      AND timestamp >= $3 AND timestamp <= $4
            '''
            
            params = [symbol, timeframe, start_time, end_time]
            
            if source:
                query += ' AND source = $5'
                params.append(source.value)
            
            query += ' ORDER BY timestamp'
            
            rows = await conn.fetch(query, *params)
            
            ohlcv_data = []
            for row in rows:
                ohlcv = OHLCV(
                    timestamp=row['timestamp'],
                    symbol=row['symbol'],
                    timeframe=row['timeframe'],
                    open=float(row['open_price']),
                    high=float(row['high_price']),
                    low=float(row['low_price']),
                    close=float(row['close_price']),
                    volume=float(row['volume']),
                    source=DataSource(row['source'])
                )
                ohlcv_data.append(ohlcv)
            
            return ohlcv_data
    
    # Technical Indicators Storage Methods
    async def store_technical_indicators(self, indicators: List[TechnicalIndicators]) -> None:
        """Store technical indicators in database."""
        if not indicators:
            return
        
        try:
            async with self.postgres_pool.acquire() as conn:
                data = []
                for indicator in indicators:
                    data.append((
                        indicator.timestamp,
                        indicator.symbol,
                        indicator.timeframe,
                        indicator.rsi,
                        indicator.macd,
                        indicator.macd_signal,
                        indicator.macd_histogram,
                        indicator.bb_upper,
                        indicator.bb_middle,
                        indicator.bb_lower,
                        indicator.sma_20,
                        indicator.sma_50,
                        indicator.sma_200,
                        indicator.ema_12,
                        indicator.ema_26,
                        indicator.atr,
                        indicator.adx,
                        indicator.stoch_k,
                        indicator.stoch_d,
                        indicator.williams_r,
                        indicator.cci
                    ))
                
                await conn.copy_records_to_table(
                    'technical_indicators',
                    records=data,
                    columns=[
                        'timestamp', 'symbol', 'timeframe', 'rsi', 'macd', 'macd_signal',
                        'macd_histogram', 'bb_upper', 'bb_middle', 'bb_lower', 'sma_20',
                        'sma_50', 'sma_200', 'ema_12', 'ema_26', 'atr', 'adx',
                        'stoch_k', 'stoch_d', 'williams_r', 'cci'
                    ]
                )
            
            # Cache latest indicators
            if self.redis_client:
                for indicator in indicators:
                    cache_key = f"technical:{indicator.symbol}:{indicator.timeframe}:latest"
                    cache_data = asdict(indicator)
                    cache_data['timestamp'] = cache_data['timestamp'].isoformat()
                    
                    await self.redis_client.setex(
                        cache_key,
                        self.cache_ttl['technical'],
                        json.dumps(cache_data)
                    )
            
            self.logger.debug(f"Stored {len(indicators)} technical indicator records")
            
        except Exception as e:
            self.logger.error(f"Failed to store technical indicators: {e}")
            raise
    
    # Sentiment Data Storage Methods
    async def store_sentiment_data(self, sentiment_list: List[SentimentData]) -> None:
        """Store sentiment data in database."""
        if not sentiment_list:
            return
        
        try:
            async with self.postgres_pool.acquire() as conn:
                data = []
                for sentiment in sentiment_list:
                    data.append((
                        sentiment.timestamp,
                        sentiment.source.value,
                        sentiment.symbol,
                        sentiment.sentiment_score,
                        sentiment.sentiment_type.value,
                        sentiment.confidence,
                        sentiment.volume,
                        sentiment.keywords,
                        sentiment.raw_text
                    ))
                
                await conn.copy_records_to_table(
                    'sentiment_data',
                    records=data,
                    columns=[
                        'timestamp', 'source', 'symbol', 'sentiment_score',
                        'sentiment_type', 'confidence', 'volume', 'keywords', 'raw_text'
                    ]
                )
            
            self.logger.debug(f"Stored {len(sentiment_list)} sentiment records")
            
        except Exception as e:
            self.logger.error(f"Failed to store sentiment data: {e}")
            raise
    
    # On-chain Data Storage Methods
    async def store_onchain_data(self, onchain_list: List[OnChainData]) -> None:
        """Store on-chain data in database."""
        if not onchain_list:
            return
        
        try:
            async with self.postgres_pool.acquire() as conn:
                data = []
                for onchain in onchain_list:
                    data.append((
                        onchain.timestamp,
                        onchain.symbol,
                        onchain.network,
                        onchain.transaction_count,
                        onchain.active_addresses,
                        onchain.total_value_transferred,
                        onchain.average_transaction_value,
                        json.dumps(onchain.whale_movements),
                        onchain.network_fees,
                        onchain.hash_rate,
                        onchain.difficulty
                    ))
                
                await conn.copy_records_to_table(
                    'onchain_data',
                    records=data,
                    columns=[
                        'timestamp', 'symbol', 'network', 'transaction_count',
                        'active_addresses', 'total_value_transferred', 'average_transaction_value',
                        'whale_movements', 'network_fees', 'hash_rate', 'difficulty'
                    ]
                )
            
            self.logger.debug(f"Stored {len(onchain_list)} on-chain records")
            
        except Exception as e:
            self.logger.error(f"Failed to store on-chain data: {e}")
            raise
    
    # Data Quality Storage Methods
    async def store_data_quality_metrics(self, quality_metrics: List[DataQualityMetrics]) -> None:
        """Store data quality metrics."""
        if not quality_metrics:
            return
        
        try:
            async with self.postgres_pool.acquire() as conn:
                data = []
                for metric in quality_metrics:
                    data.append((
                        metric.timestamp,
                        metric.source.value,
                        metric.symbol,
                        metric.completeness_score,
                        metric.accuracy_score,
                        metric.freshness_score,
                        metric.consistency_score,
                        metric.overall_score,
                        metric.missing_fields,
                        metric.outliers_detected,
                        metric.latency_ms
                    ))
                
                await conn.copy_records_to_table(
                    'data_quality_metrics',
                    records=data,
                    columns=[
                        'timestamp', 'source', 'symbol', 'completeness_score',
                        'accuracy_score', 'freshness_score', 'consistency_score',
                        'overall_score', 'missing_fields', 'outliers_detected', 'latency_ms'
                    ]
                )
            
            self.logger.debug(f"Stored {len(quality_metrics)} data quality records")
            
        except Exception as e:
            self.logger.error(f"Failed to store data quality metrics: {e}")
            raise
    
    # Data Cleanup Methods
    async def cleanup_old_data(self) -> None:
        """Clean up old data based on retention policies."""
        try:
            async with self.postgres_pool.acquire() as conn:
                cleanup_count = 0
                
                # Clean up OHLCV data based on timeframe
                for timeframe_key, retention_days in self.retention_policies.items():
                    if timeframe_key.startswith('ohlcv_'):
                        timeframe = timeframe_key.replace('ohlcv_', '')
                        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
                        
                        result = await conn.execute('''
                            DELETE FROM ohlcv_data 
                            WHERE timeframe = $1 AND timestamp < $2
                        ''', timeframe, cutoff_date)
                        
                        cleanup_count += int(result.split()[-1])
                
                # Clean up other data types
                for table, retention_days in [
                    ('sentiment_data', self.retention_policies['sentiment']),
                    ('onchain_data', self.retention_policies['onchain']),
                    ('technical_indicators', self.retention_policies['technical'])
                ]:
                    cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
                    result = await conn.execute(f'''
                        DELETE FROM {table} WHERE timestamp < $1
                    ''', cutoff_date)
                    
                    cleanup_count += int(result.split()[-1])
                
                self.logger.info(f"Cleaned up {cleanup_count} old records")
                
        except Exception as e:
            self.logger.error(f"Data cleanup failed: {e}")
    
    # Cache Management Methods
    async def get_cached_data(self, cache_key: str) -> Optional[Any]:
        """Get data from Redis cache."""
        if not self.redis_client:
            return None
        
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            self.logger.warning(f"Cache retrieval failed for {cache_key}: {e}")
        
        return None
    
    async def set_cached_data(self, cache_key: str, data: Any, ttl: int = 300) -> None:
        """Set data in Redis cache."""
        if not self.redis_client:
            return
        
        try:
            await self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(data)
            )
        except Exception as e:
            self.logger.warning(f"Cache storage failed for {cache_key}: {e}")
    
    async def get_database_health(self) -> Dict[str, Any]:
        """Get database health status."""
        health_status = {
            'postgres': 'unknown',
            'influxdb': 'unknown',
            'redis': 'unknown',
            'total_records': {}
        }
        
        try:
            # Check PostgreSQL
            if self.postgres_pool:
                async with self.postgres_pool.acquire() as conn:
                    await conn.fetchval('SELECT 1')
                    health_status['postgres'] = 'healthy'
                    
                    # Get record counts
                    tables = ['ohlcv_data', 'technical_indicators', 'sentiment_data', 'onchain_data']
                    for table in tables:
                        count = await conn.fetchval(f'SELECT COUNT(*) FROM {table}')
                        health_status['total_records'][table] = count
            
            # Check InfluxDB
            if self.influx_client:
                health = await self.influx_client.health()
                health_status['influxdb'] = 'healthy' if health.status == 'pass' else 'unhealthy'
            
            # Check Redis
            if self.redis_client:
                await self.redis_client.ping()
                health_status['redis'] = 'healthy'
            
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
        
        return health_status