"""
Data models for the crypto trading system.
Enterprise-grade data structures with comprehensive validation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import numpy as np
import pandas as pd


class DataSource(Enum):
    """Enumeration of supported data sources."""
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    BINANCE = "binance"
    HELIUS = "helius"
    TRADINGVIEW = "tradingview"
    COINGECKO = "coingecko"
    TWITTER = "twitter"
    REDDIT = "reddit"
    NEWS = "news"
    FEAR_GREED = "fear_greed"


class MarketDataType(Enum):
    """Types of market data."""
    OHLCV = "ohlcv"
    ORDERBOOK = "orderbook"
    TRADES = "trades"
    TICKER = "ticker"
    FUNDING_RATE = "funding_rate"
    OPEN_INTEREST = "open_interest"


class SentimentType(Enum):
    """Types of sentiment data."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"


@dataclass
class OHLCV:
    """Open, High, Low, Close, Volume data structure."""
    timestamp: datetime
    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: DataSource
    
    def __post_init__(self):
        """Validate OHLCV data integrity."""
        if self.high < max(self.open, self.close):
            raise ValueError("High must be >= max(open, close)")
        if self.low > min(self.open, self.close):
            raise ValueError("Low must be <= min(open, close)")
        if self.volume < 0:
            raise ValueError("Volume must be non-negative")
        if any(v <= 0 for v in [self.open, self.high, self.low, self.close]):
            raise ValueError("All prices must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'source': self.source.value
        }


@dataclass
class OrderBookLevel:
    """Single order book level (bid or ask)."""
    price: float
    quantity: float
    orders: int = 0
    
    def __post_init__(self):
        """Validate order book level data."""
        if self.price <= 0:
            raise ValueError("Price must be positive")
        if self.quantity < 0:
            raise ValueError("Quantity must be non-negative")
        if self.orders < 0:
            raise ValueError("Number of orders must be non-negative")


@dataclass
class OrderBook:
    """Complete order book data structure."""
    timestamp: datetime
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    source: DataSource
    
    def __post_init__(self):
        """Validate and sort order book data."""
        # Sort bids by price (descending) and asks by price (ascending)
        self.bids.sort(key=lambda x: x.price, reverse=True)
        self.asks.sort(key=lambda x: x.price)
        
        # Validate bid/ask spread
        if self.bids and self.asks:
            if self.bids[0].price >= self.asks[0].price:
                raise ValueError("Best bid must be lower than best ask")
    
    @property
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread."""
        if not self.bids or not self.asks:
            return None
        return self.asks[0].price - self.bids[0].price
    
    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid price."""
        if not self.bids or not self.asks:
            return None
        return (self.bids[0].price + self.asks[0].price) / 2


@dataclass
class TechnicalIndicators:
    """Technical analysis indicators."""
    timestamp: datetime
    symbol: str
    timeframe: str
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    atr: Optional[float] = None
    adx: Optional[float] = None
    stoch_k: Optional[float] = None
    stoch_d: Optional[float] = None
    williams_r: Optional[float] = None
    cci: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            **{k: v for k, v in self.__dict__.items() 
               if k not in ['timestamp', 'symbol', 'timeframe']}
        }


@dataclass
class SentimentData:
    """Social sentiment data structure."""
    timestamp: datetime
    source: DataSource
    symbol: str
    sentiment_score: float  # -1.0 (bearish) to 1.0 (bullish)
    sentiment_type: SentimentType
    confidence: float  # 0.0 to 1.0
    volume: int  # Number of mentions/posts
    keywords: List[str] = field(default_factory=list)
    raw_text: Optional[str] = None
    
    def __post_init__(self):
        """Validate sentiment data."""
        if not -1.0 <= self.sentiment_score <= 1.0:
            raise ValueError("Sentiment score must be between -1.0 and 1.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.volume < 0:
            raise ValueError("Volume must be non-negative")


@dataclass
class OnChainData:
    """Blockchain on-chain analytics data."""
    timestamp: datetime
    symbol: str
    network: str  # e.g., "solana", "ethereum", "bitcoin"
    transaction_count: int
    active_addresses: int
    total_value_transferred: float
    average_transaction_value: float
    whale_movements: List[Dict[str, Any]] = field(default_factory=list)
    network_fees: Optional[float] = None
    hash_rate: Optional[float] = None
    difficulty: Optional[float] = None
    
    def __post_init__(self):
        """Validate on-chain data."""
        if self.transaction_count < 0:
            raise ValueError("Transaction count must be non-negative")
        if self.active_addresses < 0:
            raise ValueError("Active addresses must be non-negative")
        if self.total_value_transferred < 0:
            raise ValueError("Total value transferred must be non-negative")


@dataclass
class MarketMetrics:
    """Market-wide metrics and indicators."""
    timestamp: datetime
    fear_greed_index: Optional[int] = None  # 0-100
    bitcoin_dominance: Optional[float] = None
    total_market_cap: Optional[float] = None
    total_volume_24h: Optional[float] = None
    defi_tvl: Optional[float] = None
    stablecoin_supply: Optional[float] = None
    
    def __post_init__(self):
        """Validate market metrics."""
        if self.fear_greed_index is not None:
            if not 0 <= self.fear_greed_index <= 100:
                raise ValueError("Fear & Greed Index must be between 0 and 100")
        
        if self.bitcoin_dominance is not None:
            if not 0 <= self.bitcoin_dominance <= 100:
                raise ValueError("Bitcoin dominance must be between 0% and 100%")


@dataclass
class ProcessedData:
    """Aggregated and processed data for ML models."""
    timestamp: datetime
    symbol: str
    timeframe: str
    
    # Price data
    ohlcv: OHLCV
    
    # Technical indicators
    technical: TechnicalIndicators
    
    # Sentiment data
    sentiment: Optional[SentimentData] = None
    
    # On-chain data
    onchain: Optional[OnChainData] = None
    
    # Market metrics
    market_metrics: Optional[MarketMetrics] = None
    
    # Order book data
    orderbook: Optional[OrderBook] = None
    
    # Data quality score (0.0 to 1.0)
    quality_score: float = 1.0
    
    def __post_init__(self):
        """Validate processed data integrity."""
        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError("Quality score must be between 0.0 and 1.0")
        
        # Ensure timestamp consistency
        if self.ohlcv.timestamp != self.timestamp:
            raise ValueError("OHLCV timestamp must match ProcessedData timestamp")
        
        if self.technical.timestamp != self.timestamp:
            raise ValueError("Technical indicators timestamp must match ProcessedData timestamp")
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to ML-ready feature vector."""
        features = []
        
        # Price features
        features.extend([
            self.ohlcv.open, self.ohlcv.high, self.ohlcv.low, 
            self.ohlcv.close, self.ohlcv.volume
        ])
        
        # Technical indicator features
        technical_features = [
            self.technical.rsi or 0.0,
            self.technical.macd or 0.0,
            self.technical.macd_signal or 0.0,
            self.technical.bb_upper or 0.0,
            self.technical.bb_lower or 0.0,
            self.technical.sma_20 or 0.0,
            self.technical.sma_50 or 0.0,
            self.technical.atr or 0.0,
            self.technical.adx or 0.0,
            self.technical.stoch_k or 0.0,
            self.technical.williams_r or 0.0,
        ]
        features.extend(technical_features)
        
        # Sentiment features
        if self.sentiment:
            features.extend([
                self.sentiment.sentiment_score,
                self.sentiment.confidence,
                float(self.sentiment.volume)
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # On-chain features
        if self.onchain:
            features.extend([
                float(self.onchain.transaction_count),
                float(self.onchain.active_addresses),
                self.onchain.total_value_transferred,
                self.onchain.average_transaction_value
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Market metrics features
        if self.market_metrics:
            features.extend([
                float(self.market_metrics.fear_greed_index or 50),
                self.market_metrics.bitcoin_dominance or 0.0,
                self.market_metrics.total_market_cap or 0.0
            ])
        else:
            features.extend([50.0, 0.0, 0.0])
        
        # Order book features
        if self.orderbook and self.orderbook.spread is not None:
            features.extend([
                self.orderbook.spread,
                self.orderbook.mid_price or 0.0
            ])
        else:
            features.extend([0.0, 0.0])
        
        # Quality score
        features.append(self.quality_score)
        
        return np.array(features, dtype=np.float32)
    
    def to_pandas_series(self) -> pd.Series:
        """Convert to pandas Series for analysis."""
        data_dict = {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'open': self.ohlcv.open,
            'high': self.ohlcv.high,
            'low': self.ohlcv.low,
            'close': self.ohlcv.close,
            'volume': self.ohlcv.volume,
            'quality_score': self.quality_score
        }
        
        # Add technical indicators
        for attr_name in self.technical.__dict__:
            if attr_name not in ['timestamp', 'symbol', 'timeframe']:
                data_dict[f'tech_{attr_name}'] = getattr(self.technical, attr_name)
        
        # Add sentiment data
        if self.sentiment:
            data_dict.update({
                'sentiment_score': self.sentiment.sentiment_score,
                'sentiment_confidence': self.sentiment.confidence,
                'sentiment_volume': self.sentiment.volume
            })
        
        # Add on-chain data
        if self.onchain:
            data_dict.update({
                'onchain_tx_count': self.onchain.transaction_count,
                'onchain_active_addresses': self.onchain.active_addresses,
                'onchain_total_value': self.onchain.total_value_transferred
            })
        
        # Add market metrics
        if self.market_metrics:
            data_dict.update({
                'fear_greed': self.market_metrics.fear_greed_index,
                'btc_dominance': self.market_metrics.bitcoin_dominance,
                'total_market_cap': self.market_metrics.total_market_cap
            })
        
        return pd.Series(data_dict)


@dataclass
class DataQualityMetrics:
    """Data quality assessment metrics."""
    timestamp: datetime
    source: DataSource
    symbol: str
    completeness_score: float  # 0.0 to 1.0
    accuracy_score: float  # 0.0 to 1.0
    freshness_score: float  # 0.0 to 1.0
    consistency_score: float  # 0.0 to 1.0
    overall_score: float  # 0.0 to 1.0
    missing_fields: List[str] = field(default_factory=list)
    outliers_detected: int = 0
    latency_ms: Optional[float] = None
    
    def __post_init__(self):
        """Calculate overall quality score."""
        scores = [
            self.completeness_score,
            self.accuracy_score,
            self.freshness_score,
            self.consistency_score
        ]
        
        if not all(0.0 <= score <= 1.0 for score in scores):
            raise ValueError("All quality scores must be between 0.0 and 1.0")
        
        # Weighted average with emphasis on completeness and accuracy
        weights = [0.3, 0.4, 0.2, 0.1]
        self.overall_score = sum(score * weight for score, weight in zip(scores, weights))


# Type aliases for better code readability
MarketDataPoint = Union[OHLCV, OrderBook, TechnicalIndicators]
DataPoint = Union[MarketDataPoint, SentimentData, OnChainData, MarketMetrics]