"""
Multi-exchange market data collector.
Integrates with Coinbase, Kraken, and other major cryptocurrency exchanges.
"""

import asyncio
import ccxt.async_support as ccxt
import websockets
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from .base import BaseMarketDataCollector, CollectorConfig
from ..models import (
    DataSource, OHLCV, OrderBook, OrderBookLevel, 
    TechnicalIndicators, DataQualityMetrics
)
from ...utils.logger import get_logger
from ...security.credential_manager import CredentialManager


class ExchangeCollector(BaseMarketDataCollector):
    """
    Multi-exchange data collector supporting major crypto exchanges.
    Implements both REST API and WebSocket connections for real-time data.
    """
    
    SUPPORTED_EXCHANGES = {
        DataSource.COINBASE: 'coinbasepro',
        DataSource.KRAKEN: 'kraken',
        DataSource.BINANCE: 'binance'
    }
    
    def __init__(self, config: CollectorConfig, credential_manager: CredentialManager):
        super().__init__(config, credential_manager)
        self.exchange = None
        self.websocket_urls = {
            DataSource.COINBASE: 'wss://ws-feed.pro.coinbase.com',
            DataSource.KRAKEN: 'wss://ws.kraken.com',
            DataSource.BINANCE: 'wss://stream.binance.com:9443/ws'
        }
        
        # Symbol mapping between exchanges
        self.symbol_mappings = {
            DataSource.COINBASE: self._coinbase_symbol_format,
            DataSource.KRAKEN: self._kraken_symbol_format,
            DataSource.BINANCE: self._binance_symbol_format
        }
    
    async def initialize(self) -> None:
        """Initialize exchange connection and authentication."""
        try:
            exchange_class = getattr(ccxt, self.SUPPORTED_EXCHANGES[self.config.source])
            
            # Get API credentials securely
            api_credentials = await self._get_api_credentials()
            
            self.exchange = exchange_class({
                **api_credentials,
                'timeout': self.config.timeout * 1000,  # ccxt expects milliseconds
                'enableRateLimit': True,
                'sandbox': False,  # Use production environment
            })
            
            # Load markets
            await self.exchange.load_markets()
            
            self.logger.info(
                f"Initialized {self.config.source.value} exchange collector",
                extra={'markets_loaded': len(self.exchange.markets)}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize exchange: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up exchange connection."""
        if self.exchange:
            await self.exchange.close()
        
        # Close WebSocket connections
        for ws in self._websocket_connections.values():
            if not ws.closed:
                await ws.close()
    
    async def _get_api_credentials(self) -> Dict[str, str]:
        """Get API credentials for the exchange."""
        source_name = self.config.source.value.upper()
        
        try:
            credentials = {
                'apiKey': await self.credential_manager.get_credential(f'{source_name}_API_KEY'),
                'secret': await self.credential_manager.get_credential(f'{source_name}_SECRET'),
            }
            
            # Some exchanges require additional credentials
            if self.config.source == DataSource.COINBASE:
                credentials['password'] = await self.credential_manager.get_credential(
                    'COINBASE_PASSPHRASE'
                )
            
            return credentials
            
        except Exception as e:
            self.logger.warning(f"No API credentials found for {source_name}: {e}")
            return {}  # Return empty dict for public-only access
    
    def _coinbase_symbol_format(self, symbol: str) -> str:
        """Convert symbol to Coinbase format (e.g., BTC/USD -> BTC-USD)."""
        return symbol.replace('/', '-')
    
    def _kraken_symbol_format(self, symbol: str) -> str:
        """Convert symbol to Kraken format."""
        # Kraken uses different naming conventions
        if symbol == 'BTC/USD':
            return 'XBTUSD'
        elif symbol == 'ETH/USD':
            return 'ETHUSD'
        return symbol.replace('/', '')
    
    def _binance_symbol_format(self, symbol: str) -> str:
        """Convert symbol to Binance format (e.g., BTC/USDT -> BTCUSDT)."""
        return symbol.replace('/', '')
    
    def _format_symbol(self, symbol: str) -> str:
        """Format symbol for the current exchange."""
        formatter = self.symbol_mappings.get(self.config.source)
        return formatter(symbol) if formatter else symbol
    
    async def collect_ohlcv(
        self, 
        symbol: str, 
        timeframe: str, 
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[OHLCV]:
        """Collect OHLCV data from exchange."""
        formatted_symbol = self._format_symbol(symbol)
        
        try:
            # Convert since datetime to timestamp if provided
            since_timestamp = None
            if since:
                since_timestamp = int(since.timestamp() * 1000)
            
            # Fetch OHLCV data
            ohlcv_data = await self.collect_with_retry(
                self.exchange.fetch_ohlcv,
                formatted_symbol,
                timeframe,
                since_timestamp,
                limit
            )
            
            # Convert to OHLCV objects
            result = []
            for candle in ohlcv_data:
                timestamp, open_price, high, low, close, volume = candle
                
                ohlcv = OHLCV(
                    timestamp=datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc),
                    symbol=symbol,
                    timeframe=timeframe,
                    open=float(open_price),
                    high=float(high),
                    low=float(low),
                    close=float(close),
                    volume=float(volume),
                    source=self.config.source
                )
                result.append(ohlcv)
            
            self.metrics['total_data_points'] += len(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to collect OHLCV for {symbol}: {e}")
            raise
    
    async def collect_orderbook(self, symbol: str, limit: Optional[int] = None) -> Optional[OrderBook]:
        """Collect order book data from exchange."""
        formatted_symbol = self._format_symbol(symbol)
        
        try:
            orderbook_data = await self.collect_with_retry(
                self.exchange.fetch_order_book,
                formatted_symbol,
                limit
            )
            
            # Convert bids and asks
            bids = [
                OrderBookLevel(price=float(bid[0]), quantity=float(bid[1]))
                for bid in orderbook_data['bids']
            ]
            
            asks = [
                OrderBookLevel(price=float(ask[0]), quantity=float(ask[1]))
                for ask in orderbook_data['asks']
            ]
            
            orderbook = OrderBook(
                timestamp=datetime.fromtimestamp(
                    orderbook_data['timestamp'] / 1000, tz=timezone.utc
                ),
                symbol=symbol,
                bids=bids,
                asks=asks,
                source=self.config.source
            )
            
            return orderbook
            
        except Exception as e:
            self.logger.error(f"Failed to collect order book for {symbol}: {e}")
            return None
    
    async def collect_ticker(self, symbol: str) -> Dict[str, Any]:
        """Collect ticker data from exchange."""
        formatted_symbol = self._format_symbol(symbol)
        
        try:
            ticker = await self.collect_with_retry(
                self.exchange.fetch_ticker,
                formatted_symbol
            )
            
            return {
                'symbol': symbol,
                'timestamp': datetime.fromtimestamp(ticker['timestamp'] / 1000, tz=timezone.utc),
                'last': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['baseVolume'],
                'high': ticker['high'],
                'low': ticker['low'],
                'change': ticker['change'],
                'percentage': ticker['percentage'],
                'source': self.config.source.value
            }
            
        except Exception as e:
            self.logger.error(f"Failed to collect ticker for {symbol}: {e}")
            raise
    
    async def collect_trades(
        self, 
        symbol: str, 
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Collect recent trades from exchange."""
        formatted_symbol = self._format_symbol(symbol)
        
        try:
            since_timestamp = None
            if since:
                since_timestamp = int(since.timestamp() * 1000)
            
            trades = await self.collect_with_retry(
                self.exchange.fetch_trades,
                formatted_symbol,
                since_timestamp,
                limit
            )
            
            result = []
            for trade in trades:
                result.append({
                    'id': trade['id'],
                    'timestamp': datetime.fromtimestamp(trade['timestamp'] / 1000, tz=timezone.utc),
                    'symbol': symbol,
                    'side': trade['side'],
                    'amount': trade['amount'],
                    'price': trade['price'],
                    'cost': trade['cost'],
                    'source': self.config.source.value
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to collect trades for {symbol}: {e}")
            raise
    
    async def get_supported_symbols(self) -> List[str]:
        """Get list of supported trading symbols."""
        if not self.exchange:
            await self.initialize()
        
        # Filter for relevant crypto pairs
        symbols = []
        for symbol in self.exchange.symbols:
            # Focus on major USD, USDT, EUR pairs
            if any(quote in symbol for quote in ['/USD', '/USDT', '/EUR', '/BTC']):
                symbols.append(symbol)
        
        return sorted(symbols)
    
    async def get_supported_timeframes(self) -> List[str]:
        """Get list of supported timeframes."""
        if not self.exchange:
            await self.initialize()
        
        return list(self.exchange.timeframes.keys()) if self.exchange.timeframes else []
    
    async def start_websocket_stream(self, symbols: List[str], data_types: List[str]) -> None:
        """Start WebSocket streams for real-time data."""
        if not self.config.enable_websocket:
            self.logger.info("WebSocket streaming disabled in configuration")
            return
        
        websocket_url = self.websocket_urls.get(self.config.source)
        if not websocket_url:
            self.logger.warning(f"WebSocket not supported for {self.config.source.value}")
            return
        
        try:
            # Start WebSocket connection based on exchange
            if self.config.source == DataSource.COINBASE:
                await self._start_coinbase_websocket(symbols, data_types)
            elif self.config.source == DataSource.KRAKEN:
                await self._start_kraken_websocket(symbols, data_types)
            elif self.config.source == DataSource.BINANCE:
                await self._start_binance_websocket(symbols, data_types)
        
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket stream: {e}")
            raise
    
    async def _start_coinbase_websocket(self, symbols: List[str], data_types: List[str]) -> None:
        """Start Coinbase Pro WebSocket stream."""
        uri = self.websocket_urls[DataSource.COINBASE]
        
        # Format symbols for Coinbase
        formatted_symbols = [self._format_symbol(symbol) for symbol in symbols]
        
        subscribe_message = {
            "type": "subscribe",
            "channels": [
                {
                    "name": "ticker",
                    "product_ids": formatted_symbols
                },
                {
                    "name": "level2",
                    "product_ids": formatted_symbols
                }
            ]
        }
        
        async def handle_coinbase_messages():
            try:
                async with websockets.connect(uri) as websocket:
                    self._websocket_connections['coinbase'] = websocket
                    
                    # Subscribe to channels
                    await websocket.send(json.dumps(subscribe_message))
                    
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            await self._process_coinbase_message(data)
                        except Exception as e:
                            self.logger.error(f"Error processing Coinbase message: {e}")
            
            except Exception as e:
                self.logger.error(f"Coinbase WebSocket connection failed: {e}")
        
        # Start WebSocket in background
        asyncio.create_task(handle_coinbase_messages())
    
    async def _start_kraken_websocket(self, symbols: List[str], data_types: List[str]) -> None:
        """Start Kraken WebSocket stream."""
        uri = self.websocket_urls[DataSource.KRAKEN]
        
        formatted_symbols = [self._format_symbol(symbol) for symbol in symbols]
        
        subscribe_message = {
            "event": "subscribe",
            "pair": formatted_symbols,
            "subscription": {"name": "ticker"}
        }
        
        async def handle_kraken_messages():
            try:
                async with websockets.connect(uri) as websocket:
                    self._websocket_connections['kraken'] = websocket
                    
                    await websocket.send(json.dumps(subscribe_message))
                    
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            await self._process_kraken_message(data)
                        except Exception as e:
                            self.logger.error(f"Error processing Kraken message: {e}")
            
            except Exception as e:
                self.logger.error(f"Kraken WebSocket connection failed: {e}")
        
        asyncio.create_task(handle_kraken_messages())
    
    async def _start_binance_websocket(self, symbols: List[str], data_types: List[str]) -> None:
        """Start Binance WebSocket stream."""
        # Binance uses individual streams, so we create multiple connections
        for symbol in symbols:
            formatted_symbol = self._format_symbol(symbol).lower()
            stream_name = f"{formatted_symbol}@ticker"
            uri = f"{self.websocket_urls[DataSource.BINANCE]}/{stream_name}"
            
            async def handle_binance_stream(stream_uri, symbol_name):
                try:
                    async with websockets.connect(stream_uri) as websocket:
                        self._websocket_connections[f'binance_{symbol_name}'] = websocket
                        
                        async for message in websocket:
                            try:
                                data = json.loads(message)
                                await self._process_binance_message(data, symbol_name)
                            except Exception as e:
                                self.logger.error(f"Error processing Binance message: {e}")
                
                except Exception as e:
                    self.logger.error(f"Binance WebSocket connection failed for {symbol_name}: {e}")
            
            asyncio.create_task(handle_binance_stream(uri, symbol))
    
    async def _process_coinbase_message(self, data: Dict[str, Any]) -> None:
        """Process Coinbase WebSocket message."""
        if data.get('type') == 'ticker':
            # Process ticker update
            self.logger.debug(f"Coinbase ticker update: {data['product_id']} @ {data['price']}")
        elif data.get('type') == 'l2update':
            # Process order book update
            self.logger.debug(f"Coinbase L2 update: {data['product_id']}")
    
    async def _process_kraken_message(self, data: Dict[str, Any]) -> None:
        """Process Kraken WebSocket message."""
        if isinstance(data, list) and len(data) >= 2:
            # Kraken ticker data
            self.logger.debug(f"Kraken ticker update: {data}")
    
    async def _process_binance_message(self, data: Dict[str, Any], symbol: str) -> None:
        """Process Binance WebSocket message."""
        if 'c' in data:  # Current price
            self.logger.debug(f"Binance ticker update: {symbol} @ {data['c']}")


class TechnicalIndicatorCalculator:
    """
    Calculate technical indicators from OHLCV data.
    Uses TA-Lib for efficient calculation of technical indicators.
    """
    
    def __init__(self):
        self.logger = get_logger("technical_indicators")
    
    def calculate_all_indicators(
        self, 
        ohlcv_data: List[OHLCV], 
        symbol: str, 
        timeframe: str
    ) -> List[TechnicalIndicators]:
        """Calculate all technical indicators for OHLCV data."""
        if len(ohlcv_data) < 50:  # Need sufficient data for indicators
            self.logger.warning(f"Insufficient data for {symbol}: {len(ohlcv_data)} candles")
            return []
        
        # Convert to pandas for easier manipulation
        df = pd.DataFrame([
            {
                'timestamp': candle.timestamp,
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume
            }
            for candle in ohlcv_data
        ])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate indicators
        indicators = []
        
        try:
            # Import TA-Lib
            import talib
            
            # Convert to numpy arrays
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            volume = df['volume'].values
            
            # Calculate RSI
            rsi = talib.RSI(close, timeperiod=14)
            
            # Calculate MACD
            macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            
            # Calculate Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            
            # Calculate Moving Averages
            sma_20 = talib.SMA(close, timeperiod=20)
            sma_50 = talib.SMA(close, timeperiod=50)
            sma_200 = talib.SMA(close, timeperiod=200)
            ema_12 = talib.EMA(close, timeperiod=12)
            ema_26 = talib.EMA(close, timeperiod=26)
            
            # Calculate ATR
            atr = talib.ATR(high, low, close, timeperiod=14)
            
            # Calculate ADX
            adx = talib.ADX(high, low, close, timeperiod=14)
            
            # Calculate Stochastic
            stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
            
            # Calculate Williams %R
            williams_r = talib.WILLR(high, low, close, timeperiod=14)
            
            # Calculate CCI
            cci = talib.CCI(high, low, close, timeperiod=14)
            
            # Create TechnicalIndicators objects
            for i, row in df.iterrows():
                tech_indicators = TechnicalIndicators(
                    timestamp=row['timestamp'],
                    symbol=symbol,
                    timeframe=timeframe,
                    rsi=rsi[i] if not np.isnan(rsi[i]) else None,
                    macd=macd[i] if not np.isnan(macd[i]) else None,
                    macd_signal=macd_signal[i] if not np.isnan(macd_signal[i]) else None,
                    macd_histogram=macd_hist[i] if not np.isnan(macd_hist[i]) else None,
                    bb_upper=bb_upper[i] if not np.isnan(bb_upper[i]) else None,
                    bb_middle=bb_middle[i] if not np.isnan(bb_middle[i]) else None,
                    bb_lower=bb_lower[i] if not np.isnan(bb_lower[i]) else None,
                    sma_20=sma_20[i] if not np.isnan(sma_20[i]) else None,
                    sma_50=sma_50[i] if not np.isnan(sma_50[i]) else None,
                    sma_200=sma_200[i] if not np.isnan(sma_200[i]) else None,
                    ema_12=ema_12[i] if not np.isnan(ema_12[i]) else None,
                    ema_26=ema_26[i] if not np.isnan(ema_26[i]) else None,
                    atr=atr[i] if not np.isnan(atr[i]) else None,
                    adx=adx[i] if not np.isnan(adx[i]) else None,
                    stoch_k=stoch_k[i] if not np.isnan(stoch_k[i]) else None,
                    stoch_d=stoch_d[i] if not np.isnan(stoch_d[i]) else None,
                    williams_r=williams_r[i] if not np.isnan(williams_r[i]) else None,
                    cci=cci[i] if not np.isnan(cci[i]) else None
                )
                indicators.append(tech_indicators)
            
            self.logger.info(f"Calculated indicators for {symbol}: {len(indicators)} data points")
            return indicators
            
        except Exception as e:
            self.logger.error(f"Failed to calculate technical indicators for {symbol}: {e}")
            return []