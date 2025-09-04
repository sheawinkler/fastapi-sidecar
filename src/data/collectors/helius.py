"""
Helius RPC collector for Solana on-chain data.
Optimized for $50 budget with intelligent request batching and caching.
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union
import hashlib
import time

from .base import BaseOnChainCollector, CollectorConfig
from ..models import DataSource, OnChainData
from ...utils.logger import get_logger
from ...security.credential_manager import CredentialManager


class HeliusRPCCollector(BaseOnChainCollector):
    """
    Helius RPC collector optimized for cost efficiency.
    Implements intelligent batching, caching, and adaptive polling.
    """
    
    # Helius RPC endpoints
    MAINNET_URL = "https://mainnet.helius-rpc.com"
    DEVNET_URL = "https://devnet.helius-rpc.com"
    
    # Rate limits for $50 plan (100 requests per second)
    DEFAULT_RATE_LIMIT = 100
    
    # Popular Solana tokens and their addresses
    POPULAR_TOKENS = {
        'SOL': 'So11111111111111111111111111111111111111112',
        'USDC': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
        'USDT': 'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB',
        'RAY': '4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R',
        'SRM': 'SRMuApVNdxXokk5GT7XD5cUUgXMBCoAz2LHeuAoKWRt',
        'ORCA': 'orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE'
    }
    
    def __init__(self, config: CollectorConfig, credential_manager: CredentialManager):
        super().__init__(config, credential_manager)
        self.api_key = None
        self.base_url = self.MAINNET_URL
        self.session = None
        
        # Request optimization
        self.request_cache = {}  # Simple in-memory cache
        self.cache_ttl = 60  # 1 minute cache TTL
        self.batch_queue = []
        self.batch_size = 50  # Batch multiple requests
        self.last_batch_time = 0
        
        # Adaptive polling intervals
        self.polling_intervals = {
            'high_activity': 5,    # 5 seconds for high-activity accounts
            'medium_activity': 30,  # 30 seconds for medium activity
            'low_activity': 300    # 5 minutes for low activity
        }
        
        # Account activity tracking
        self.account_activity = {}
        
        # Cost tracking
        self.request_count = 0
        self.daily_request_limit = 50000  # Conservative limit for $50 plan
    
    async def initialize(self) -> None:
        """Initialize Helius RPC client."""
        try:
            # Get Helius API key
            self.api_key = await self.credential_manager.get_credential('HELIUS_API_KEY')
            self.base_url = f"{self.MAINNET_URL}/?api-key={self.api_key}"
            
            # Initialize HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                headers={'Content-Type': 'application/json'}
            )
            
            # Test connection
            await self._test_connection()
            
            self.logger.info("Initialized Helius RPC collector successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Helius RPC collector: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.session:
            await self.session.close()
    
    async def _test_connection(self) -> None:
        """Test Helius RPC connection."""
        try:
            response = await self._make_rpc_call('getHealth')
            if response.get('result') == 'ok':
                self.logger.info("Helius RPC connection test successful")
            else:
                raise Exception(f"Health check failed: {response}")
        except Exception as e:
            self.logger.error(f"Helius RPC connection test failed: {e}")
            raise
    
    async def _make_rpc_call(
        self, 
        method: str, 
        params: List[Any] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Make a single RPC call with caching."""
        # Check request limit
        if self.request_count >= self.daily_request_limit:
            raise Exception("Daily request limit reached")
        
        # Create cache key
        cache_key = None
        if use_cache:
            cache_key = hashlib.md5(
                f"{method}:{json.dumps(params or [])}".encode()
            ).hexdigest()
            
            # Check cache
            if cache_key in self.request_cache:
                cached_data, timestamp = self.request_cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    return cached_data
        
        # Prepare RPC request
        payload = {
            'jsonrpc': '2.0',
            'id': int(time.time() * 1000),
            'method': method,
            'params': params or []
        }
        
        try:
            async with self.session.post(self.base_url, json=payload) as response:
                self.request_count += 1
                
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
                
                result = await response.json()
                
                # Check for RPC errors
                if 'error' in result:
                    raise Exception(f"RPC error: {result['error']}")
                
                # Cache successful result
                if use_cache and cache_key:
                    self.request_cache[cache_key] = (result, time.time())
                
                return result
                
        except Exception as e:
            self.logger.error(f"RPC call failed for {method}: {e}")
            raise
    
    async def _batch_rpc_calls(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make multiple RPC calls in a single batch request."""
        if not calls:
            return []
        
        # Check request limit
        if self.request_count + len(calls) > self.daily_request_limit:
            raise Exception("Batch would exceed daily request limit")
        
        try:
            async with self.session.post(self.base_url, json=calls) as response:
                self.request_count += len(calls)
                
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
                
                results = await response.json()
                return results if isinstance(results, list) else [results]
                
        except Exception as e:
            self.logger.error(f"Batch RPC call failed: {e}")
            raise
    
    async def collect_onchain_data(
        self, 
        symbol: str, 
        network: str = 'solana',
        since: Optional[datetime] = None
    ) -> Optional[OnChainData]:
        """Collect on-chain analytics data for Solana."""
        if network.lower() != 'solana':
            self.logger.warning(f"Unsupported network: {network}")
            return None
        
        try:
            # Get token mint address
            token_address = self.POPULAR_TOKENS.get(symbol.upper())
            if not token_address:
                self.logger.warning(f"Token address not found for {symbol}")
                return None
            
            # Collect various on-chain metrics
            current_time = datetime.utcnow()
            
            # Batch multiple requests for efficiency
            batch_calls = [
                {
                    'jsonrpc': '2.0',
                    'id': 1,
                    'method': 'getTokenSupply',
                    'params': [token_address]
                },
                {
                    'jsonrpc': '2.0',
                    'id': 2,
                    'method': 'getTokenAccountsByMint',
                    'params': [
                        token_address,
                        {'programId': 'TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA'}
                    ]
                }
            ]
            
            # Get network stats
            batch_calls.extend([
                {
                    'jsonrpc': '2.0',
                    'id': 3,
                    'method': 'getEpochInfo',
                    'params': []
                },
                {
                    'jsonrpc': '2.0',
                    'id': 4,
                    'method': 'getRecentBlockhash',
                    'params': []
                }
            ])
            
            results = await self._batch_rpc_calls(batch_calls)
            
            # Process results
            token_supply = results[0].get('result', {}).get('value', {}).get('uiAmount', 0)
            token_accounts = results[1].get('result', {}).get('value', [])
            epoch_info = results[2].get('result', {})
            
            # Calculate metrics
            active_addresses = len(token_accounts)
            
            # Get recent transaction data
            transaction_data = await self._get_recent_transactions(token_address, limit=100)
            
            transaction_count = len(transaction_data)
            total_value = sum(tx.get('amount', 0) for tx in transaction_data)
            avg_transaction_value = total_value / transaction_count if transaction_count > 0 else 0
            
            # Detect whale movements (large transactions)
            whale_movements = [
                tx for tx in transaction_data 
                if tx.get('amount', 0) > 1000000  # $1M threshold
            ]
            
            onchain_data = OnChainData(
                timestamp=current_time,
                symbol=symbol,
                network=network,
                transaction_count=transaction_count,
                active_addresses=active_addresses,
                total_value_transferred=total_value,
                average_transaction_value=avg_transaction_value,
                whale_movements=whale_movements
            )
            
            self.logger.info(f"Collected on-chain data for {symbol}: {transaction_count} transactions")
            return onchain_data
            
        except Exception as e:
            self.logger.error(f"Failed to collect on-chain data for {symbol}: {e}")
            return None
    
    async def _get_recent_transactions(
        self, 
        token_address: str, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent transactions for a token."""
        try:
            # Get signatures for the token account
            response = await self._make_rpc_call(
                'getSignaturesForAddress',
                [token_address, {'limit': limit}]
            )
            
            signatures = response.get('result', [])
            
            if not signatures:
                return []
            
            # Get transaction details in batches
            transactions = []
            batch_size = 20  # Helius batch limit
            
            for i in range(0, len(signatures), batch_size):
                batch_sigs = signatures[i:i + batch_size]
                sig_list = [sig['signature'] for sig in batch_sigs]
                
                batch_calls = [
                    {
                        'jsonrpc': '2.0',
                        'id': idx,
                        'method': 'getTransaction',
                        'params': [sig, 'json']
                    }
                    for idx, sig in enumerate(sig_list)
                ]
                
                batch_results = await self._batch_rpc_calls(batch_calls)
                
                for result in batch_results:
                    tx_data = result.get('result')
                    if tx_data:
                        parsed_tx = self._parse_transaction(tx_data)
                        if parsed_tx:
                            transactions.append(parsed_tx)
            
            return transactions
            
        except Exception as e:
            self.logger.error(f"Failed to get recent transactions: {e}")
            return []
    
    def _parse_transaction(self, tx_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse Solana transaction data."""
        try:
            meta = tx_data.get('meta', {})
            transaction = tx_data.get('transaction', {})
            
            # Extract basic transaction info
            parsed_tx = {
                'signature': transaction.get('signatures', [None])[0],
                'timestamp': datetime.fromtimestamp(
                    tx_data.get('blockTime', 0), 
                    tz=timezone.utc
                ),
                'fee': meta.get('fee', 0),
                'success': meta.get('err') is None,
                'amount': 0  # Will be calculated from balance changes
            }
            
            # Calculate net balance changes
            pre_balances = meta.get('preBalances', [])
            post_balances = meta.get('postBalances', [])
            
            if len(pre_balances) == len(post_balances):
                balance_changes = [
                    post - pre 
                    for pre, post in zip(pre_balances, post_balances)
                ]
                parsed_tx['amount'] = sum(abs(change) for change in balance_changes)
            
            return parsed_tx
            
        except Exception as e:
            self.logger.error(f"Failed to parse transaction: {e}")
            return None
    
    async def collect_whale_movements(
        self, 
        network: str = 'solana',
        min_value: float = 1000000  # $1M minimum
    ) -> List[Dict[str, Any]]:
        """Collect large transaction movements across popular tokens."""
        whale_movements = []
        
        try:
            # Check whale movements for all popular tokens
            for symbol, token_address in self.POPULAR_TOKENS.items():
                recent_txs = await self._get_recent_transactions(token_address, limit=50)
                
                for tx in recent_txs:
                    if tx.get('amount', 0) >= min_value:
                        whale_movement = {
                            'symbol': symbol,
                            'token_address': token_address,
                            'signature': tx.get('signature'),
                            'timestamp': tx.get('timestamp'),
                            'amount': tx.get('amount'),
                            'fee': tx.get('fee'),
                            'success': tx.get('success')
                        }
                        whale_movements.append(whale_movement)
            
            # Sort by amount (descending)
            whale_movements.sort(key=lambda x: x.get('amount', 0), reverse=True)
            
            self.logger.info(f"Detected {len(whale_movements)} whale movements")
            return whale_movements
            
        except Exception as e:
            self.logger.error(f"Failed to collect whale movements: {e}")
            return []
    
    async def get_supported_symbols(self) -> List[str]:
        """Get supported Solana tokens."""
        return list(self.POPULAR_TOKENS.keys())
    
    async def get_supported_timeframes(self) -> List[str]:
        """Get supported timeframes for on-chain data."""
        return ['1m', '5m', '15m', '1h', '4h', '1d']
    
    async def collect_ohlcv(
        self, 
        symbol: str, 
        timeframe: str, 
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Any]:
        """On-chain collectors don't provide OHLCV data."""
        return []
    
    async def collect_orderbook(self, symbol: str, limit: Optional[int] = None) -> None:
        """On-chain collectors don't provide order book data."""
        return None
    
    def get_cost_optimization_stats(self) -> Dict[str, Any]:
        """Get cost optimization statistics."""
        daily_cost = (self.request_count / self.daily_request_limit) * 50  # $50 plan
        
        return {
            'requests_made': self.request_count,
            'daily_limit': self.daily_request_limit,
            'utilization_percentage': (self.request_count / self.daily_request_limit) * 100,
            'estimated_daily_cost': daily_cost,
            'cache_hit_ratio': len(self.request_cache) / max(self.request_count, 1),
            'average_batch_size': len(self.batch_queue) / max(1, self.request_count / 10)
        }
    
    async def optimize_polling_intervals(self) -> None:
        """Dynamically adjust polling intervals based on account activity."""
        try:
            for symbol, token_address in self.POPULAR_TOKENS.items():
                # Get recent activity
                recent_txs = await self._get_recent_transactions(token_address, limit=10)
                
                # Calculate activity score
                if not recent_txs:
                    activity_score = 0
                else:
                    recent_count = len([
                        tx for tx in recent_txs 
                        if tx.get('timestamp', datetime.min) > 
                        (datetime.utcnow() - timedelta(minutes=10))
                    ])
                    activity_score = recent_count
                
                # Determine polling interval
                if activity_score >= 5:
                    interval = self.polling_intervals['high_activity']
                elif activity_score >= 2:
                    interval = self.polling_intervals['medium_activity']
                else:
                    interval = self.polling_intervals['low_activity']
                
                self.account_activity[symbol] = {
                    'activity_score': activity_score,
                    'polling_interval': interval,
                    'last_updated': datetime.utcnow()
                }
            
            self.logger.info(f"Updated polling intervals for {len(self.account_activity)} tokens")
            
        except Exception as e:
            self.logger.error(f"Failed to optimize polling intervals: {e}")


class SolanaNetworkMonitor:
    """
    Monitor Solana network health and performance metrics.
    Provides insights into network congestion and validator performance.
    """
    
    def __init__(self, helius_collector: HeliusRPCCollector):
        self.collector = helius_collector
        self.logger = get_logger("solana_network_monitor")
    
    async def get_network_metrics(self) -> Dict[str, Any]:
        """Get comprehensive Solana network metrics."""
        try:
            # Batch network health requests
            batch_calls = [
                {
                    'jsonrpc': '2.0',
                    'id': 1,
                    'method': 'getEpochInfo',
                    'params': []
                },
                {
                    'jsonrpc': '2.0',
                    'id': 2,
                    'method': 'getBlockHeight',
                    'params': []
                },
                {
                    'jsonrpc': '2.0',
                    'id': 3,
                    'method': 'getSlot',
                    'params': []
                },
                {
                    'jsonrpc': '2.0',
                    'id': 4,
                    'method': 'getVersion',
                    'params': []
                }
            ]
            
            results = await self.collector._batch_rpc_calls(batch_calls)
            
            epoch_info = results[0].get('result', {})
            block_height = results[1].get('result', 0)
            current_slot = results[2].get('result', 0)
            version_info = results[3].get('result', {})
            
            # Calculate network metrics
            network_metrics = {
                'timestamp': datetime.utcnow(),
                'epoch': epoch_info.get('epoch', 0),
                'slot_index': epoch_info.get('slotIndex', 0),
                'slots_in_epoch': epoch_info.get('slotsInEpoch', 0),
                'block_height': block_height,
                'current_slot': current_slot,
                'solana_version': version_info.get('solana-core', 'unknown'),
                'epoch_progress': (
                    epoch_info.get('slotIndex', 0) / 
                    max(epoch_info.get('slotsInEpoch', 1), 1)
                ) * 100
            }
            
            return network_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get network metrics: {e}")
            return {}
    
    async def check_network_congestion(self) -> Dict[str, Any]:
        """Check for network congestion indicators."""
        try:
            # Get recent performance samples
            response = await self.collector._make_rpc_call(
                'getRecentPerformanceSamples',
                [10]  # Last 10 samples
            )
            
            samples = response.get('result', [])
            
            if not samples:
                return {'congestion_level': 'unknown'}
            
            # Calculate average metrics
            avg_tps = sum(sample.get('numTransactions', 0) for sample in samples) / len(samples)
            avg_slot_time = sum(sample.get('samplePeriodSecs', 0) for sample in samples) / len(samples)
            
            # Determine congestion level
            if avg_tps < 1000:
                congestion_level = 'high'
            elif avg_tps < 2000:
                congestion_level = 'medium'
            else:
                congestion_level = 'low'
            
            return {
                'congestion_level': congestion_level,
                'average_tps': avg_tps,
                'average_slot_time': avg_slot_time,
                'samples_analyzed': len(samples)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to check network congestion: {e}")
            return {'congestion_level': 'unknown'}