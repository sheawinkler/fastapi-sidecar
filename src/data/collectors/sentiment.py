"""
Social sentiment data collector.
Aggregates sentiment from Twitter/X, Reddit, and news sources.
"""

import asyncio
import aiohttp
import praw
import tweepy
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import json
from textblob import TextBlob
from bs4 import BeautifulSoup

from .base import BaseSentimentCollector, CollectorConfig
from ..models import DataSource, SentimentData, SentimentType, MarketMetrics
from ...utils.logger import get_logger
from ...security.credential_manager import CredentialManager


@dataclass
class SentimentAnalysisResult:
    """Result of sentiment analysis."""
    text: str
    sentiment_score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    keywords: List[str]
    source_platform: str
    timestamp: datetime


class SentimentCollector(BaseSentimentCollector):
    """
    Multi-platform sentiment data collector.
    Integrates with Twitter/X, Reddit, and news APIs.
    """
    
    # Crypto-related keywords for filtering
    CRYPTO_KEYWORDS = [
        'bitcoin', 'btc', 'ethereum', 'eth', 'solana', 'sol', 'cardano', 'ada',
        'polkadot', 'dot', 'chainlink', 'link', 'crypto', 'cryptocurrency',
        'defi', 'nft', 'blockchain', 'altcoin', 'bull', 'bear', 'moon', 'hodl',
        'dip', 'pump', 'dump', 'whale', 'diamond hands', 'paper hands'
    ]
    
    # Sentiment keywords
    BULLISH_KEYWORDS = [
        'moon', 'bullish', 'buy', 'hold', 'hodl', 'diamond hands', 'pump',
        'rocket', 'green', 'profit', 'gains', 'up', 'rise', 'surge', 'rally'
    ]
    
    BEARISH_KEYWORDS = [
        'dump', 'bearish', 'sell', 'crash', 'dip', 'red', 'loss', 'down',
        'fall', 'drop', 'paper hands', 'correction', 'bear market', 'fud'
    ]
    
    def __init__(self, config: CollectorConfig, credential_manager: CredentialManager):
        super().__init__(config, credential_manager)
        
        # API clients
        self.reddit_client = None
        self.twitter_client = None
        self.session = None
        
        # News API configuration
        self.news_sources = [
            'CoinDesk', 'CoinTelegraph', 'Decrypt', 'The Block', 
            'CryptoSlate', 'BeInCrypto', 'U.Today'
        ]
        
        # Fear & Greed Index API
        self.fear_greed_url = 'https://api.alternative.me/fng/'
        
    async def initialize(self) -> None:
        """Initialize sentiment data collectors."""
        try:
            # Initialize HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
            
            # Initialize Reddit client
            await self._initialize_reddit()
            
            # Initialize Twitter client
            await self._initialize_twitter()
            
            self.logger.info("Initialized sentiment collectors successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize sentiment collectors: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.session:
            await self.session.close()
    
    async def _initialize_reddit(self) -> None:
        """Initialize Reddit API client."""
        try:
            reddit_config = {
                'client_id': await self.credential_manager.get_credential('REDDIT_CLIENT_ID'),
                'client_secret': await self.credential_manager.get_credential('REDDIT_CLIENT_SECRET'),
                'user_agent': 'CryptoTradingBot/1.0 by YourUsername'
            }
            
            self.reddit_client = praw.Reddit(**reddit_config)
            
            # Test connection
            subreddit = self.reddit_client.subreddit('CryptoCurrency')
            next(subreddit.hot(limit=1))  # Try to fetch one post
            
            self.logger.info("Reddit API initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Reddit API initialization failed: {e}")
            self.reddit_client = None
    
    async def _initialize_twitter(self) -> None:
        """Initialize Twitter API client."""
        try:
            twitter_config = {
                'bearer_token': await self.credential_manager.get_credential('TWITTER_BEARER_TOKEN'),
                'wait_on_rate_limit': True
            }
            
            self.twitter_client = tweepy.Client(**twitter_config)
            
            # Test connection
            self.twitter_client.get_me()
            
            self.logger.info("Twitter API initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Twitter API initialization failed: {e}")
            self.twitter_client = None
    
    async def collect_sentiment(
        self, 
        symbol: str, 
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[SentimentData]:
        """Collect sentiment data from all sources."""
        all_sentiment = []
        
        # Set default time window
        if since is None:
            since = datetime.utcnow() - timedelta(hours=24)
        
        if limit is None:
            limit = 100
        
        try:
            # Collect from different sources concurrently
            tasks = []
            
            if self.reddit_client:
                tasks.append(self._collect_reddit_sentiment(symbol, since, limit // 3))
            
            if self.twitter_client:
                tasks.append(self._collect_twitter_sentiment(symbol, since, limit // 3))
            
            tasks.append(self._collect_news_sentiment(symbol, since, limit // 3))
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Aggregate results
            for result in results:
                if isinstance(result, list):
                    all_sentiment.extend(result)
                elif isinstance(result, Exception):
                    self.logger.error(f"Sentiment collection task failed: {result}")
            
            # Sort by timestamp (most recent first)
            all_sentiment.sort(key=lambda x: x.timestamp, reverse=True)
            
            self.logger.info(f"Collected {len(all_sentiment)} sentiment data points for {symbol}")
            return all_sentiment[:limit] if limit else all_sentiment
            
        except Exception as e:
            self.logger.error(f"Failed to collect sentiment for {symbol}: {e}")
            return []
    
    async def _collect_reddit_sentiment(
        self, 
        symbol: str, 
        since: datetime, 
        limit: int
    ) -> List[SentimentData]:
        """Collect sentiment from Reddit."""
        if not self.reddit_client:
            return []
        
        sentiment_data = []
        
        try:
            # Define relevant subreddits
            subreddits = [
                'CryptoCurrency', 'Bitcoin', 'ethereum', 'solana', 'altcoin',
                'CryptoMarkets', 'CryptoMoonShots', 'ethtrader', 'btc'
            ]
            
            # Search for posts about the symbol
            search_terms = [symbol.lower(), symbol.upper()]
            if symbol.upper() == 'BTC':
                search_terms.extend(['bitcoin'])
            elif symbol.upper() == 'ETH':
                search_terms.extend(['ethereum'])
            elif symbol.upper() == 'SOL':
                search_terms.extend(['solana'])
            
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit_client.subreddit(subreddit_name)
                    
                    # Get recent hot posts
                    for post in subreddit.hot(limit=20):
                        post_time = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
                        
                        if post_time < since:
                            continue
                        
                        # Check if post is relevant to symbol
                        post_text = f"{post.title} {post.selftext}".lower()
                        if not any(term in post_text for term in search_terms):
                            continue
                        
                        # Analyze sentiment
                        analysis = self._analyze_text_sentiment(post_text)
                        
                        sentiment = SentimentData(
                            timestamp=post_time,
                            source=DataSource.REDDIT,
                            symbol=symbol,
                            sentiment_score=analysis.sentiment_score,
                            sentiment_type=self._score_to_type(analysis.sentiment_score),
                            confidence=analysis.confidence,
                            volume=post.score + post.num_comments,  # Use engagement as volume
                            keywords=analysis.keywords,
                            raw_text=post.title[:200]  # Truncate for storage
                        )
                        
                        sentiment_data.append(sentiment)
                        
                        if len(sentiment_data) >= limit:
                            break
                
                except Exception as e:
                    self.logger.warning(f"Error collecting from r/{subreddit_name}: {e}")
                    continue
            
            self.logger.info(f"Collected {len(sentiment_data)} Reddit sentiment data points")
            return sentiment_data
            
        except Exception as e:
            self.logger.error(f"Reddit sentiment collection failed: {e}")
            return []
    
    async def _collect_twitter_sentiment(
        self, 
        symbol: str, 
        since: datetime, 
        limit: int
    ) -> List[SentimentData]:
        """Collect sentiment from Twitter/X."""
        if not self.twitter_client:
            return []
        
        sentiment_data = []
        
        try:
            # Construct search query
            search_terms = [f"${symbol}", symbol.upper(), symbol.lower()]
            if symbol.upper() == 'BTC':
                search_terms.extend(['#bitcoin', '#btc'])
            elif symbol.upper() == 'ETH':
                search_terms.extend(['#ethereum', '#eth'])
            elif symbol.upper() == 'SOL':
                search_terms.extend(['#solana', '#sol'])
            
            query = ' OR '.join(search_terms)
            query += ' -is:retweet lang:en'  # Filter out retweets, English only
            
            # Search for tweets
            tweets = tweepy.Paginator(
                self.twitter_client.search_recent_tweets,
                query=query,
                max_results=100,
                tweet_fields=['created_at', 'public_metrics', 'context_annotations']
            ).flatten(limit=limit)
            
            for tweet in tweets:
                tweet_time = tweet.created_at
                
                if tweet_time < since.replace(tzinfo=timezone.utc):
                    continue
                
                # Analyze sentiment
                analysis = self._analyze_text_sentiment(tweet.text)
                
                # Calculate engagement volume
                metrics = tweet.public_metrics or {}
                volume = (
                    metrics.get('retweet_count', 0) +
                    metrics.get('like_count', 0) +
                    metrics.get('reply_count', 0)
                )
                
                sentiment = SentimentData(
                    timestamp=tweet_time,
                    source=DataSource.TWITTER,
                    symbol=symbol,
                    sentiment_score=analysis.sentiment_score,
                    sentiment_type=self._score_to_type(analysis.sentiment_score),
                    confidence=analysis.confidence,
                    volume=volume,
                    keywords=analysis.keywords,
                    raw_text=tweet.text[:200]  # Truncate for storage
                )
                
                sentiment_data.append(sentiment)
            
            self.logger.info(f"Collected {len(sentiment_data)} Twitter sentiment data points")
            return sentiment_data
            
        except Exception as e:
            self.logger.error(f"Twitter sentiment collection failed: {e}")
            return []
    
    async def _collect_news_sentiment(
        self, 
        symbol: str, 
        since: datetime, 
        limit: int
    ) -> List[SentimentData]:
        """Collect sentiment from crypto news sources."""
        sentiment_data = []
        
        try:
            # Use NewsAPI if available
            news_api_key = None
            try:
                news_api_key = await self.credential_manager.get_credential('NEWS_API_KEY')
            except:
                pass
            
            if news_api_key:
                sentiment_data.extend(
                    await self._collect_newsapi_sentiment(symbol, since, limit // 2)
                )
            
            # Scrape crypto news websites
            sentiment_data.extend(
                await self._scrape_crypto_news(symbol, since, limit // 2)
            )
            
            return sentiment_data
            
        except Exception as e:
            self.logger.error(f"News sentiment collection failed: {e}")
            return []
    
    async def _collect_newsapi_sentiment(
        self, 
        symbol: str, 
        since: datetime, 
        limit: int
    ) -> List[SentimentData]:
        """Collect news sentiment using NewsAPI."""
        sentiment_data = []
        
        try:
            news_api_key = await self.credential_manager.get_credential('NEWS_API_KEY')
            
            # Construct search query
            query_terms = [symbol, symbol.upper()]
            if symbol.upper() == 'BTC':
                query_terms.append('bitcoin')
            elif symbol.upper() == 'ETH':
                query_terms.append('ethereum')
            
            query = ' OR '.join(query_terms)
            
            # API request
            url = 'https://newsapi.org/v2/everything'
            params = {
                'q': query,
                'sources': ','.join(self.news_sources),
                'from': since.isoformat(),
                'sortBy': 'publishedAt',
                'pageSize': min(limit, 100),
                'apiKey': news_api_key
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = data.get('articles', [])
                    
                    for article in articles:
                        pub_date = datetime.fromisoformat(
                            article['publishedAt'].replace('Z', '+00:00')
                        )
                        
                        # Analyze sentiment of title + description
                        text = f"{article['title']} {article.get('description', '')}"
                        analysis = self._analyze_text_sentiment(text)
                        
                        sentiment = SentimentData(
                            timestamp=pub_date,
                            source=DataSource.NEWS,
                            symbol=symbol,
                            sentiment_score=analysis.sentiment_score,
                            sentiment_type=self._score_to_type(analysis.sentiment_score),
                            confidence=analysis.confidence,
                            volume=1,  # News articles have volume of 1
                            keywords=analysis.keywords,
                            raw_text=article['title'][:200]
                        )
                        
                        sentiment_data.append(sentiment)
                
                else:
                    self.logger.warning(f"NewsAPI request failed: {response.status}")
            
            return sentiment_data
            
        except Exception as e:
            self.logger.error(f"NewsAPI sentiment collection failed: {e}")
            return []
    
    async def _scrape_crypto_news(
        self, 
        symbol: str, 
        since: datetime, 
        limit: int
    ) -> List[SentimentData]:
        """Scrape sentiment from crypto news websites."""
        sentiment_data = []
        
        try:
            # CoinDesk RSS feed
            coindesk_url = 'https://www.coindesk.com/arc/outboundfeeds/rss/'
            
            async with self.session.get(coindesk_url) as response:
                if response.status == 200:
                    content = await response.text()
                    soup = BeautifulSoup(content, 'xml')
                    
                    items = soup.find_all('item')[:limit]
                    
                    for item in items:
                        title = item.find('title')
                        pub_date = item.find('pubDate')
                        description = item.find('description')
                        
                        if not all([title, pub_date]):
                            continue
                        
                        # Parse publication date
                        try:
                            article_time = datetime.strptime(
                                pub_date.text, '%a, %d %b %Y %H:%M:%S %z'
                            )
                        except:
                            continue
                        
                        if article_time < since.replace(tzinfo=timezone.utc):
                            continue
                        
                        # Check if article mentions the symbol
                        text = title.text.lower()
                        if description:
                            text += f" {description.text.lower()}"
                        
                        if symbol.lower() not in text and symbol.upper() not in text:
                            continue
                        
                        # Analyze sentiment
                        analysis = self._analyze_text_sentiment(text)
                        
                        sentiment = SentimentData(
                            timestamp=article_time,
                            source=DataSource.NEWS,
                            symbol=symbol,
                            sentiment_score=analysis.sentiment_score,
                            sentiment_type=self._score_to_type(analysis.sentiment_score),
                            confidence=analysis.confidence,
                            volume=1,
                            keywords=analysis.keywords,
                            raw_text=title.text[:200]
                        )
                        
                        sentiment_data.append(sentiment)
            
            return sentiment_data
            
        except Exception as e:
            self.logger.error(f"News scraping failed: {e}")
            return []
    
    def _analyze_text_sentiment(self, text: str) -> SentimentAnalysisResult:
        """Analyze sentiment of text using multiple methods."""
        try:
            # Clean text
            cleaned_text = self._clean_text(text)
            
            # TextBlob sentiment analysis
            blob = TextBlob(cleaned_text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            # Keyword-based sentiment adjustment
            keyword_sentiment = self._calculate_keyword_sentiment(cleaned_text)
            
            # Combine TextBlob and keyword sentiment
            final_sentiment = (polarity * 0.7) + (keyword_sentiment * 0.3)
            final_sentiment = max(-1.0, min(1.0, final_sentiment))  # Clamp to [-1, 1]
            
            # Calculate confidence (higher subjectivity = lower confidence for sentiment)
            confidence = 1.0 - subjectivity
            confidence = max(0.1, min(1.0, confidence))  # Clamp to [0.1, 1.0]
            
            # Extract relevant keywords
            keywords = self._extract_keywords(cleaned_text)
            
            return SentimentAnalysisResult(
                text=text[:200],
                sentiment_score=final_sentiment,
                confidence=confidence,
                keywords=keywords,
                source_platform='textblob',
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return SentimentAnalysisResult(
                text=text[:200],
                sentiment_score=0.0,
                confidence=0.0,
                keywords=[],
                source_platform='error',
                timestamp=datetime.utcnow()
            )
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text for sentiment analysis."""
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions and hashtags symbols (but keep the words)
        text = re.sub(r'[@#]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.lower()
    
    def _calculate_keyword_sentiment(self, text: str) -> float:
        """Calculate sentiment based on keyword presence."""
        bullish_count = sum(1 for keyword in self.BULLISH_KEYWORDS if keyword in text)
        bearish_count = sum(1 for keyword in self.BEARISH_KEYWORDS if keyword in text)
        
        total_keywords = bullish_count + bearish_count
        
        if total_keywords == 0:
            return 0.0
        
        # Calculate sentiment score
        sentiment = (bullish_count - bearish_count) / total_keywords
        return sentiment
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant crypto keywords from text."""
        found_keywords = []
        
        # Check for crypto keywords
        for keyword in self.CRYPTO_KEYWORDS:
            if keyword in text:
                found_keywords.append(keyword)
        
        # Check for bullish/bearish keywords
        for keyword in self.BULLISH_KEYWORDS + self.BEARISH_KEYWORDS:
            if keyword in text:
                found_keywords.append(keyword)
        
        return list(set(found_keywords))  # Remove duplicates
    
    def _score_to_type(self, score: float) -> SentimentType:
        """Convert sentiment score to sentiment type."""
        if score <= -0.6:
            return SentimentType.EXTREME_FEAR
        elif score <= -0.2:
            return SentimentType.BEARISH
        elif score >= 0.6:
            return SentimentType.EXTREME_GREED
        elif score >= 0.2:
            return SentimentType.BULLISH
        else:
            return SentimentType.NEUTRAL
    
    async def get_fear_greed_index(self) -> Optional[MarketMetrics]:
        """Get Fear & Greed Index from API."""
        try:
            async with self.session.get(self.fear_greed_url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'data' in data and len(data['data']) > 0:
                        latest = data['data'][0]
                        
                        metrics = MarketMetrics(
                            timestamp=datetime.fromtimestamp(
                                int(latest['timestamp']), 
                                tz=timezone.utc
                            ),
                            fear_greed_index=int(latest['value'])
                        )
                        
                        return metrics
                
                else:
                    self.logger.warning(f"Fear & Greed API failed: {response.status}")
                    
        except Exception as e:
            self.logger.error(f"Failed to get Fear & Greed Index: {e}")
        
        return None
    
    async def get_trending_topics(self, limit: int = 10) -> List[str]:
        """Get trending crypto topics."""
        trending_topics = []
        
        try:
            # Get trending from Reddit
            if self.reddit_client:
                subreddit = self.reddit_client.subreddit('CryptoCurrency')
                
                for post in subreddit.hot(limit=20):
                    # Extract topics from post titles
                    title_words = post.title.lower().split()
                    for word in title_words:
                        if word in self.CRYPTO_KEYWORDS and word not in trending_topics:
                            trending_topics.append(word)
                            
                            if len(trending_topics) >= limit:
                                break
            
            # Get trending from Twitter if available
            if self.twitter_client and len(trending_topics) < limit:
                try:
                    trends = self.twitter_client.get_place_trends(1)  # Worldwide trends
                    for trend in trends[0]['trends']:
                        name = trend['name'].lower()
                        if any(keyword in name for keyword in self.CRYPTO_KEYWORDS):
                            if name not in trending_topics:
                                trending_topics.append(name)
                                
                                if len(trending_topics) >= limit:
                                    break
                except Exception as e:
                    self.logger.warning(f"Failed to get Twitter trends: {e}")
            
            return trending_topics[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to get trending topics: {e}")
            return []
    
    async def get_supported_symbols(self) -> List[str]:
        """Get supported symbols for sentiment analysis."""
        return ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'LINK', 'AVAX', 'MATIC']
    
    async def get_supported_timeframes(self) -> List[str]:
        """Get supported timeframes for sentiment data."""
        return ['1h', '4h', '12h', '24h', '7d']
    
    # Required abstract methods (not applicable for sentiment collectors)
    async def collect_ohlcv(self, symbol: str, timeframe: str, 
                          since: Optional[datetime] = None, 
                          limit: Optional[int] = None) -> List[Any]:
        """Sentiment collectors don't provide OHLCV data."""
        return []
    
    async def collect_orderbook(self, symbol: str, limit: Optional[int] = None) -> None:
        """Sentiment collectors don't provide order book data."""
        return None