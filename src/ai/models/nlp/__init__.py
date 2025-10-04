"""
Natural Language Processing Models for Cryptocurrency Sentiment Analysis

This module contains NLP models optimized for cryptocurrency sentiment analysis:
- CryptoBERT-Enhanced Multi-Platform Sentiment Fusion
- Custom tokenization and crypto-specific vocabulary
- Multi-platform sentiment integration (Twitter, Reddit, News)
"""

from .crypto_bert_sentiment import CryptoBERTSentimentFusion

__all__ = ['CryptoBERTSentimentFusion']