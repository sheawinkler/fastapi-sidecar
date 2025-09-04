"""
Data validation utilities for the crypto trading system.
Ensures data integrity and quality across all data sources.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..models import (
    OHLCV, TechnicalIndicators, SentimentData, 
    OnChainData, DataQualityMetrics, DataSource
)
from ...utils.logger import get_logger


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    quality_score: float
    corrected_data: Optional[Any] = None


class DataValidator:
    """
    Comprehensive data validation and quality assurance.
    Implements multiple validation strategies for different data types.
    """
    
    def __init__(self):
        self.logger = get_logger("data_validator")
        
        # Validation thresholds
        self.price_change_threshold = 0.5  # 50% max price change
        self.volume_spike_threshold = 10.0  # 10x volume spike
        self.sentiment_range = (-1.0, 1.0)
        self.rsi_range = (0, 100)
        
        # Data completeness requirements
        self.min_data_points = {
            'ohlcv_1m': 10,      # At least 10 minutes of data
            'ohlcv_5m': 12,      # At least 1 hour of data
            'ohlcv_1h': 24,      # At least 1 day of data
            'sentiment': 5,      # At least 5 sentiment points
            'technical': 20      # At least 20 data points for indicators
        }
    
    def validate_ohlcv_data(self, ohlcv_list: List[OHLCV]) -> ValidationResult:
        """Validate OHLCV data integrity and quality."""
        errors = []
        warnings = []
        corrected_data = []
        
        if not ohlcv_list:
            return ValidationResult(
                is_valid=False,
                errors=["No OHLCV data provided"],
                warnings=[],
                quality_score=0.0
            )
        
        # Sort by timestamp for proper validation
        sorted_data = sorted(ohlcv_list, key=lambda x: x.timestamp)
        
        for i, candle in enumerate(sorted_data):
            candle_errors = []
            candle_warnings = []
            
            try:
                # Basic OHLCV validation (already done in __post_init__)
                candle.__post_init__()
            except ValueError as e:
                candle_errors.append(f"Candle {i}: {str(e)}")
                continue
            
            # Price consistency checks
            if i > 0:
                prev_candle = sorted_data[i - 1]
                
                # Check for extreme price changes
                price_change = abs(candle.close - prev_candle.close) / prev_candle.close
                if price_change > self.price_change_threshold:
                    candle_warnings.append(
                        f"Candle {i}: Extreme price change {price_change:.2%}"
                    )
                
                # Check timestamp ordering
                if candle.timestamp <= prev_candle.timestamp:
                    candle_errors.append(
                        f"Candle {i}: Timestamp not in chronological order"
                    )
                
                # Check for gaps in data
                expected_gap = self._get_expected_timeframe_gap(candle.timeframe)
                actual_gap = (candle.timestamp - prev_candle.timestamp).total_seconds()
                
                if expected_gap and abs(actual_gap - expected_gap) > expected_gap * 0.1:
                    candle_warnings.append(
                        f"Candle {i}: Unexpected time gap: {actual_gap}s vs expected {expected_gap}s"
                    )
            
            # Volume validation
            if candle.volume < 0:
                candle_errors.append(f"Candle {i}: Negative volume")
            elif i > 0 and candle.volume > prev_candle.volume * self.volume_spike_threshold:
                candle_warnings.append(
                    f"Candle {i}: Unusual volume spike: {candle.volume / prev_candle.volume:.1f}x"
                )
            
            # Check for zero prices (invalid)
            if any(price <= 0 for price in [candle.open, candle.high, candle.low, candle.close]):
                candle_errors.append(f"Candle {i}: Contains zero or negative prices")
            
            # Add to results
            errors.extend(candle_errors)
            warnings.extend(candle_warnings)
            
            # Keep valid candles for corrected data
            if not candle_errors:
                corrected_data.append(candle)
        
        # Calculate quality score
        total_candles = len(ohlcv_list)
        valid_candles = len(corrected_data)
        warning_penalty = len(warnings) * 0.02  # 2% penalty per warning
        
        quality_score = max(0.0, (valid_candles / total_candles) - warning_penalty)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            quality_score=quality_score,
            corrected_data=corrected_data if corrected_data != ohlcv_list else None
        )
    
    def validate_technical_indicators(self, indicators: List[TechnicalIndicators]) -> ValidationResult:
        """Validate technical indicators data."""
        errors = []
        warnings = []
        corrected_data = []
        
        if not indicators:
            return ValidationResult(
                is_valid=False,
                errors=["No technical indicators provided"],
                warnings=[],
                quality_score=0.0
            )
        
        for i, indicator in enumerate(indicators):
            indicator_errors = []
            indicator_warnings = []
            
            # RSI validation (0-100 range)
            if indicator.rsi is not None:
                if not (0 <= indicator.rsi <= 100):
                    indicator_errors.append(f"Indicator {i}: RSI out of range: {indicator.rsi}")
            
            # MACD validation (reasonable values)
            if indicator.macd is not None and abs(indicator.macd) > 1000:
                indicator_warnings.append(f"Indicator {i}: Extreme MACD value: {indicator.macd}")
            
            # Moving average validation (positive values)
            ma_fields = ['sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26']
            for field in ma_fields:
                value = getattr(indicator, field, None)
                if value is not None and value <= 0:
                    indicator_errors.append(f"Indicator {i}: Invalid {field}: {value}")
            
            # Bollinger Bands validation (upper > middle > lower)
            if all(getattr(indicator, f'bb_{band}', None) is not None 
                   for band in ['upper', 'middle', 'lower']):
                if not (indicator.bb_upper > indicator.bb_middle > indicator.bb_lower):
                    indicator_errors.append(
                        f"Indicator {i}: Invalid Bollinger Bands order"
                    )
            
            # Stochastic validation (0-100 range)
            if indicator.stoch_k is not None and not (0 <= indicator.stoch_k <= 100):
                indicator_errors.append(f"Indicator {i}: Stoch %K out of range: {indicator.stoch_k}")
            
            if indicator.stoch_d is not None and not (0 <= indicator.stoch_d <= 100):
                indicator_errors.append(f"Indicator {i}: Stoch %D out of range: {indicator.stoch_d}")
            
            # Williams %R validation (-100 to 0 range)
            if indicator.williams_r is not None and not (-100 <= indicator.williams_r <= 0):
                indicator_errors.append(f"Indicator {i}: Williams %R out of range: {indicator.williams_r}")
            
            # Add to results
            errors.extend(indicator_errors)
            warnings.extend(indicator_warnings)
            
            if not indicator_errors:
                corrected_data.append(indicator)
        
        # Calculate quality score
        total_indicators = len(indicators)
        valid_indicators = len(corrected_data)
        warning_penalty = len(warnings) * 0.02
        
        quality_score = max(0.0, (valid_indicators / total_indicators) - warning_penalty)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            quality_score=quality_score,
            corrected_data=corrected_data if corrected_data != indicators else None
        )
    
    def validate_sentiment_data(self, sentiment_list: List[SentimentData]) -> ValidationResult:
        """Validate sentiment data."""
        errors = []
        warnings = []
        corrected_data = []
        
        if not sentiment_list:
            return ValidationResult(
                is_valid=False,
                errors=["No sentiment data provided"],
                warnings=[],
                quality_score=0.0
            )
        
        for i, sentiment in enumerate(sentiment_list):
            sentiment_errors = []
            sentiment_warnings = []
            
            try:
                # Basic validation (already done in __post_init__)
                sentiment.__post_init__()
            except ValueError as e:
                sentiment_errors.append(f"Sentiment {i}: {str(e)}")
                continue
            
            # Additional sentiment-specific validation
            # Check for reasonable text length
            if sentiment.raw_text and len(sentiment.raw_text) < 5:
                sentiment_warnings.append(f"Sentiment {i}: Very short text")
            
            # Check for extreme confidence with neutral sentiment
            if abs(sentiment.sentiment_score) < 0.1 and sentiment.confidence > 0.8:
                sentiment_warnings.append(
                    f"Sentiment {i}: High confidence with neutral sentiment"
                )
            
            # Check for missing keywords with strong sentiment
            if abs(sentiment.sentiment_score) > 0.5 and len(sentiment.keywords) == 0:
                sentiment_warnings.append(
                    f"Sentiment {i}: Strong sentiment with no keywords"
                )
            
            # Check timestamp recency
            age_hours = (datetime.utcnow() - sentiment.timestamp).total_seconds() / 3600
            if age_hours > 168:  # More than 1 week old
                sentiment_warnings.append(f"Sentiment {i}: Data is {age_hours:.1f} hours old")
            
            errors.extend(sentiment_errors)
            warnings.extend(sentiment_warnings)
            
            if not sentiment_errors:
                corrected_data.append(sentiment)
        
        # Calculate quality score
        total_sentiment = len(sentiment_list)
        valid_sentiment = len(corrected_data)
        warning_penalty = len(warnings) * 0.02
        
        quality_score = max(0.0, (valid_sentiment / total_sentiment) - warning_penalty)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            quality_score=quality_score,
            corrected_data=corrected_data if corrected_data != sentiment_list else None
        )
    
    def validate_onchain_data(self, onchain_list: List[OnChainData]) -> ValidationResult:
        """Validate on-chain data."""
        errors = []
        warnings = []
        corrected_data = []
        
        if not onchain_list:
            return ValidationResult(
                is_valid=False,
                errors=["No on-chain data provided"],
                warnings=[],
                quality_score=0.0
            )
        
        for i, onchain in enumerate(onchain_list):
            onchain_errors = []
            onchain_warnings = []
            
            try:
                # Basic validation
                onchain.__post_init__()
            except ValueError as e:
                onchain_errors.append(f"OnChain {i}: {str(e)}")
                continue
            
            # Network-specific validation
            if onchain.network.lower() == 'solana':
                # Solana-specific checks
                if onchain.transaction_count > 1000000:  # Unusually high for single collection
                    onchain_warnings.append(
                        f"OnChain {i}: Very high transaction count: {onchain.transaction_count}"
                    )
                
                if onchain.active_addresses > 10000000:  # Unusually high
                    onchain_warnings.append(
                        f"OnChain {i}: Very high active addresses: {onchain.active_addresses}"
                    )
            
            # Logical consistency checks
            if onchain.average_transaction_value * onchain.transaction_count > onchain.total_value_transferred * 2:
                onchain_warnings.append(
                    f"OnChain {i}: Inconsistent transaction values"
                )
            
            # Check whale movements data
            if onchain.whale_movements:
                whale_errors = []
                for j, whale_tx in enumerate(onchain.whale_movements):
                    if not isinstance(whale_tx, dict):
                        whale_errors.append(f"Whale movement {j}: Invalid format")
                    elif 'amount' not in whale_tx or whale_tx['amount'] < 100000:  # Less than $100K
                        whale_errors.append(f"Whale movement {j}: Amount too small for whale")
                
                if whale_errors:
                    onchain_warnings.extend(whale_errors)
            
            errors.extend(onchain_errors)
            warnings.extend(onchain_warnings)
            
            if not onchain_errors:
                corrected_data.append(onchain)
        
        # Calculate quality score
        total_onchain = len(onchain_list)
        valid_onchain = len(corrected_data)
        warning_penalty = len(warnings) * 0.02
        
        quality_score = max(0.0, (valid_onchain / total_onchain) - warning_penalty)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            quality_score=quality_score,
            corrected_data=corrected_data if corrected_data != onchain_list else None
        )
    
    def _get_expected_timeframe_gap(self, timeframe: str) -> Optional[int]:
        """Get expected gap in seconds for a timeframe."""
        timeframe_gaps = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }
        return timeframe_gaps.get(timeframe)
    
    def detect_outliers(self, data: List[float], method: str = 'iqr') -> List[int]:
        """Detect outliers in numerical data."""
        if len(data) < 10:  # Need sufficient data
            return []
        
        data_array = np.array(data)
        outlier_indices = []
        
        if method == 'iqr':
            q1 = np.percentile(data_array, 25)
            q3 = np.percentile(data_array, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outlier_indices = [
                i for i, value in enumerate(data) 
                if value < lower_bound or value > upper_bound
            ]
        
        elif method == 'zscore':
            mean = np.mean(data_array)
            std = np.std(data_array)
            
            if std > 0:
                z_scores = np.abs((data_array - mean) / std)
                outlier_indices = [i for i, z in enumerate(z_scores) if z > 3]
        
        return outlier_indices
    
    def validate_data_completeness(
        self, 
        data: List[Any], 
        data_type: str, 
        timeframe: str = None
    ) -> ValidationResult:
        """Validate data completeness requirements."""
        errors = []
        warnings = []
        
        # Determine minimum data requirements
        min_required = self.min_data_points.get('technical', 10)  # Default
        
        if data_type == 'ohlcv' and timeframe:
            key = f'ohlcv_{timeframe}'
            min_required = self.min_data_points.get(key, self.min_data_points.get('ohlcv_1m', 10))
        elif data_type in self.min_data_points:
            min_required = self.min_data_points[data_type]
        
        # Check data count
        if len(data) < min_required:
            errors.append(
                f"Insufficient data points: {len(data)} < {min_required} required"
            )
        elif len(data) < min_required * 2:
            warnings.append(
                f"Low data count: {len(data)} (recommended: {min_required * 2}+)"
            )
        
        # Check for data gaps (timestamps only)
        if hasattr(data[0] if data else None, 'timestamp'):
            gaps = self._find_data_gaps(data, timeframe)
            if gaps:
                warnings.extend([f"Data gap detected: {gap}" for gap in gaps[:5]])  # Max 5 warnings
        
        quality_score = max(0.0, len(data) / max(min_required, 1))
        quality_score = min(1.0, quality_score)  # Cap at 1.0
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            quality_score=quality_score
        )
    
    def _find_data_gaps(self, data: List[Any], timeframe: str = None) -> List[str]:
        """Find gaps in timestamped data."""
        if len(data) < 2 or not timeframe:
            return []
        
        expected_gap = self._get_expected_timeframe_gap(timeframe)
        if not expected_gap:
            return []
        
        gaps = []
        sorted_data = sorted(data, key=lambda x: x.timestamp)
        
        for i in range(1, len(sorted_data)):
            actual_gap = (sorted_data[i].timestamp - sorted_data[i-1].timestamp).total_seconds()
            
            if actual_gap > expected_gap * 1.5:  # 50% tolerance
                gap_duration = actual_gap - expected_gap
                gaps.append(
                    f"{sorted_data[i-1].timestamp} to {sorted_data[i].timestamp} "
                    f"({gap_duration:.0f}s gap)"
                )
        
        return gaps
    
    def generate_quality_report(
        self, 
        validation_results: Dict[str, ValidationResult]
    ) -> Dict[str, Any]:
        """Generate comprehensive data quality report."""
        total_errors = sum(len(result.errors) for result in validation_results.values())
        total_warnings = sum(len(result.warnings) for result in validation_results.values())
        
        # Calculate overall quality score
        quality_scores = [result.quality_score for result in validation_results.values()]
        overall_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        # Determine quality grade
        if overall_quality >= 0.9:
            quality_grade = 'A'
        elif overall_quality >= 0.8:
            quality_grade = 'B'
        elif overall_quality >= 0.7:
            quality_grade = 'C'
        elif overall_quality >= 0.6:
            quality_grade = 'D'
        else:
            quality_grade = 'F'
        
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_quality_score': overall_quality,
            'quality_grade': quality_grade,
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'validation_results': {
                data_type: {
                    'is_valid': result.is_valid,
                    'quality_score': result.quality_score,
                    'error_count': len(result.errors),
                    'warning_count': len(result.warnings),
                    'errors': result.errors[:5],  # First 5 errors
                    'warnings': result.warnings[:5]  # First 5 warnings
                }
                for data_type, result in validation_results.items()
            },
            'recommendations': []
        }
        
        # Add recommendations
        if overall_quality < 0.7:
            report['recommendations'].append("Data quality is below acceptable threshold. Review data sources.")
        
        if total_errors > 0:
            report['recommendations'].append(f"Fix {total_errors} critical data errors.")
        
        if total_warnings > 10:
            report['recommendations'].append(f"Address {total_warnings} data quality warnings.")
        
        # Data type specific recommendations
        for data_type, result in validation_results.items():
            if result.quality_score < 0.6:
                report['recommendations'].append(f"Improve {data_type} data quality (score: {result.quality_score:.2f})")
        
        return report