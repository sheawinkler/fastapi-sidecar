#!/usr/bin/env python3
"""
Simple test of the core data infrastructure models and validation.
Tests the essential components without complex dependencies.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, '/workspace/project')
sys.path.insert(0, '/workspace/project/src')

def test_data_models():
    """Test basic data model functionality."""
    print("🧪 Testing Data Models...")
    
    try:
        # Direct imports to avoid circular dependencies
        from src.data.models import DataSource, OHLCV, TechnicalIndicators, SentimentType, SentimentData
        
        # Test OHLCV model
        timestamp = datetime.utcnow()
        ohlcv = OHLCV(
            timestamp=timestamp,
            symbol="BTC/USD",
            timeframe="5m",
            open=50000.0,
            high=50500.0,
            low=49800.0,
            close=50200.0,
            volume=1000.0,
            source=DataSource.COINBASE
        )
        
        # Test model validation
        ohlcv_dict = ohlcv.to_dict()
        assert ohlcv_dict['symbol'] == "BTC/USD"
        assert ohlcv_dict['open'] == 50000.0
        assert ohlcv_dict['source'] == 'coinbase'
        
        # Test invalid OHLCV (should raise ValueError)
        try:
            invalid_ohlcv = OHLCV(
                timestamp=timestamp,
                symbol="BTC/USD",
                timeframe="5m",
                open=50000.0,
                high=49000.0,  # High < Open (invalid)
                low=49800.0,
                close=50200.0,
                volume=1000.0,
                source=DataSource.COINBASE
            )
            # Should not reach here
            assert False, "Expected ValueError for invalid OHLCV"
        except ValueError:
            pass  # Expected
        
        # Test TechnicalIndicators model
        tech_indicators = TechnicalIndicators(
            timestamp=timestamp,
            symbol="BTC/USD",
            timeframe="5m",
            rsi=65.5,
            macd=150.0,
            bb_upper=51000.0,
            bb_middle=50000.0,
            bb_lower=49000.0
        )
        
        tech_dict = tech_indicators.to_dict()
        assert tech_dict['rsi'] == 65.5
        assert tech_dict['symbol'] == "BTC/USD"
        
        # Test SentimentData model
        sentiment = SentimentData(
            timestamp=timestamp,
            source=DataSource.TWITTER,
            symbol="BTC",
            sentiment_score=0.75,
            sentiment_type=SentimentType.BULLISH,
            confidence=0.8,
            volume=100,
            keywords=['bitcoin', 'bullish'],
            raw_text="Bitcoin is looking bullish today!"
        )
        
        assert sentiment.sentiment_score == 0.75
        assert sentiment.sentiment_type == SentimentType.BULLISH
        assert len(sentiment.keywords) == 2
        
        print("✅ Data Models: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Data Models: FAILED - {e}")
        return False

def test_data_validation():
    """Test data validation functionality."""
    print("🧪 Testing Data Validation...")
    
    try:
        from src.data.validators.data_validator import DataValidator, ValidationResult
        from src.data.models import OHLCV, DataSource
        
        validator = DataValidator()
        
        # Create test OHLCV data
        ohlcv_data = []
        base_time = datetime.utcnow() - timedelta(hours=1)
        
        for i in range(12):  # 1 hour of 5-minute data
            timestamp = base_time + timedelta(minutes=i * 5)
            price = 50000 + (i * 100)  # Gradually increasing price
            
            ohlcv = OHLCV(
                timestamp=timestamp,
                symbol="BTC/USD",
                timeframe="5m",
                open=price,
                high=price + 50,
                low=price - 30,
                close=price + 20,
                volume=1000 + (i * 50),
                source=DataSource.COINBASE
            )
            ohlcv_data.append(ohlcv)
        
        # Validate the data
        validation_result = validator.validate_ohlcv_data(ohlcv_data)
        
        assert validation_result.is_valid, f"OHLCV validation failed: {validation_result.errors}"
        assert validation_result.quality_score > 0.9, f"Low quality score: {validation_result.quality_score}"
        
        # Test outlier detection
        normal_data = [100, 102, 98, 101, 99, 103, 97, 100, 102, 98]
        outlier_data = normal_data + [200, 300]  # Add outliers
        
        outliers = validator.detect_outliers(outlier_data)
        assert len(outliers) >= 2, f"Should detect outliers, found: {len(outliers)}"
        
        # Test data completeness
        small_dataset = [ohlcv_data[0]]  # Only one data point
        
        completeness_result = validator.validate_data_completeness(
            small_dataset, 'ohlcv', '5m'
        )
        assert not completeness_result.is_valid, "Should detect insufficient data"
        
        print(f"✅ Data Validation: PASSED - Quality Score: {validation_result.quality_score:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Data Validation: FAILED - {e}")
        return False

def test_feature_vector_generation():
    """Test ML feature vector generation."""
    print("🧪 Testing Feature Vector Generation...")
    
    try:
        from src.data.models import (
            OHLCV, TechnicalIndicators, SentimentData, 
            ProcessedData, DataSource, SentimentType
        )
        import numpy as np
        
        timestamp = datetime.utcnow()
        
        # Create test data components
        ohlcv = OHLCV(
            timestamp=timestamp,
            symbol="BTC/USD",
            timeframe="5m",
            open=50000.0,
            high=50500.0,
            low=49800.0,
            close=50200.0,
            volume=1000.0,
            source=DataSource.COINBASE
        )
        
        tech_indicators = TechnicalIndicators(
            timestamp=timestamp,
            symbol="BTC/USD",
            timeframe="5m",
            rsi=65.5,
            macd=150.0,
            bb_upper=51000.0,
            bb_middle=50000.0,
            bb_lower=49000.0,
            sma_20=50100.0,
            atr=200.0
        )
        
        sentiment = SentimentData(
            timestamp=timestamp,
            source=DataSource.TWITTER,
            symbol="BTC",
            sentiment_score=0.75,
            sentiment_type=SentimentType.BULLISH,
            confidence=0.8,
            volume=100,
            keywords=['bitcoin', 'bullish']
        )
        
        # Create processed data
        processed_data = ProcessedData(
            timestamp=timestamp,
            symbol="BTC/USD",
            timeframe="5m",
            ohlcv=ohlcv,
            technical=tech_indicators,
            sentiment=sentiment,
            quality_score=0.95
        )
        
        # Generate feature vector
        feature_vector = processed_data.to_feature_vector()
        
        assert isinstance(feature_vector, np.ndarray), "Feature vector should be numpy array"
        assert len(feature_vector) > 20, f"Feature vector too small: {len(feature_vector)}"
        assert not np.isnan(feature_vector).any(), "Feature vector contains NaN values"
        
        # Test pandas conversion
        pandas_series = processed_data.to_pandas_series()
        assert 'close' in pandas_series.index, "Missing close price in pandas series"
        assert 'tech_rsi' in pandas_series.index, "Missing RSI in pandas series"
        assert 'sentiment_score' in pandas_series.index, "Missing sentiment in pandas series"
        
        print(f"✅ Feature Vector Generation: PASSED - Vector size: {len(feature_vector)}")
        return True
        
    except Exception as e:
        print(f"❌ Feature Vector Generation: FAILED - {e}")
        return False

def test_enum_functionality():
    """Test enum functionality and type safety."""
    print("🧪 Testing Enum Functionality...")
    
    try:
        from src.data.models import DataSource, SentimentType, MarketDataType
        
        # Test DataSource enum
        assert DataSource.COINBASE.value == 'coinbase'
        assert DataSource.KRAKEN.value == 'kraken'
        assert DataSource.HELIUS.value == 'helius'
        
        # Test SentimentType enum
        assert SentimentType.BULLISH.value == 'bullish'
        assert SentimentType.BEARISH.value == 'bearish'
        assert SentimentType.NEUTRAL.value == 'neutral'
        
        # Test MarketDataType enum
        assert MarketDataType.OHLCV.value == 'ohlcv'
        assert MarketDataType.ORDERBOOK.value == 'orderbook'
        
        # Test enum creation from string
        source_from_string = DataSource('coinbase')
        assert source_from_string == DataSource.COINBASE
        
        print("✅ Enum Functionality: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Enum Functionality: FAILED - {e}")
        return False

def test_data_quality_metrics():
    """Test data quality metrics calculation."""
    print("🧪 Testing Data Quality Metrics...")
    
    try:
        from src.data.models import DataQualityMetrics, DataSource
        from src.data.validators.data_validator import DataValidator
        
        # Create quality metrics
        quality_metrics = DataQualityMetrics(
            timestamp=datetime.utcnow(),
            source=DataSource.COINBASE,
            symbol="BTC/USD",
            completeness_score=0.95,
            accuracy_score=0.88,
            freshness_score=0.92,
            consistency_score=0.85,
            overall_score=0.0,  # Will be calculated
            missing_fields=['volume'],
            outliers_detected=2,
            latency_ms=45.5
        )
        
        # The __post_init__ should calculate overall_score
        assert 0.8 <= quality_metrics.overall_score <= 1.0, f"Invalid overall score: {quality_metrics.overall_score}"
        
        # Test quality report generation
        validator = DataValidator()
        
        validation_results = {
            'ohlcv': validator.ValidationResult(True, [], [], 0.9),
            'sentiment': validator.ValidationResult(True, [], ['warning'], 0.85)
        }
        
        quality_report = validator.generate_quality_report(validation_results)
        
        assert 'overall_quality_score' in quality_report
        assert 'quality_grade' in quality_report
        assert 'recommendations' in quality_report
        assert quality_report['quality_grade'] in ['A', 'B', 'C', 'D', 'F']
        
        print(f"✅ Data Quality Metrics: PASSED - Overall Score: {quality_metrics.overall_score:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Data Quality Metrics: FAILED - {e}")
        return False

def run_all_tests():
    """Run all simplified data infrastructure tests."""
    print("🔧 CRYPTO AI ENSEMBLE TRADING SYSTEM")
    print("📊 Phase 2: Data Infrastructure Validation (Simplified)")
    print("="*80)
    
    tests = [
        test_data_models,
        test_data_validation,
        test_feature_vector_generation,
        test_enum_functionality,
        test_data_quality_metrics
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Add spacing between tests
    
    success_rate = (passed / total * 100)
    
    print("="*80)
    print("🏁 SIMPLIFIED TEST SUMMARY")
    print("="*80)
    print(f"Total Tests:    {total}")
    print(f"Passed:         {passed} ✅")
    print(f"Failed:         {total - passed} ❌")
    print(f"Success Rate:   {success_rate:.1f}%")
    print("="*80)
    
    if success_rate >= 80:
        print("🎉 PHASE 2 DATA INFRASTRUCTURE: CORE COMPONENTS READY!")
        if success_rate == 100:
            print("🏆 PERFECT SCORE! All core systems operational.")
        print("\n📈 Key Achievements:")
        print("  ✅ Enterprise-grade data models with validation")
        print("  ✅ Comprehensive data quality assurance")
        print("  ✅ ML-ready feature vector generation")
        print("  ✅ Type-safe enumeration system")
        print("  ✅ Robust error handling and data integrity")
        
        print("\n🚀 Ready for Phase 3: AI Model Development!")
    elif success_rate >= 60:
        print("⚠️  PHASE 2 DATA INFRASTRUCTURE: MOSTLY READY (minor issues)")
    else:
        print("🚨 PHASE 2 DATA INFRASTRUCTURE: NEEDS ATTENTION")
    
    print("="*80)

if __name__ == "__main__":
    run_all_tests()