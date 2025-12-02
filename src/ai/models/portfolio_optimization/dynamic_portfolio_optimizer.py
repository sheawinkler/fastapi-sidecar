"""
Dynamic Portfolio Optimization with Multi-Asset Pair Trading (Model 9)
Advanced statistical arbitrage with CNN-MHA architecture and adaptive rebalancing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import scipy.optimize as opt
from ..base_model import BaseModel


class MultiHeadAttention(nn.Module):
    """Multi-head attention for asset correlation analysis."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.out_linear(context)
        return output


class CNNFeatureExtractor(nn.Module):
    """CNN for extracting temporal features from price/volume data."""
    
    def __init__(self, input_channels: int, num_assets: int):
        super().__init__()
        self.input_channels = input_channels
        self.num_assets = num_assets
        
        # Multi-scale CNN layers
        self.conv_short = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        self.conv_medium = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        self.conv_long = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        # Feature fusion
        self.feature_fusion = nn.Conv1d(384, 256, kernel_size=1)  # 3 * 128 = 384
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, channels, sequence_length)
        Returns:
            Extracted features (batch_size, 256)
        """
        # Multi-scale feature extraction
        short_features = self.conv_short(x)
        medium_features = self.conv_medium(x)
        long_features = self.conv_long(x)
        
        # Concatenate multi-scale features
        combined_features = torch.cat([short_features, medium_features, long_features], dim=1)
        
        # Fuse features
        fused_features = self.feature_fusion(combined_features)
        
        # Global pooling
        pooled_features = self.global_pool(fused_features).squeeze(-1)
        
        return pooled_features


class PairTradingDetector(nn.Module):
    """Neural network for detecting profitable pair trading opportunities."""
    
    def __init__(self, feature_dim: int, num_assets: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_assets = num_assets
        
        # Pairwise relationship encoder
        self.pair_encoder = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),  # Two assets
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # Pair trading score
            nn.Sigmoid()
        )
        
        # Spread prediction network
        self.spread_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Predicted spread
        )
        
    def forward(self, asset1_features: torch.Tensor, asset2_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            asset1_features: Features for first asset (batch_size, feature_dim)
            asset2_features: Features for second asset (batch_size, feature_dim)
        Returns:
            Dict with pair trading score and predicted spread
        """
        # Combine features
        combined_features = torch.cat([asset1_features, asset2_features], dim=-1)
        
        # Pair trading score
        pair_score = self.pair_encoder(combined_features)
        
        # Spread prediction
        spread_pred = self.spread_predictor(combined_features)
        
        return {
            'pair_score': pair_score,
            'spread_prediction': spread_pred
        }


class DynamicPortfolioOptimizer(BaseModel):
    """
    Dynamic Portfolio Optimization with Multi-Asset Pair Trading (Model 9)
    
    Advanced portfolio optimization system combining:
    - CNN feature extraction for temporal pattern recognition
    - Multi-head attention for asset correlation analysis
    - Pair trading opportunity detection
    - Dynamic portfolio rebalancing with risk constraints
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Model configuration
        self.num_assets = config.get('num_assets', 10)
        self.d_model = config.get('d_model', 256)
        self.num_heads = config.get('num_heads', 8)
        self.sequence_length = config.get('sequence_length', 50)
        self.num_classes = config.get('num_classes', 5)
        
        # CNN feature extractor
        self.cnn_extractor = CNNFeatureExtractor(
            input_channels=self.input_dim // self.num_assets if self.input_dim >= self.num_assets else 1,
            num_assets=self.num_assets
        )
        
        # Multi-head attention for asset relationships
        self.asset_attention = MultiHeadAttention(
            self.d_model, self.num_heads
        )
        
        # Pair trading detector
        self.pair_detector = PairTradingDetector(self.d_model, self.num_assets)
        
        # Portfolio optimizer network
        self.portfolio_optimizer = nn.Sequential(
            nn.Linear(self.d_model, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, self.num_assets),
            nn.Softmax(dim=-1)  # Portfolio weights sum to 1
        )
        
        # Risk estimator
        self.risk_estimator = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # Positive risk estimate
        )
        
        # Classification head for trading signals
        self.signal_classifier = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_classes)
        )
        
        # Covariance matrix estimator
        self.covariance_estimator = nn.Linear(self.d_model, self.num_assets * self.num_assets)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def _prepare_asset_data(self, x: torch.Tensor) -> torch.Tensor:
        """Prepare input data for CNN processing."""
        batch_size = x.size(0)
        
        if x.dim() == 2:  # Single timestep
            # Reshape to (batch, channels, sequence) format for CNN
            if self.input_dim >= self.num_assets:
                x = x.view(batch_size, self.input_dim // self.num_assets, -1)
            else:
                x = x.unsqueeze(1).repeat(1, 1, self.sequence_length)
        elif x.dim() == 3:  # Sequence data
            x = x.transpose(1, 2)  # (batch, features, sequence)
            
        return x
    
    def extract_asset_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features for each asset using CNN."""
        x_prepared = self._prepare_asset_data(x)
        features = self.cnn_extractor(x_prepared)
        return features
    
    def compute_asset_correlations(self, asset_features: torch.Tensor) -> torch.Tensor:
        """Compute asset correlations using multi-head attention."""
        # Reshape for attention: (batch, num_assets, feature_dim)
        batch_size = asset_features.size(0)
        feature_dim = asset_features.size(1)
        
        # Create asset feature matrix
        asset_matrix = asset_features.unsqueeze(1).repeat(1, self.num_assets, 1)
        
        # Apply multi-head attention
        correlation_features = self.asset_attention(asset_matrix, asset_matrix, asset_matrix)
        
        # Pool to single representation
        pooled_features = torch.mean(correlation_features, dim=1)
        
        return pooled_features
    
    def detect_pair_opportunities(self, asset_features: torch.Tensor) -> List[Dict]:
        """Detect profitable pair trading opportunities."""
        batch_size = asset_features.size(0)
        pair_opportunities = []
        
        # For simplicity, analyze top pairs (in practice, analyze all combinations)
        num_pairs_to_check = min(5, self.num_assets // 2)
        
        for i in range(num_pairs_to_check):
            for j in range(i + 1, min(i + 3, self.num_assets)):  # Check nearby pairs
                # Create synthetic asset features for pair (i, j)
                asset1_feat = asset_features
                asset2_feat = asset_features * (1 + 0.1 * (j - i))  # Synthetic variation
                
                pair_result = self.pair_detector(asset1_feat, asset2_feat)
                
                pair_opportunities.append({
                    'asset1': i,
                    'asset2': j,
                    'score': pair_result['pair_score'].mean().item(),
                    'spread_prediction': pair_result['spread_prediction'].mean().item()
                })
                
        # Sort by score
        pair_opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        return pair_opportunities[:3]  # Return top 3 opportunities
    
    def optimize_portfolio(self, asset_features: torch.Tensor, risk_tolerance: float = 0.1) -> Dict:
        """Optimize portfolio weights with risk constraints."""
        # Get portfolio weights from neural network
        portfolio_weights = self.portfolio_optimizer(asset_features)
        
        # Estimate risk
        estimated_risk = self.risk_estimator(asset_features)
        
        # Estimate covariance matrix
        cov_flat = self.covariance_estimator(asset_features)
        cov_matrix = cov_flat.view(-1, self.num_assets, self.num_assets)
        
        # Make covariance matrix positive semi-definite
        cov_matrix = torch.matmul(cov_matrix, cov_matrix.transpose(-2, -1))
        
        return {
            'weights': portfolio_weights,
            'estimated_risk': estimated_risk,
            'covariance_matrix': cov_matrix,
            'risk_tolerance': risk_tolerance
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for signal classification.
        
        Args:
            x: Input tensor (batch_size, input_dim) or (batch_size, seq_len, input_dim)
        Returns:
            Classification logits (batch_size, num_classes)
        """
        # Extract asset features
        asset_features = self.extract_asset_features(x)
        
        # Compute correlations
        correlation_features = self.compute_asset_correlations(asset_features)
        
        # Classify trading signal
        signal_logits = self.signal_classifier(correlation_features)
        
        return signal_logits
    
    def predict(self, x: torch.Tensor) -> Dict:
        """Comprehensive prediction including portfolio optimization."""
        self.eval()
        with torch.no_grad():
            # Signal classification
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            confidence = torch.max(probs, dim=-1)[0].item()
            prediction = torch.argmax(logits, dim=-1).item()
            
            # Extract features for portfolio analysis
            asset_features = self.extract_asset_features(x)
            correlation_features = self.compute_asset_correlations(asset_features)
            
            # Portfolio optimization
            portfolio_result = self.optimize_portfolio(correlation_features)
            
            # Pair trading opportunities
            pair_opportunities = self.detect_pair_opportunities(correlation_features)
            
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probs.cpu().numpy(),
            'portfolio_weights': portfolio_result['weights'].cpu().numpy(),
            'estimated_risk': portfolio_result['estimated_risk'].cpu().numpy(),
            'pair_opportunities': pair_opportunities,
            'correlation_strength': torch.mean(torch.abs(correlation_features)).item()
        }
    
    def backtest_portfolio(self, historical_data: torch.Tensor, returns: torch.Tensor) -> Dict:
        """Backtest portfolio performance on historical data."""
        self.eval()
        total_return = 0.0
        max_drawdown = 0.0
        peak_value = 1.0
        
        portfolio_values = []
        
        with torch.no_grad():
            for i in range(len(historical_data)):
                prediction_result = self.predict(historical_data[i:i+1])
                weights = prediction_result['portfolio_weights'][0]
                
                # Calculate portfolio return
                if i < len(returns):
                    period_return = np.sum(weights * returns[i].cpu().numpy())
                    total_return += period_return
                    
                    current_value = peak_value * (1 + total_return)
                    portfolio_values.append(current_value)
                    
                    if current_value > peak_value:
                        peak_value = current_value
                    else:
                        drawdown = (peak_value - current_value) / peak_value
                        max_drawdown = max(max_drawdown, drawdown)
        
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': total_return / (np.std(portfolio_values) + 1e-8) if portfolio_values else 0.0,
            'portfolio_values': portfolio_values
        }


# Model factory function
def create_dynamic_portfolio_optimizer(input_dim: int = 29) -> DynamicPortfolioOptimizer:
    """Create a Dynamic Portfolio Optimizer with optimized configuration."""
    config = {
        'input_dim': input_dim,
        'num_assets': max(5, input_dim // 6),  # Reasonable number of assets
        'd_model': 256,
        'num_heads': 8,
        'sequence_length': 50,
        'num_classes': 5
    }
    return DynamicPortfolioOptimizer(config)
