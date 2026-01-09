"""
Temporal Fusion Transformer with Multi-Scale Attention (Model 6)
Advanced transformer architecture with multi-timeframe analysis and regime-aware processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math
from ..base_model import BaseModel


class MultiScaleAttention(nn.Module):
    """Multi-scale attention mechanism across different time horizons."""
    
    def __init__(self, d_model: int, num_heads: int, scales: List[int] = [1, 5, 15, 60]):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.scales = scales
        self.head_dim = d_model // num_heads
        
        # Separate attention layers for each scale
        self.scale_attentions = nn.ModuleDict({
            f"scale_{scale}": nn.MultiheadAttention(
                d_model, num_heads, dropout=0.1, batch_first=True
            ) for scale in scales
        })
        
        # Scale fusion layer
        self.scale_fusion = nn.Linear(d_model * len(scales), d_model)
        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Multi-scale attended features
        """
        batch_size, seq_len, _ = x.shape
        scale_outputs = []
        
        for i, scale in enumerate(self.scales):
            # Apply temporal pooling for different scales
            if scale > 1:
                # Downsample for larger scales, clamping padding to valid range
                kernel = min(scale, seq_len)
                stride = max(1, scale // 2)
                padding = min(scale // 2, kernel // 2)
                pooled_x = F.avg_pool1d(
                    x.transpose(1, 2),
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                ).transpose(1, 2)
            else:
                pooled_x = x
                
            # Apply attention at this scale
            attn_out, _ = self.scale_attentions[f"scale_{scale}"](
                pooled_x, pooled_x, pooled_x, attn_mask=mask
            )
            
            # Upsample back to original sequence length if needed
            if scale > 1 and attn_out.size(1) != seq_len:
                attn_out = F.interpolate(
                    attn_out.transpose(1, 2),
                    size=seq_len,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
                
            scale_outputs.append(attn_out)
        
        # Weighted fusion of scale outputs
        weighted_scales = []
        for i, scale_out in enumerate(scale_outputs):
            weighted_scales.append(self.scale_weights[i] * scale_out)
        
        # Concatenate and fuse
        fused = torch.cat(scale_outputs, dim=-1)
        output = self.scale_fusion(fused)
        
        return output


class RegimeAwareEncoder(nn.Module):
    """Encoder layer with regime awareness for market condition detection."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.multi_scale_attention = MultiScaleAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Regime detection components
        self.regime_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 4)  # 4 market regimes: bull, bear, sideways, volatile
        )
        
        self.regime_gate = nn.Linear(4, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
        Returns:
            Tuple of (encoded_features, regime_probabilities)
        """
        # Multi-scale attention
        attn_out = self.multi_scale_attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        # Regime detection
        regime_logits = self.regime_detector(x.mean(dim=1))  # Global average pooling
        regime_probs = F.softmax(regime_logits, dim=-1)
        
        # Apply regime-aware gating
        regime_gate = self.regime_gate(regime_probs).unsqueeze(1)
        x = self.norm3(x * torch.sigmoid(regime_gate))
        
        return x, regime_probs


class TemporalFusionTransformer(BaseModel):
    """
    Temporal Fusion Transformer with Multi-Scale Attention (Model 6)
    
    Advanced transformer architecture for multi-timeframe cryptocurrency analysis
    with regime-aware neural processing and sophisticated attention mechanisms.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Model configuration
        self.d_model = config.get('d_model', 256)
        self.num_heads = config.get('num_heads', 8)
        self.num_layers = config.get('num_layers', 6)
        self.d_ff = config.get('d_ff', 1024)
        self.dropout = config.get('dropout', 0.1)
        self.num_classes = config.get('num_classes', 5)
        
        # Input projection
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        
        # Positional encoding with multiple time scales
        self.pos_encoding = TemporalPositionalEncoding(
            self.d_model, max_seq_len=config.get('max_seq_len', 1000)
        )
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            RegimeAwareEncoder(self.d_model, self.num_heads, self.d_ff, self.dropout)
            for _ in range(self.num_layers)
        ])
        
        # Output layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, self.num_classes)
        )
        
        # Regime prediction head
        self.regime_classifier = nn.Linear(self.d_model, 4)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights with Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the temporal fusion transformer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim) or (batch_size, input_dim)
            
        Returns:
            Prediction logits of shape (batch_size, num_classes)
        """
        # Handle both sequence and single timestep inputs
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
            
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through encoder layers
        regime_probs_all = []
        for layer in self.encoder_layers:
            x, regime_probs = layer(x)
            regime_probs_all.append(regime_probs)
            
        # Global pooling and classification
        x_pooled = self.global_pool(x.transpose(1, 2)).squeeze(-1)
        predictions = self.classifier(x_pooled)
        
        return predictions
    
    def predict(self, x: torch.Tensor) -> Dict:
        """Make prediction with confidence and regime information."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            confidence = torch.max(probs, dim=-1)[0].item()
            prediction = torch.argmax(logits, dim=-1).item()
            
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probs.cpu().numpy(),
            'logits': logits.cpu().numpy()
        }
    
    def get_regime_analysis(self, x: torch.Tensor) -> Dict:
        """Get detailed regime analysis for market conditions."""
        self.eval()
        with torch.no_grad():
            if x.dim() == 2:
                x = x.unsqueeze(1)
                
            x = self.input_projection(x)
            x = self.pos_encoding(x)
            
            regime_analysis = []
            for i, layer in enumerate(self.encoder_layers):
                x, regime_probs = layer(x)
                regime_names = ['bull', 'bear', 'sideways', 'volatile']
                regime_dict = {
                    name: prob.item() 
                    for name, prob in zip(regime_names, regime_probs[0])
                }
                regime_analysis.append({
                    'layer': i,
                    'regime_probabilities': regime_dict,
                    'dominant_regime': regime_names[torch.argmax(regime_probs[0]).item()]
                })
                
        return regime_analysis


class TemporalPositionalEncoding(nn.Module):
    """Positional encoding with multiple temporal scales."""
    
    def __init__(self, d_model: int, max_seq_len: int = 1000):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        # Multiple frequency scales for different time patterns
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


# Model factory function
def create_temporal_fusion_transformer(input_dim: int = 29) -> TemporalFusionTransformer:
    """Create a Temporal Fusion Transformer model with optimized configuration."""
    config = {
        'input_dim': input_dim,
        'd_model': 256,
        'num_heads': 8,
        'num_layers': 4,  # Reduced for efficiency
        'd_ff': 512,
        'dropout': 0.1,
        'num_classes': 5,
        'max_seq_len': 200
    }
    return TemporalFusionTransformer(config)