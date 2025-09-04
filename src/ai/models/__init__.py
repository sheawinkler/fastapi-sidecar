"""
AI Models Module

Implementation of 10 advanced AI ensemble models for cryptocurrency trading.

Priority 1 Models:
- ExecutiveAuxiliaryAgent: Hierarchical RL with dual architecture
- CrossModalTemporalFusion: Transformer with attention weighting
- ProgressiveDenoisingVAE: VAE with financial pattern recognition
- QuantileEnsemble: Data-driven quantile prediction system
- CryptoBERT: Enhanced NLP for crypto sentiment fusion

Each model implements the BaseModel interface for consistent integration
with the ensemble orchestration system.
"""

from .base_model import BaseModel
from .reinforcement_learning.executive_auxiliary_agent import ExecutiveAuxiliaryAgent
from .transformers.cross_modal_temporal_fusion import CrossModalTemporalFusion

# TODO: Implement remaining Priority 2 models
# from .variational_autoencoders.progressive_denoising_vae import ProgressiveDenoisingVAE
# from .quantile_models.functional_quantile_ensemble import QuantileEnsemble
# from .nlp.cryptobert_sentiment_fusion import CryptoBERTSentimentFusion

__all__ = [
    'BaseModel',
    'ExecutiveAuxiliaryAgent', 
    'CrossModalTemporalFusion',
    # 'ProgressiveDenoisingVAE',
    # 'QuantileEnsemble',
    # 'CryptoBERTSentimentFusion'
]