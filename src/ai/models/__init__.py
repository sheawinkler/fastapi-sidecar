"""
AI Models Module

Implementation of 10 advanced AI ensemble models for cryptocurrency trading.

Priority 1 Models (COMPLETED):
- ExecutiveAuxiliaryAgent: Hierarchical RL with dual architecture
- CrossModalTemporalFusion: Transformer with attention weighting

Priority 2 Models (COMPLETED):
- ProgressiveDenoisingVAE: VAE with financial pattern recognition
- FunctionalQuantileEnsemble: Data-driven quantile prediction system
- CryptoBERTSentimentFusion: Enhanced NLP for crypto sentiment fusion

Each model implements the BaseTradingModel interface for consistent integration
with the ensemble orchestration system.
"""

from .base_model import BaseModel
from .reinforcement_learning.executive_auxiliary_agent import ExecutiveAuxiliaryAgent
from .transformers.cross_modal_temporal_fusion import CrossModalTemporalFusion
from .variational_autoencoders.progressive_denoising_vae import ProgressiveDenoisingVAE
from .quantile_models.quantile_ensemble import FunctionalQuantileEnsemble
from .nlp.crypto_bert_sentiment import CryptoBERTSentimentFusion

__all__ = [
    'BaseModel',
    'ExecutiveAuxiliaryAgent', 
    'CrossModalTemporalFusion',
    'ProgressiveDenoisingVAE',
    'FunctionalQuantileEnsemble',
    'CryptoBERTSentimentFusion'
]