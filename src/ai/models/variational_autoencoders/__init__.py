"""
Variational Autoencoder Models for Financial Pattern Recognition

This module contains VAE-based models optimized for cryptocurrency trading:
- Progressive Denoising VAE with theoretical guarantees
- Financial signal preservation with pattern recognition
- Anomaly detection for market regime changes
"""

from .progressive_denoising_vae import ProgressiveDenoisingVAE

__all__ = ['ProgressiveDenoisingVAE']