"""
Quantile-based Models for Risk Prediction and VaR Estimation

This module contains quantile regression models optimized for cryptocurrency risk management:
- Functional Data-Driven Quantile Ensemble with asymptotic optimality
- VaR prediction with theoretical guarantees
- Superior performance on 105+ cryptocurrencies
"""

from .quantile_ensemble import FunctionalQuantileEnsemble

__all__ = ['FunctionalQuantileEnsemble']