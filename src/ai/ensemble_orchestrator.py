"""
Complete Ensemble Orchestration System
Utility-based dynamic weighting with 25-minute windows and real-time ensemble predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import time
import json
from datetime import datetime, timedelta

# Import all models
from .models.reinforcement_learning.executive_auxiliary_agent import create_executive_auxiliary_agent
from .models.transformers.cross_modal_temporal_fusion import create_cross_modal_temporal_fusion
from .models.variational_autoencoders.progressive_denoising_vae import create_progressive_denoising_vae
from .models.quantile_models.quantile_ensemble import create_quantile_ensemble
from .models.nlp.crypto_bert_sentiment import create_crypto_bert_sentiment
from .models.transformers.temporal_fusion_transformer import create_temporal_fusion_transformer
from .models.pattern_generation.cnn_gan_autoencoder import create_cnn_gan_autoencoder
from .models.random_forest.generalized_rf_var import create_generalized_rf_var
from .models.portfolio_optimization.dynamic_portfolio_optimizer import create_dynamic_portfolio_optimizer
from .models.volatility_prediction.multi_modal_volatility_predictor import create_multi_modal_volatility_predictor

from ..utils.logger import system_logger, audit_logger, performance_logger


class UtilityMetricCalculator:
    """Calculate utility metrics for ensemble weighting based on trading profitability."""
    
    def __init__(self):
        # Utility matrix based on trading profitability
        self.utility_matrix = np.array([
            [2, 0, 0, 0, -2],   # Actual class 0 (strong sell)
            [1, 0, 0, 0, -1],   # Actual class 1 (sell)
            [0, 0, 0, 0, 0],    # Actual class 2 (hold)
            [-1, 0, 0, 0, 1],   # Actual class 3 (buy)
            [-2, 0, 0, 0, 2]    # Actual class 4 (strong buy)
        ])
        
    def calculate_utility(self, predictions: List[int], outcomes: List[int]) -> float:
        """
        Calculate utility metric based on trading profitability.
        
        Args:
            predictions: List of predicted classes
            outcomes: List of actual outcome classes
            
        Returns:
            Utility score (higher is better)
        """
        if len(predictions) != len(outcomes) or len(predictions) == 0:
            return 0.0
            
        total_utility = 0.0
        for pred, actual in zip(predictions, outcomes):
            # Ensure indices are within bounds
            pred = max(0, min(4, pred))
            actual = max(0, min(4, actual))
            total_utility += self.utility_matrix[actual, pred]
            
        return total_utility / len(predictions)


class ModelPerformanceTracker:
    """Track performance metrics for individual models."""
    
    def __init__(self, window_size: int = 25):
        self.window_size = window_size
        self.predictions_history = {}
        self.outcomes_history = deque(maxlen=window_size)
        self.utility_history = {}
        self.accuracy_history = {}
        self.confidence_history = {}
        self.timestamp_history = deque(maxlen=window_size)
        
        self.utility_calculator = UtilityMetricCalculator()
        
    def add_prediction(self, model_name: str, prediction: int, confidence: float, timestamp: datetime = None):
        """Add a prediction from a model."""
        if model_name not in self.predictions_history:
            self.predictions_history[model_name] = deque(maxlen=self.window_size)
            self.utility_history[model_name] = deque(maxlen=self.window_size)
            self.accuracy_history[model_name] = deque(maxlen=self.window_size)
            self.confidence_history[model_name] = deque(maxlen=self.window_size)
            
        self.predictions_history[model_name].append(prediction)
        self.confidence_history[model_name].append(confidence)
        
        if timestamp is None:
            timestamp = datetime.now()
        self.timestamp_history.append(timestamp)
        
    def add_outcome(self, outcome: int, timestamp: datetime = None):
        """Add actual outcome."""
        self.outcomes_history.append(outcome)
        
        if timestamp is None:
            timestamp = datetime.now()
            
        # Update utility and accuracy for all models
        for model_name in self.predictions_history:
            if len(self.predictions_history[model_name]) > 0:
                # Get recent predictions for utility calculation
                recent_predictions = list(self.predictions_history[model_name])
                recent_outcomes = list(self.outcomes_history)
                
                min_length = min(len(recent_predictions), len(recent_outcomes))
                if min_length > 0:
                    # Calculate utility
                    utility = self.utility_calculator.calculate_utility(
                        recent_predictions[-min_length:],
                        recent_outcomes[-min_length:]
                    )
                    self.utility_history[model_name].append(utility)
                    
                    # Calculate accuracy
                    correct_predictions = sum(
                        1 for p, o in zip(recent_predictions[-min_length:], recent_outcomes[-min_length:])
                        if p == o
                    )
                    accuracy = correct_predictions / min_length
                    self.accuracy_history[model_name].append(accuracy)
                    
    def get_recent_utility(self, model_name: str) -> float:
        """Get recent utility score for a model."""
        if model_name not in self.utility_history or len(self.utility_history[model_name]) == 0:
            return 0.0
        return np.mean(list(self.utility_history[model_name]))
    
    def get_recent_accuracy(self, model_name: str) -> float:
        """Get recent accuracy for a model."""
        if model_name not in self.accuracy_history or len(self.accuracy_history[model_name]) == 0:
            return 0.2  # Default for 5-class classification
        return np.mean(list(self.accuracy_history[model_name]))
    
    def get_recent_confidence(self, model_name: str) -> float:
        """Get recent average confidence for a model."""
        if model_name not in self.confidence_history or len(self.confidence_history[model_name]) == 0:
            return 0.5
        return np.mean(list(self.confidence_history[model_name]))


class EnsembleOrchestrator:
    """
    Complete Ensemble Orchestration System with utility-based dynamic weighting.
    
    Implements the Numin framework with 25-minute windows and exponential moving average
    for optimal ensemble performance in cryptocurrency trading.
    """
    
    def __init__(self, input_dim: int = 29, device: torch.device = None):
        self.input_dim = input_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Performance tracking
        self.window_size = 25  # 25-minute windows for optimal performance
        self.alpha = 2 / (self.window_size + 1)  # EMA smoothing factor
        self.performance_tracker = ModelPerformanceTracker(self.window_size)
        
        # Model weights and initialization
        self.model_weights = {}
        self.model_names = [
            'executive_auxiliary_agent',
            'cross_modal_temporal_fusion',
            'progressive_denoising_vae',
            'quantile_ensemble',
            'crypto_bert_sentiment',
            'temporal_fusion_transformer',
            'cnn_gan_autoencoder',
            'generalized_rf_var',
            'dynamic_portfolio_optimizer',
            'multi_modal_volatility_predictor'
        ]
        
        # Initialize all models
        self.models = self._initialize_models()
        
        # Initialize equal weights
        for model_name in self.model_names:
            self.model_weights[model_name] = 1.0 / len(self.model_names)
            
        # Ensemble performance tracking
        self.ensemble_predictions_history = deque(maxlen=1000)
        self.ensemble_accuracy_history = deque(maxlen=100)
        
        system_logger.info(f"Ensemble Orchestrator initialized with {len(self.models)} models")
        
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize all 10 AI models."""
        models = {}
        
        try:
            # Priority 1 Models
            models['executive_auxiliary_agent'] = create_executive_auxiliary_agent(self.input_dim).to(self.device)
            models['cross_modal_temporal_fusion'] = create_cross_modal_temporal_fusion(self.input_dim).to(self.device)
            
            # Priority 2 Models  
            models['progressive_denoising_vae'] = create_progressive_denoising_vae(self.input_dim).to(self.device)
            models['quantile_ensemble'] = create_quantile_ensemble(self.input_dim).to(self.device)
            models['crypto_bert_sentiment'] = create_crypto_bert_sentiment(self.input_dim).to(self.device)
            
            # Advanced Models
            models['temporal_fusion_transformer'] = create_temporal_fusion_transformer(self.input_dim).to(self.device)
            models['cnn_gan_autoencoder'] = create_cnn_gan_autoencoder(self.input_dim).to(self.device)
            models['generalized_rf_var'] = create_generalized_rf_var(self.input_dim).to(self.device)
            models['dynamic_portfolio_optimizer'] = create_dynamic_portfolio_optimizer(self.input_dim).to(self.device)
            models['multi_modal_volatility_predictor'] = create_multi_modal_volatility_predictor(self.input_dim).to(self.device)
            
            system_logger.info("All 10 AI models initialized successfully")
            
        except Exception as e:
            system_logger.error(f"Error initializing models: {str(e)}")
            raise
            
        return models
    
    def update_weights(self):
        """Update model weights using utility-based EMA."""
        for model_name in self.model_names:
            recent_utility = self.performance_tracker.get_recent_utility(model_name)
            
            # Update weight using exponential moving average
            if model_name in self.model_weights:
                self.model_weights[model_name] = (
                    self.alpha * recent_utility + 
                    (1 - self.alpha) * self.model_weights[model_name]
                )
            else:
                self.model_weights[model_name] = recent_utility
                
        # Normalize weights (softmax for positivity)
        weight_values = list(self.model_weights.values())
        weight_tensor = torch.tensor(weight_values, dtype=torch.float32)
        normalized_weights = F.softmax(weight_tensor, dim=0)
        
        for i, model_name in enumerate(self.model_names):
            self.model_weights[model_name] = normalized_weights[i].item()
            
        performance_logger.info(f"Updated model weights: {self.model_weights}")
    
    def predict_ensemble(self, x: torch.Tensor, update_history: bool = True) -> Dict[str, Any]:
        """
        Make ensemble prediction with confidence aggregation.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            update_history: Whether to update prediction history
            
        Returns:
            Comprehensive ensemble prediction results
        """
        start_time = time.time()
        
        # Ensure input is on correct device
        x = x.to(self.device)
        
        # Individual model predictions
        model_predictions = {}
        model_confidences = {}
        model_probabilities = {}
        
        for model_name, model in self.models.items():
            try:
                model.eval()
                with torch.no_grad():
                    # Handle different model interfaces
                    if hasattr(model, 'predict'):
                        result = model.predict(x)
                        prediction = result['prediction']
                        confidence = result['confidence']
                        probabilities = result.get('probabilities', np.array([0.2]*5))
                    else:
                        # Fallback to forward pass
                        logits = model(x)
                        probs = F.softmax(logits, dim=-1)
                        confidence = torch.max(probs, dim=-1)[0].item()
                        prediction = torch.argmax(logits, dim=-1).item()
                        probabilities = probs.cpu().numpy()
                    
                    model_predictions[model_name] = prediction
                    model_confidences[model_name] = confidence
                    model_probabilities[model_name] = probabilities
                    
                    # Update performance tracking
                    if update_history:
                        self.performance_tracker.add_prediction(
                            model_name, prediction, confidence
                        )
                        
            except Exception as e:
                system_logger.error(f"Error in model {model_name}: {str(e)}")
                # Use default values for failed models
                model_predictions[model_name] = 2  # Hold
                model_confidences[model_name] = 0.0
                model_probabilities[model_name] = np.array([0.2]*5)
        
        # Weighted ensemble prediction
        weighted_probs = np.zeros(5)
        total_weight = 0.0
        confidence_weighted_sum = 0.0
        
        for model_name in model_predictions:
            weight = self.model_weights.get(model_name, 0.0)
            confidence = model_confidences[model_name]
            probs = model_probabilities[model_name]
            
            # Confidence-weighted ensemble
            final_weight = weight * (1 + confidence)  # Boost high-confidence predictions
            
            if len(probs) == 5:  # Ensure correct probability dimensions
                weighted_probs += final_weight * probs.flatten()[:5]
            total_weight += final_weight
            confidence_weighted_sum += weight * confidence
        
        # Normalize probabilities
        if total_weight > 0:
            weighted_probs /= total_weight
            ensemble_confidence = confidence_weighted_sum / sum(self.model_weights.values())
        else:
            weighted_probs = np.array([0.2]*5)  # Uniform distribution
            ensemble_confidence = 0.0
            
        # Final ensemble prediction
        ensemble_prediction = np.argmax(weighted_probs)
        
        # Calculate prediction latency
        prediction_latency = (time.time() - start_time) * 1000  # milliseconds
        
        # Ensemble result
        ensemble_result = {
            'prediction': int(ensemble_prediction),
            'confidence': float(ensemble_confidence),
            'probabilities': weighted_probs.tolist(),
            'individual_predictions': model_predictions,
            'individual_confidences': model_confidences,
            'model_weights': self.model_weights.copy(),
            'prediction_latency_ms': prediction_latency,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update ensemble history
        if update_history:
            self.ensemble_predictions_history.append(ensemble_result)
            
        # Log performance if latency target exceeded
        if prediction_latency > 10.0:  # Target <10ms
            performance_logger.warning(f"High prediction latency: {prediction_latency:.2f}ms")
            
        audit_logger.info(f"Ensemble prediction: {ensemble_prediction}, confidence: {ensemble_confidence:.3f}")
        
        return ensemble_result
    
    def add_outcome(self, outcome: int):
        """Add actual outcome for model weight updates."""
        self.performance_tracker.add_outcome(outcome)
        self.update_weights()
        
        # Update ensemble accuracy
        if len(self.ensemble_predictions_history) > 0:
            recent_pred = self.ensemble_predictions_history[-1]['prediction']
            is_correct = (recent_pred == outcome)
            self.ensemble_accuracy_history.append(is_correct)
            
        performance_logger.info(f"Added outcome: {outcome}, updated model weights")
    
    def get_ensemble_performance(self) -> Dict[str, Any]:
        """Get comprehensive ensemble performance metrics."""
        # Recent accuracy
        recent_accuracy = np.mean(self.ensemble_accuracy_history) if self.ensemble_accuracy_history else 0.0
        
        # Individual model performance
        model_performance = {}
        for model_name in self.model_names:
            model_performance[model_name] = {
                'weight': self.model_weights.get(model_name, 0.0),
                'utility': self.performance_tracker.get_recent_utility(model_name),
                'accuracy': self.performance_tracker.get_recent_accuracy(model_name),
                'confidence': self.performance_tracker.get_recent_confidence(model_name)
            }
        
        # Prediction latency statistics
        if self.ensemble_predictions_history:
            latencies = [p['prediction_latency_ms'] for p in self.ensemble_predictions_history]
            avg_latency = np.mean(latencies)
            max_latency = np.max(latencies)
            p95_latency = np.percentile(latencies, 95)
        else:
            avg_latency = max_latency = p95_latency = 0.0
        
        return {
            'ensemble_accuracy': recent_accuracy,
            'model_performance': model_performance,
            'prediction_latency': {
                'average_ms': avg_latency,
                'max_ms': max_latency,
                'p95_ms': p95_latency,
                'target_met': avg_latency < 10.0
            },
            'total_predictions': len(self.ensemble_predictions_history),
            'window_size': self.window_size,
            'alpha': self.alpha
        }
    
    def save_model_weights(self, filepath: str):
        """Save current model weights to file."""
        weight_data = {
            'weights': self.model_weights,
            'timestamp': datetime.now().isoformat(),
            'window_size': self.window_size,
            'alpha': self.alpha
        }
        
        with open(filepath, 'w') as f:
            json.dump(weight_data, f, indent=2)
            
        system_logger.info(f"Model weights saved to {filepath}")
    
    def load_model_weights(self, filepath: str):
        """Load model weights from file."""
        try:
            with open(filepath, 'r') as f:
                weight_data = json.load(f)
                
            self.model_weights = weight_data['weights']
            system_logger.info(f"Model weights loaded from {filepath}")
            
        except Exception as e:
            system_logger.error(f"Error loading model weights: {str(e)}")
            # Keep current weights on error
    
    def get_model_diagnostics(self) -> Dict[str, Any]:
        """Get detailed model diagnostics and health status."""
        diagnostics = {}
        
        for model_name, model in self.models.items():
            try:
                # Basic model info
                param_count = sum(p.numel() for p in model.parameters() if hasattr(model, 'parameters'))
                is_training = model.training if hasattr(model, 'training') else False
                
                # Performance metrics
                utility = self.performance_tracker.get_recent_utility(model_name)
                accuracy = self.performance_tracker.get_recent_accuracy(model_name)
                confidence = self.performance_tracker.get_recent_confidence(model_name)
                weight = self.model_weights.get(model_name, 0.0)
                
                diagnostics[model_name] = {
                    'parameter_count': param_count,
                    'is_training': is_training,
                    'current_weight': weight,
                    'utility_score': utility,
                    'accuracy': accuracy,
                    'avg_confidence': confidence,
                    'health_status': 'healthy' if utility > -0.5 and accuracy > 0.15 else 'needs_attention'
                }
                
            except Exception as e:
                diagnostics[model_name] = {
                    'error': str(e),
                    'health_status': 'error'
                }
        
        return diagnostics


# Factory function for ensemble orchestrator
def create_ensemble_orchestrator(input_dim: int = 29, device: torch.device = None) -> EnsembleOrchestrator:
    """Create an Ensemble Orchestrator with all 10 AI models."""
    return EnsembleOrchestrator(input_dim, device)