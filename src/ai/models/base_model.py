"""
Base Model Interface for AI Ensemble Trading System

Defines the standard interface that all AI models must implement for consistent
integration with the ensemble orchestration system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelOutput:
    """
    Standardized output from AI models for ensemble integration.
    
    Attributes:
        prediction: Main prediction value (0-4 scale for trading signals)
        confidence: Model confidence in prediction (0-1)
        probabilities: Full probability distribution across classes
        features: Additional features or intermediate representations
        metadata: Model-specific metadata and diagnostics
    """
    
    def __init__(
        self,
        prediction: int,
        confidence: float,
        probabilities: np.ndarray,
        features: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.prediction = prediction
        self.confidence = confidence
        self.probabilities = probabilities
        self.features = features or {}
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'prediction': self.prediction,
            'confidence': self.confidence,
            'probabilities': self.probabilities.tolist(),
            'features': self.features,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }
    
    def __str__(self) -> str:
        return f"ModelOutput(pred={self.prediction}, conf={self.confidence:.3f})"

class BaseModel(ABC, nn.Module):
    """
    Abstract base class for all AI models in the ensemble system.
    
    Provides standard interface for training, inference, model persistence,
    and integration with the ensemble orchestration system.
    """
    
    def __init__(
        self,
        model_name: str,
        model_version: str = "1.0.0",
        input_dim: int = 29,
        output_dim: int = 5,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        self.model_name = model_name
        self.model_version = model_version
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config or {}
        
        # Model state tracking
        self.is_trained = False
        self.training_history = []
        self.performance_metrics = {}
        
        # Device management
        from ..utils.device_manager import device_manager
        self.device_manager = device_manager
        
        logger.info(f"Initialized {model_name} v{model_version}")
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        pass
    
    @abstractmethod
    def train_model(
        self,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        val_data: Optional[torch.Tensor] = None,
        val_labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model on provided data.
        
        Args:
            train_data: Training input data
            train_labels: Training labels
            val_data: Validation input data (optional)
            val_labels: Validation labels (optional)
            **kwargs: Additional training parameters
            
        Returns:
            Training results and metrics
        """
        pass
    
    def predict(self, x: torch.Tensor) -> ModelOutput:
        """
        Make prediction on input data with standardized output.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            ModelOutput with prediction, confidence, and metadata
        """
        self.eval()
        
        with torch.no_grad():
            # Move input to device
            x = self.device_manager.move_to_device(x)
            
            # Forward pass
            start_time = datetime.utcnow()
            logits = self.forward(x)
            inference_time = (datetime.utcnow() - start_time).total_seconds() * 1000  # ms
            
            # Convert to probabilities
            probabilities = torch.softmax(logits, dim=-1)
            
            # Get prediction and confidence
            max_prob, prediction = torch.max(probabilities, dim=-1)
            
            # For batch processing, return first sample
            if len(prediction.shape) > 0:
                prediction = prediction[0].item()
                confidence = max_prob[0].item()
                prob_array = probabilities[0].cpu().numpy()
            else:
                prediction = prediction.item()
                confidence = max_prob.item()
                prob_array = probabilities.cpu().numpy()
            
            # Create metadata
            metadata = {
                'inference_time_ms': inference_time,
                'model_name': self.model_name,
                'model_version': self.model_version,
                'device': str(self.device_manager.device),
                'input_shape': list(x.shape)
            }
            
            return ModelOutput(
                prediction=prediction,
                confidence=confidence,
                probabilities=prob_array,
                metadata=metadata
            )
    
    def evaluate(
        self,
        test_data: torch.Tensor,
        test_labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data: Test input data
            test_labels: Test labels
            
        Returns:
            Dictionary of performance metrics
        """
        self.eval()
        
        predictions = []
        confidences = []
        
        with torch.no_grad():
            # Process in batches to avoid memory issues
            batch_size = 32
            for i in range(0, len(test_data), batch_size):
                batch_x = test_data[i:i+batch_size]
                
                # Get model predictions
                outputs = []
                for j in range(len(batch_x)):
                    output = self.predict(batch_x[j:j+1])
                    outputs.append(output)
                
                predictions.extend([out.prediction for out in outputs])
                confidences.extend([out.confidence for out in outputs])
        
        # Calculate metrics
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        test_labels_np = test_labels.cpu().numpy()
        
        accuracy = np.mean(predictions == test_labels_np)
        
        # Calculate per-class metrics
        unique_classes = np.unique(test_labels_np)
        class_metrics = {}
        
        for cls in unique_classes:
            mask = test_labels_np == cls
            if np.sum(mask) > 0:
                class_acc = np.mean(predictions[mask] == cls)
                class_conf = np.mean(confidences[mask])
                class_metrics[f'class_{cls}_accuracy'] = class_acc
                class_metrics[f'class_{cls}_confidence'] = class_conf
        
        metrics = {
            'accuracy': accuracy,
            'mean_confidence': np.mean(confidences),
            'prediction_distribution': {str(i): np.sum(predictions == i) for i in unique_classes},
            **class_metrics
        }
        
        self.performance_metrics.update(metrics)
        return metrics
    
    def save_model(self, path: str) -> bool:
        """
        Save model state and configuration.
        
        Args:
            path: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create comprehensive save data
            save_data = {
                'model_name': self.model_name,
                'model_version': self.model_version,
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'config': self.config,
                'is_trained': self.is_trained,
                'training_history': self.training_history,
                'performance_metrics': self.performance_metrics,
                'state_dict': self.state_dict(),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            torch.save(save_data, save_path)
            logger.info(f"Saved model {self.model_name} to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model {self.model_name}: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """
        Load model state and configuration.
        
        Args:
            path: Path to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            save_data = torch.load(path, map_location=self.device_manager.device)
            
            # Verify model compatibility
            if save_data['model_name'] != self.model_name:
                logger.warning(f"Model name mismatch: {save_data['model_name']} != {self.model_name}")
            
            # Load state
            self.model_version = save_data.get('model_version', self.model_version)
            self.input_dim = save_data.get('input_dim', self.input_dim)
            self.output_dim = save_data.get('output_dim', self.output_dim)
            self.config.update(save_data.get('config', {}))
            self.is_trained = save_data.get('is_trained', False)
            self.training_history = save_data.get('training_history', [])
            self.performance_metrics = save_data.get('performance_metrics', {})
            
            # Load model parameters
            self.load_state_dict(save_data['state_dict'])
            
            logger.info(f"Loaded model {self.model_name} from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'is_trained': self.is_trained,
            'parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'device': str(self.device_manager.device),
            'performance_metrics': self.performance_metrics,
            'config': self.config
        }
    
    def reset_model(self):
        """Reset model to untrained state."""
        # Reinitialize parameters
        for layer in self.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        self.is_trained = False
        self.training_history = []
        self.performance_metrics = {}
        
        logger.info(f"Reset model {self.model_name}")
    
    def __str__(self) -> str:
        return f"{self.model_name}(v{self.model_version}, params={sum(p.numel() for p in self.parameters())})"
    
    def __repr__(self) -> str:
        return self.__str__()