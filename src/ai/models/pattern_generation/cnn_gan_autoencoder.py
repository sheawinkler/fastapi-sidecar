"""
CNN-GAN-Autoencoder Ensemble for Pattern Generation (Model 7)
Innovative pattern generation and anomaly detection framework with GAN-based synthetic data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from ..base_model import BaseModel


class CNNEncoder(nn.Module):
    """CNN encoder for pattern extraction from time series data."""
    
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 1D CNN layers for time series pattern extraction
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv1d(input_dim, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling
        )
        
        # Latent space projection
        self.latent_projection = nn.Linear(256, latent_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
        Returns:
            Latent representation (batch_size, latent_dim)
        """
        # Transpose for Conv1d: (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # CNN feature extraction
        features = self.conv_layers(x)
        features = features.squeeze(-1)  # Remove last dimension after global pooling
        
        # Project to latent space
        latent = self.latent_projection(features)
        
        return latent


class CNNDecoder(nn.Module):
    """CNN decoder for pattern reconstruction."""
    
    def __init__(self, latent_dim: int, output_dim: int, seq_len: int = 48):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        
        # Initial projection
        self.initial_projection = nn.Linear(latent_dim, 256 * (seq_len // 4))
        
        # Transpose convolution layers
        self.deconv_layers = nn.Sequential(
            # First deconv block
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # Second deconv block
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # Output layer
            nn.Conv1d(64, output_dim, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent tensor (batch_size, latent_dim)
        Returns:
            Reconstructed sequence (batch_size, seq_len, output_dim)
        """
        batch_size = z.size(0)
        
        # Initial projection and reshape
        x = self.initial_projection(z)
        x = x.view(batch_size, 256, self.seq_len // 4)
        
        # Transpose convolution
        x = self.deconv_layers(x)
        
        # Transpose back: (batch_size, seq_len, output_dim)
        x = x.transpose(1, 2)
        
        return x


class Generator(nn.Module):
    """GAN Generator for synthetic pattern generation."""
    
    def __init__(self, noise_dim: int, output_dim: int, seq_len: int = 48):
        super().__init__()
        self.noise_dim = noise_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        
        self.generator = nn.Sequential(
            # Initial dense layer
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Expand to sequence
            nn.Linear(256, 512 * (seq_len // 8)),
            nn.ReLU(),
        )
        
        # Reshape and convolution layers
        self.conv_layers = nn.Sequential(
            # Upsample with transpose convolutions
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # Output layer
            nn.Conv1d(64, output_dim, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noise: Random noise tensor (batch_size, noise_dim)
        Returns:
            Generated sequence (batch_size, seq_len, output_dim)
        """
        batch_size = noise.size(0)
        
        # Generate features
        x = self.generator(noise)
        x = x.view(batch_size, 512, self.seq_len // 8)
        
        # Apply convolutions
        x = self.conv_layers(x)
        
        # Transpose to correct format
        x = x.transpose(1, 2)
        
        return x


class Discriminator(nn.Module):
    """GAN Discriminator for distinguishing real from synthetic patterns."""
    
    def __init__(self, input_dim: int, seq_len: int = 48):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        
        self.discriminator = nn.Sequential(
            # Conv layers
            nn.Conv1d(input_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # Global pooling
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence (batch_size, seq_len, input_dim)
        Returns:
            Probability of being real (batch_size, 1)
        """
        # Transpose for Conv1d
        x = x.transpose(1, 2)
        
        # Feature extraction
        features = self.discriminator(x)
        features = features.squeeze(-1)
        
        # Classification
        output = self.classifier(features)
        
        return output


class CNNGANAutoencoder(BaseModel):
    """
    CNN-GAN-Autoencoder Ensemble for Pattern Generation (Model 7)
    
    Innovative pattern generation and anomaly detection framework combining:
    - CNN Autoencoder for pattern compression and reconstruction
    - GAN for synthetic data generation of rare market events
    - Ensemble scoring for anomaly detection and pattern classification
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Model configuration
        self.latent_dim = config.get('latent_dim', 64)
        self.noise_dim = config.get('noise_dim', 100)
        self.seq_len = config.get('seq_len', 50)
        self.num_classes = config.get('num_classes', 5)
        
        # CNN Autoencoder components
        self.encoder = CNNEncoder(self.input_dim, self.latent_dim)
        self.decoder = CNNDecoder(self.latent_dim, self.input_dim, self.seq_len)
        
        # GAN components
        self.generator = Generator(self.noise_dim, self.input_dim, self.seq_len)
        self.discriminator = Discriminator(self.input_dim, self.seq_len)
        
        # Classifier for pattern recognition
        self.pattern_classifier = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, self.num_classes)
        )
        
        # Anomaly scorer
        self.anomaly_scorer = nn.Sequential(
            nn.Linear(self.latent_dim + 1, 64),  # +1 for reconstruction error
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with appropriate strategies for different components."""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input sequence to latent representation."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to sequence."""
        return self.decoder(z)
    
    def generate_synthetic(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """Generate synthetic sequences using the GAN generator."""
        if device is None:
            device = next(self.parameters()).device
            
        noise = torch.randn(batch_size, self.noise_dim, device=device)
        synthetic_data = self.generator(noise)
        return synthetic_data
    
    def compute_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction error for anomaly detection."""
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        
        # Mean squared error per sample
        mse = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
        return mse
    
    def detect_anomaly(self, x: torch.Tensor) -> Dict:
        """Comprehensive anomaly detection using multiple signals."""
        latent = self.encode(x)
        recon_error = self.compute_reconstruction_error(x)
        
        # Combine latent features with reconstruction error
        anomaly_features = torch.cat([
            latent, recon_error.unsqueeze(-1)
        ], dim=-1)
        
        anomaly_score = self.anomaly_scorer(anomaly_features)
        
        return {
            'anomaly_score': anomaly_score.cpu().numpy(),
            'reconstruction_error': recon_error.cpu().numpy(),
            'is_anomaly': (anomaly_score > 0.7).cpu().numpy()
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for pattern classification.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim) or (batch_size, input_dim)
        Returns:
            Classification logits (batch_size, num_classes)
        """
        # Handle single timestep input
        if x.dim() == 2:
            # Create a simple sequence by repeating the input
            x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        elif x.size(1) != self.seq_len:
            # Interpolate to required sequence length
            x = F.interpolate(
                x.transpose(1, 2), 
                size=self.seq_len, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
        
        # Encode to latent space
        latent = self.encode(x)
        
        # Classify pattern
        logits = self.pattern_classifier(latent)
        
        return logits
    
    def predict(self, x: torch.Tensor) -> Dict:
        """Make comprehensive prediction including anomaly detection."""
        self.eval()
        with torch.no_grad():
            # Pattern classification
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            confidence = torch.max(probs, dim=-1)[0].item()
            prediction = torch.argmax(logits, dim=-1).item()
            
            # Anomaly detection
            if x.dim() == 2:
                x_seq = x.unsqueeze(1).repeat(1, self.seq_len, 1)
            else:
                x_seq = x
                
            anomaly_info = self.detect_anomaly(x_seq)
            
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probs.cpu().numpy(),
            'anomaly_score': anomaly_info['anomaly_score'][0],
            'is_anomaly': anomaly_info['is_anomaly'][0],
            'reconstruction_error': anomaly_info['reconstruction_error'][0]
        }
    
    def train_autoencoder(self, x: torch.Tensor) -> Dict[str, float]:
        """Train the autoencoder component."""
        self.train()
        
        # Prepare sequence data
        if x.dim() == 2:
            x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        elif x.size(1) != self.seq_len:
            x = F.interpolate(
                x.transpose(1, 2), 
                size=self.seq_len, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
        
        # Autoencoder forward pass
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, x)
        
        # Pattern classification loss (if labels available)
        class_logits = self.pattern_classifier(latent)
        
        return {
            'reconstruction_loss': recon_loss.item(),
            'class_logits': class_logits
        }


# Model factory function
def create_cnn_gan_autoencoder(input_dim: int = 29) -> CNNGANAutoencoder:
    """Create a CNN-GAN-Autoencoder model with optimized configuration."""
    config = {
        'input_dim': input_dim,
        'latent_dim': 64,
        'noise_dim': 100,
        'seq_len': 48,
        'num_classes': 5,
    }
    return CNNGANAutoencoder(config)
