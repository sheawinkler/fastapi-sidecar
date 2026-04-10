"""
Simplified Test for Priority 2 Models

Tests the core functionality of Priority 2 models without complex dependencies.
"""

import torch
import numpy as np
import sys
import os
import traceback

# Add project root to path
sys.path.append('/workspace/project')

# Simple device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 PRIORITY 2 MODELS TEST - Using device: {device}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name()}")

print("=" * 60)

def test_progressive_denoising_vae():
    """Test Progressive Denoising VAE"""
    print("\n📋 Testing Progressive Denoising VAE...")
    try:
        # Import the model directly
        from src.ai.models.variational_autoencoders.progressive_denoising_vae import (
            ProgressiveDenoisingVAE, VAEOutput
        )
        
        # Initialize model
        model = ProgressiveDenoisingVAE(
            input_dim=29,
            latent_dim=16,  # Smaller for testing
            hidden_dims=[32, 64, 128, 64],  # Smaller for testing
            device=device
        )
        
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test forward pass
        batch_size = 4
        x = torch.randn(batch_size, 29, device=device)
        
        # Basic forward
        reconstruction = model.forward(x)
        print(f"✅ Basic forward: {reconstruction.shape}")
        
        # Full forward with latent
        vae_output = model.forward(x, return_latent=True)
        print(f"✅ Full VAE output - Latent: {vae_output.latent_z.shape}")
        print(f"✅ Denoising stages: {len(vae_output.denoising_stages)}")
        print(f"✅ Anomaly score range: {vae_output.anomaly_score.min().item():.3f} - {vae_output.anomaly_score.max().item():.3f}")
        
        # Test training step
        targets = torch.randint(0, 5, (batch_size,), device=device)
        losses = model.train_step(x, targets)
        print(f"✅ Training step completed - Total loss: {losses['total']:.4f}")
        
        print("✅ Progressive Denoising VAE: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Progressive Denoising VAE FAILED: {str(e)}")
        traceback.print_exc()
        return False

def test_functional_quantile_ensemble():
    """Test Functional Quantile Ensemble"""
    print("\n📋 Testing Functional Quantile Ensemble...")
    try:
        # Import the model directly
        from src.ai.models.quantile_models.quantile_ensemble import (
            FunctionalQuantileEnsemble, QuantileResult
        )
        
        # Initialize model with smaller parameters
        model = FunctionalQuantileEnsemble(
            input_dim=29,
            quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],  # Fewer quantiles for testing
            hidden_dim=64,  # Smaller for testing
            ensemble_size=2,  # Smaller for testing
            device=device
        )
        
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
        print(f"✅ Quantile levels: {model.quantile_levels}")
        
        # Test forward pass
        batch_size = 4
        x = torch.randn(batch_size, 29, device=device)
        
        quantile_preds = model.forward(x)
        print(f"✅ Quantile predictions: {quantile_preds.shape}")
        
        # Verify monotonicity
        with torch.no_grad():
            batch_preds = quantile_preds[0].detach().cpu().numpy()
            is_monotonic = np.all(batch_preds[:-1] <= batch_preds[1:])
            print(f"✅ Monotonicity check: {is_monotonic}")
        
        # Test training step
        targets = torch.randn(batch_size, 1, device=device)
        losses = model.train_step(x, targets)
        print(f"✅ Training step completed - Total loss: {losses['total']:.4f}")
        
        print("✅ Functional Quantile Ensemble: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Functional Quantile Ensemble FAILED: {str(e)}")
        traceback.print_exc()
        return False

def test_cryptobert_sentiment():
    """Test CryptoBERT Sentiment Fusion"""
    print("\n📋 Testing CryptoBERT Sentiment Fusion...")
    try:
        # Import the model directly
        from src.ai.models.nlp.crypto_bert_sentiment import (
            CryptoBERTSentimentFusion, SentimentResult, CryptoTokenizer
        )
        
        # Test tokenizer first
        tokenizer = CryptoTokenizer(vocab_size=5000)  # Smaller vocab for testing
        
        test_text = "Bitcoin is going to the moon! HODL diamond hands! 🚀"
        tokens = tokenizer.tokenize(test_text, max_length=64)
        sentiment = tokenizer.get_sentiment_score(test_text)
        emotions = tokenizer.detect_emotions(test_text)
        
        print(f"✅ Tokenizer test - Tokens: {len(tokens)}, Sentiment: {sentiment:.3f}")
        print(f"✅ Emotions detected: {list(emotions.keys())}")
        
        # Initialize model with very small parameters for testing
        model = CryptoBERTSentimentFusion(
            input_dim=29,
            vocab_size=5000,   # Much smaller
            d_model=128,       # Much smaller
            n_layers=1,        # Much smaller
            n_heads=2,         # Much smaller
            max_seq_length=64, # Much smaller
            device=device
        )
        
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test forward pass with market features only
        batch_size = 2  # Small batch
        market_features = torch.randn(batch_size, 29, device=device)
        
        sentiment_features = model.forward(market_features)
        print(f"✅ Forward pass (market only): {sentiment_features.shape}")
        
        print("✅ CryptoBERT Sentiment Fusion: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ CryptoBERT Sentiment Fusion FAILED: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Main test execution"""
    print("🚀 Running Simplified Priority 2 Models Test")
    
    tests = [
        ("Progressive Denoising VAE", test_progressive_denoising_vae),
        ("Functional Quantile Ensemble", test_functional_quantile_ensemble),
        ("CryptoBERT Sentiment Fusion", test_cryptobert_sentiment)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print("=" * 60)
    print(f"Overall Result: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL PRIORITY 2 MODELS PASSED!")
        print("✅ Ready for Advanced Models implementation!")
    else:
        print(f"⚠️  {len(results) - passed} test(s) failed")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    print(f"\nFinal Status: {'SUCCESS' if success else 'FAILED'}")