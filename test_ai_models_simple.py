"""
Simplified Test for Phase 3 AI Models

Tests AI models directly without complex project dependencies.
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add the src path directly
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Direct imports
from ai.utils.device_manager import DeviceManager
from ai.models.base_model import BaseModel
from ai.models.reinforcement_learning.executive_auxiliary_agent import ExecutiveAuxiliaryAgent
from ai.models.transformers.cross_modal_temporal_fusion import CrossModalTemporalFusion

print("🚀 Starting Simplified AI Models Test Suite")
print("=" * 60)

def test_device_manager():
    """Test device management."""
    print("Testing Device Manager...")
    
    try:
        device_mgr = DeviceManager()
        print(f"✅ Device: {device_mgr.device}")
        print(f"✅ Device info: {device_mgr.get_device_name()}")
        
        # Test tensor creation
        tensor = device_mgr.randn(5, 5)
        assert tensor.device == device_mgr.device
        print(f"✅ Tensor creation successful on {tensor.device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Device Manager failed: {e}")
        return False

def test_executive_auxiliary_agent():
    """Test Executive-Auxiliary Agent."""
    print("Testing Executive-Auxiliary Agent...")
    
    try:
        # Create model
        model = ExecutiveAuxiliaryAgent(
            input_dim=29,
            output_dim=5,
            executive_config={'hidden_dims': [64, 32]},
            auxiliary_config={'output_dim': 16}
        )
        
        print(f"✅ Model created with {model.count_parameters()} parameters")
        
        # Test forward pass
        test_input = torch.randn(8, 29)
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
        test_input = test_input.to('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        
        with torch.no_grad():
            output = model(test_input)
            assert output.shape == (8, 5)
            print(f"✅ Forward pass: {output.shape}")
        
        # Test prediction
        prediction = model.predict(test_input[:1])
        assert hasattr(prediction, 'prediction')
        assert hasattr(prediction, 'confidence')
        print(f"✅ Prediction: pred={prediction.prediction}, conf={prediction.confidence:.3f}")
        
        # Test forward with value
        policy, value = model.forward_with_value(test_input)
        assert policy.shape == (8, 5)
        assert value.shape == (8, 1)
        print(f"✅ Policy-Value: {policy.shape}, {value.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Executive-Auxiliary Agent failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cross_modal_temporal_fusion():
    """Test Cross-Modal Temporal Fusion."""
    print("Testing Cross-Modal Temporal Fusion...")
    
    try:
        # Create model
        model = CrossModalTemporalFusion(
            input_dim=29,
            output_dim=5,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            modality_dims={'price': 15, 'sentiment': 8, 'onchain': 6}
        )
        
        print(f"✅ Model created with {model.count_parameters()} parameters")
        
        # Test single timestep
        test_input = torch.randn(8, 29)
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
        test_input = test_input.to('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        
        with torch.no_grad():
            output = model(test_input)
            assert output.shape == (8, 5)
            print(f"✅ Single timestep forward: {output.shape}")
        
        # Test sequence input
        seq_input = torch.randn(8, 20, 29).to('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            seq_output = model(seq_input)
            assert seq_output.shape == (8, 5)
            print(f"✅ Sequence forward: {seq_output.shape}")
        
        # Test prediction
        prediction = model.predict(test_input[:1])
        assert hasattr(prediction, 'prediction')
        assert hasattr(prediction, 'confidence')
        print(f"✅ Prediction: pred={prediction.prediction}, conf={prediction.confidence:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cross-Modal Temporal Fusion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training():
    """Test model training."""
    print("Testing Model Training...")
    
    try:
        # Generate synthetic data
        train_data = torch.randn(100, 29)
        train_labels = torch.randint(0, 5, (100,))
        val_data = torch.randn(20, 29)
        val_labels = torch.randint(0, 5, (20,))
        
        # Test Executive-Auxiliary Agent training
        ea_model = ExecutiveAuxiliaryAgent(
            input_dim=29, output_dim=5,
            executive_config={'hidden_dims': [32, 16]},
            auxiliary_config={'output_dim': 8}
        )
        
        results = ea_model.train_model(
            train_data, train_labels,
            val_data, val_labels,
            num_epochs=3,
            batch_size=16
        )
        
        assert results['training_completed']
        print(f"✅ EA training completed: {results['final_train_accuracy']:.3f}")
        
        # Test Cross-Modal Temporal Fusion training
        cmtf_model = CrossModalTemporalFusion(
            input_dim=29, output_dim=5,
            embed_dim=32, num_heads=2, num_layers=2
        )
        
        results = cmtf_model.train_model(
            train_data, train_labels,
            val_data, val_labels,
            num_epochs=3,
            batch_size=16
        )
        
        assert results['training_completed']
        print(f"✅ CMTF training completed: {results['final_train_accuracy']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Run tests
tests = [
    ("Device Manager", test_device_manager),
    ("Executive-Auxiliary Agent", test_executive_auxiliary_agent),
    ("Cross-Modal Temporal Fusion", test_cross_modal_temporal_fusion),
    ("Model Training", test_training)
]

results = []
for test_name, test_func in tests:
    print(f"\n📋 Running {test_name}...")
    success = test_func()
    results.append((test_name, success))
    print(f"{'✅' if success else '❌'} {test_name}: {'PASSED' if success else 'FAILED'}")

# Summary
print("\n" + "=" * 60)
print("📊 TEST SUMMARY")
print("=" * 60)

passed = sum(1 for _, success in results if success)
total = len(results)

for test_name, success in results:
    status = "✅ PASSED" if success else "❌ FAILED"
    print(f"{test_name:.<40} {status}")

print(f"\nOverall Result: {passed}/{total} tests passed")

if passed == total:
    print("🎉 ALL TESTS PASSED! Priority 1 Models Ready!")
else:
    print(f"⚠️ {total - passed} tests failed.")

print(f"\nFinal Status: {'SUCCESS' if passed == total else 'PARTIAL SUCCESS'}")