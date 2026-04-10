"""
Test Suite for Phase 3 AI Models

Validates the implementation of Priority 1 AI models with comprehensive testing
including forward pass, training, and inference capabilities.
"""

import torch
import numpy as np
import logging
import sys
import os
from pathlib import Path

# Add project path
project_path = Path(__file__).parent
sys.path.append(str(project_path))

from src.ai.utils.device_manager import device_manager
from src.ai.models.reinforcement_learning.executive_auxiliary_agent import ExecutiveAuxiliaryAgent
from src.ai.models.transformers.cross_modal_temporal_fusion import CrossModalTemporalFusion

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIModelsTestSuite:
    """Comprehensive test suite for AI models."""
    
    def __init__(self):
        self.device = device_manager.device
        self.input_dim = 29  # From Phase 2 data infrastructure
        self.output_dim = 5  # Trading signal classes (0-4)
        self.batch_size = 16
        self.seq_len = 50
        
        logger.info(f"Test suite initialized on device: {self.device}")
    
    def generate_test_data(self, num_samples: int = 100) -> tuple:
        """Generate synthetic test data matching real data format."""
        # Generate realistic financial time series data
        data = torch.randn(num_samples, self.input_dim, device=self.device)
        
        # Add some structure to make it more realistic
        # Price features (first 15 dimensions) with some autocorrelation
        for i in range(1, num_samples):
            data[i, :15] = 0.9 * data[i-1, :15] + 0.1 * torch.randn(15, device=self.device)
        
        # Sentiment features (next 8 dimensions) with bounded values
        data[:, 15:23] = torch.tanh(data[:, 15:23])
        
        # On-chain features (last 6 dimensions) with positive values
        data[:, 23:29] = torch.abs(data[:, 23:29])
        
        # Generate labels (trading signals 0-4)
        labels = torch.randint(0, self.output_dim, (num_samples,), device=self.device)
        
        return data, labels
    
    def test_device_manager(self):
        """Test device management functionality."""
        logger.info("Testing Device Manager...")
        
        try:
            # Test device selection
            assert device_manager.device is not None
            logger.info(f"✅ Device selected: {device_manager.device}")
            
            # Test memory management
            memory_info = device_manager.get_memory_usage()
            assert 'total_mb' in memory_info
            logger.info(f"✅ Memory info: {memory_info}")
            
            # Test tensor creation
            test_tensor = device_manager.randn(10, 10)
            assert test_tensor.device == device_manager.device
            logger.info("✅ Tensor creation on correct device")
            
            # Test memory optimization
            device_manager.optimize_memory()
            logger.info("✅ Memory optimization completed")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Device Manager test failed: {e}")
            return False
    
    def test_executive_auxiliary_agent(self):
        """Test Executive-Auxiliary Agent model."""
        logger.info("Testing Executive-Auxiliary Agent...")
        
        try:
            # Initialize model
            model = ExecutiveAuxiliaryAgent(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                executive_config={'hidden_dims': [128, 64, 32]},
                auxiliary_config={'output_dim': 32}
            )
            model.to(self.device)
            
            logger.info(f"✅ Model initialized with {model.count_parameters()} parameters")
            
            # Test forward pass
            test_data, test_labels = self.generate_test_data(self.batch_size)
            
            model.eval()
            with torch.no_grad():
                output = model(test_data)
                assert output.shape == (self.batch_size, self.output_dim)
                logger.info(f"✅ Forward pass successful: {output.shape}")
            
            # Test prediction interface
            single_sample = test_data[:1]
            prediction = model.predict(single_sample)
            
            assert hasattr(prediction, 'prediction')
            assert hasattr(prediction, 'confidence')
            assert hasattr(prediction, 'probabilities')
            assert 0 <= prediction.prediction < self.output_dim
            assert 0 <= prediction.confidence <= 1
            
            logger.info(f"✅ Prediction interface: pred={prediction.prediction}, conf={prediction.confidence:.3f}")
            
            # Test forward with value
            policy, value = model.forward_with_value(test_data)
            assert policy.shape == (self.batch_size, self.output_dim)
            assert value.shape == (self.batch_size, 1)
            logger.info(f"✅ Policy-Value forward: policy={policy.shape}, value={value.shape}")
            
            # Test auxiliary analysis
            analysis = model.get_auxiliary_analysis()
            assert 'parameter_distribution' in analysis
            assert 'architecture_info' in analysis
            logger.info("✅ Auxiliary analysis completed")
            
            # Test model info
            model_info = model.get_model_info()
            assert model_info['model_name'] == "ExecutiveAuxiliaryAgent"
            assert model_info['parameters'] > 0
            logger.info(f"✅ Model info: {model_info['model_name']} with {model_info['parameters']} params")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Executive-Auxiliary Agent test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_cross_modal_temporal_fusion(self):
        """Test Cross-Modal Temporal Fusion model."""
        logger.info("Testing Cross-Modal Temporal Fusion...")
        
        try:
            # Initialize model
            modality_dims = {
                'price': 15,
                'sentiment': 8,
                'onchain': 6
            }
            
            model = CrossModalTemporalFusion(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                embed_dim=128,
                num_heads=4,
                num_layers=3,
                seq_length=self.seq_len,
                modality_dims=modality_dims
            )
            model.to(self.device)
            
            logger.info(f"✅ Model initialized with {model.count_parameters()} parameters")
            
            # Test single timestep forward pass
            test_data, test_labels = self.generate_test_data(self.batch_size)
            
            model.eval()
            with torch.no_grad():
                output = model(test_data)
                assert output.shape == (self.batch_size, self.output_dim)
                logger.info(f"✅ Single timestep forward: {output.shape}")
            
            # Test sequence forward pass
            sequence_data = torch.randn(self.batch_size, self.seq_len, self.input_dim, device=self.device)
            
            with torch.no_grad():
                seq_output = model(sequence_data)
                assert seq_output.shape == (self.batch_size, self.output_dim)
                logger.info(f"✅ Sequence forward: {seq_output.shape}")
            
            # Test prediction interface
            single_sample = test_data[:1]
            prediction = model.predict(single_sample)
            
            assert hasattr(prediction, 'prediction')
            assert hasattr(prediction, 'confidence')
            assert hasattr(prediction, 'probabilities')
            assert 0 <= prediction.prediction < self.output_dim
            assert 0 <= prediction.confidence <= 1
            
            logger.info(f"✅ Prediction interface: pred={prediction.prediction}, conf={prediction.confidence:.3f}")
            
            # Test attention analysis
            analysis = model.get_attention_analysis()
            assert 'modality_performance' in analysis
            assert 'parameter_distribution' in analysis
            logger.info("✅ Attention analysis completed")
            
            # Test model info
            model_info = model.get_model_info()
            assert model_info['model_name'] == "CrossModalTemporalFusion"
            assert model_info['parameters'] > 0
            logger.info(f"✅ Model info: {model_info['model_name']} with {model_info['parameters']} params")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Cross-Modal Temporal Fusion test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_model_training(self):
        """Test model training functionality (quick training run)."""
        logger.info("Testing Model Training...")
        
        try:
            # Generate larger training dataset
            train_data, train_labels = self.generate_test_data(200)
            val_data, val_labels = self.generate_test_data(50)
            
            # Test Executive-Auxiliary Agent training
            logger.info("Testing Executive-Auxiliary Agent training...")
            ea_model = ExecutiveAuxiliaryAgent(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                executive_config={'hidden_dims': [64, 32]},
                auxiliary_config={'output_dim': 16, 'hidden_dims': [32, 16]}
            )
            
            ea_results = ea_model.train_model(
                train_data, train_labels,
                val_data, val_labels,
                num_epochs=5,  # Quick test
                batch_size=16,
                learning_rate=1e-3
            )
            
            assert ea_results['training_completed']
            assert ea_model.is_trained
            logger.info(f"✅ EA training: {ea_results['final_train_accuracy']:.3f} accuracy")
            
            # Test Cross-Modal Temporal Fusion training
            logger.info("Testing Cross-Modal Temporal Fusion training...")
            cmtf_model = CrossModalTemporalFusion(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                embed_dim=64,
                num_heads=2,
                num_layers=2
            )
            
            cmtf_results = cmtf_model.train_model(
                train_data, train_labels,
                val_data, val_labels,
                num_epochs=5,  # Quick test
                batch_size=16,
                learning_rate=1e-4
            )
            
            assert cmtf_results['training_completed']
            assert cmtf_model.is_trained
            logger.info(f"✅ CMTF training: {cmtf_results['final_train_accuracy']:.3f} accuracy")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Model training test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_model_persistence(self):
        """Test model saving and loading."""
        logger.info("Testing Model Persistence...")
        
        try:
            # Create and train a small model
            model = ExecutiveAuxiliaryAgent(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                executive_config={'hidden_dims': [32]},
                auxiliary_config={'output_dim': 8}
            )
            
            test_data, test_labels = self.generate_test_data(20)
            
            # Get initial prediction
            initial_pred = model.predict(test_data[:1])
            
            # Save model
            save_path = "/tmp/test_model.pt"
            success = model.save_model(save_path)
            assert success
            assert os.path.exists(save_path)
            logger.info("✅ Model saved successfully")
            
            # Create new model and load
            new_model = ExecutiveAuxiliaryAgent(
                input_dim=self.input_dim,
                output_dim=self.output_dim
            )
            
            success = new_model.load_model(save_path)
            assert success
            logger.info("✅ Model loaded successfully")
            
            # Verify loaded model produces same predictions
            loaded_pred = new_model.predict(test_data[:1])
            
            # Predictions should be identical (or very close due to floating point)
            pred_diff = abs(initial_pred.confidence - loaded_pred.confidence)
            assert pred_diff < 1e-5, f"Prediction difference too large: {pred_diff}"
            
            logger.info("✅ Model persistence verified")
            
            # Clean up
            os.remove(save_path)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Model persistence test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_all_tests(self) -> bool:
        """Run all tests and return overall success."""
        logger.info("🚀 Starting AI Models Phase 3 Test Suite")
        logger.info("=" * 60)
        
        tests = [
            ("Device Manager", self.test_device_manager),
            ("Executive-Auxiliary Agent", self.test_executive_auxiliary_agent),
            ("Cross-Modal Temporal Fusion", self.test_cross_modal_temporal_fusion),
            ("Model Training", self.test_model_training),
            ("Model Persistence", self.test_model_persistence)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            logger.info(f"\n📋 Running {test_name} tests...")
            try:
                success = test_func()
                results[test_name] = success
                if success:
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name}: FAILED with exception: {e}")
                results[test_name] = False
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 TEST SUMMARY")
        logger.info("=" * 60)
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            logger.info(f"{test_name:.<40} {status}")
        
        logger.info(f"\nOverall Result: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 ALL TESTS PASSED! Phase 3 Priority 1 Models Ready!")
            return True
        else:
            logger.error(f"⚠️  {total - passed} tests failed. Review implementation.")
            return False

if __name__ == "__main__":
    test_suite = AIModelsTestSuite()
    success = test_suite.run_all_tests()
    
    if success:
        print("\n🎯 Phase 3 Priority 1 Models Implementation: SUCCESS!")
        print("Ready to proceed with Priority 2 models implementation.")
    else:
        print("\n❌ Phase 3 Priority 1 Models Implementation: ISSUES DETECTED")
        print("Please review and fix failing tests before proceeding.")
    
    sys.exit(0 if success else 1)