"""
GPU/CPU Device Management for AI Models

Handles automatic device selection, memory management, and performance optimization
for PyTorch models with CUDA support.
"""

import torch
import logging
from typing import Optional, Union, List
import psutil
import gc

logger = logging.getLogger(__name__)

class DeviceManager:
    """
    Enterprise-grade device management for AI model training and inference.
    
    Features:
    - Automatic GPU/CPU selection based on availability
    - Memory monitoring and optimization
    - Multi-GPU support for distributed training
    - Device-specific performance optimizations
    """
    
    def __init__(self):
        self.device = self._select_optimal_device()
        self.device_info = self._get_device_info()
        self._optimize_device_settings()
        
        logger.info(f"DeviceManager initialized with device: {self.device}")
        logger.info(f"Device info: {self.device_info}")
    
    def _select_optimal_device(self) -> torch.device:
        """Select the optimal device based on availability and performance."""
        if torch.cuda.is_available():
            # Select GPU with most memory available
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                best_gpu = 0
                max_memory = 0
                
                for i in range(gpu_count):
                    memory = torch.cuda.get_device_properties(i).total_memory
                    if memory > max_memory:
                        max_memory = memory
                        best_gpu = i
                
                device = torch.device(f'cuda:{best_gpu}')
            else:
                device = torch.device('cuda:0')
                
            # Verify CUDA is working
            try:
                torch.cuda.empty_cache()
                test_tensor = torch.randn(10, 10).to(device)
                _ = test_tensor @ test_tensor.T
                del test_tensor
                torch.cuda.empty_cache()
                
                logger.info(f"CUDA device verified: {device}")
                return device
                
            except Exception as e:
                logger.warning(f"CUDA verification failed: {e}, falling back to CPU")
                return torch.device('cpu')
        else:
            logger.info("CUDA not available, using CPU")
            return torch.device('cpu')
    
    def _get_device_info(self) -> dict:
        """Get detailed information about the selected device."""
        info = {
            'type': self.device.type,
            'index': self.device.index if self.device.type == 'cuda' else None
        }
        
        if self.device.type == 'cuda':
            props = torch.cuda.get_device_properties(self.device)
            info.update({
                'name': props.name,
                'total_memory': props.total_memory,
                'memory_gb': props.total_memory / (1024**3),
                'compute_capability': f"{props.major}.{props.minor}",
                'multiprocessor_count': props.multi_processor_count
            })
        else:
            info.update({
                'name': 'CPU',
                'cores': psutil.cpu_count(logical=False),
                'logical_cores': psutil.cpu_count(logical=True),
                'memory_gb': psutil.virtual_memory().total / (1024**3)
            })
        
        return info
    
    def _optimize_device_settings(self):
        """Apply device-specific optimizations."""
        if self.device.type == 'cuda':
            # Enable cuDNN benchmarking for consistent input sizes
            torch.backends.cudnn.benchmark = True
            
            # Enable TensorFloat-32 (TF32) for better performance on Ampere GPUs
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            logger.info("Applied CUDA optimizations")
        else:
            # CPU optimizations
            torch.set_num_threads(psutil.cpu_count(logical=False))
            logger.info(f"Set CPU threads to {psutil.cpu_count(logical=False)}")
    
    def get_memory_usage(self) -> dict:
        """Get current memory usage statistics."""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device)
            cached = torch.cuda.memory_reserved(self.device)
            total = torch.cuda.get_device_properties(self.device).total_memory
            
            return {
                'allocated_mb': allocated / (1024**2),
                'cached_mb': cached / (1024**2),
                'total_mb': total / (1024**2),
                'utilization_pct': (allocated / total) * 100
            }
        else:
            memory = psutil.virtual_memory()
            return {
                'used_mb': memory.used / (1024**2),
                'total_mb': memory.total / (1024**2),
                'utilization_pct': memory.percent
            }
    
    def optimize_memory(self):
        """Perform memory optimization and cleanup."""
        # Clear Python garbage collector
        gc.collect()
        
        if self.device.type == 'cuda':
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            # Synchronize to ensure operations complete
            torch.cuda.synchronize()
            
            logger.info("Performed GPU memory optimization")
        
        memory_usage = self.get_memory_usage()
        logger.info(f"Memory usage after optimization: {memory_usage}")
    
    def move_to_device(self, tensor_or_model: Union[torch.Tensor, torch.nn.Module]) -> Union[torch.Tensor, torch.nn.Module]:
        """Move tensor or model to the optimal device."""
        try:
            return tensor_or_model.to(self.device)
        except Exception as e:
            logger.error(f"Failed to move to device {self.device}: {e}")
            raise
    
    def create_tensor(self, *args, **kwargs) -> torch.Tensor:
        """Create tensor directly on the optimal device."""
        kwargs['device'] = self.device
        return torch.tensor(*args, **kwargs)
    
    def zeros(self, *size, **kwargs) -> torch.Tensor:
        """Create zero tensor on optimal device."""
        kwargs['device'] = self.device
        return torch.zeros(*size, **kwargs)
    
    def ones(self, *size, **kwargs) -> torch.Tensor:
        """Create ones tensor on optimal device."""
        kwargs['device'] = self.device
        return torch.ones(*size, **kwargs)
    
    def randn(self, *size, **kwargs) -> torch.Tensor:
        """Create random normal tensor on optimal device."""
        kwargs['device'] = self.device
        return torch.randn(*size, **kwargs)
    
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available and working."""
        return self.device.type == 'cuda'
    
    def get_device_name(self) -> str:
        """Get human-readable device name."""
        return self.device_info.get('name', str(self.device))
    
    def __str__(self) -> str:
        return f"DeviceManager(device={self.device}, name={self.get_device_name()})"
    
    def __repr__(self) -> str:
        return self.__str__()

# Global device manager instance
device_manager = DeviceManager()