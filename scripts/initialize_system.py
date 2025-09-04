#!/usr/bin/env python3
"""
System Initialization Script for Crypto AI Trading Platform

This script initializes the entire trading system, including:
- Configuration validation
- Security setup
- Database initialization 
- API connectivity testing
- Logging system setup
- Health checks

Run this script before starting any trading operations.
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import get_config_manager, get_config
from utils.logger import get_logger_system, get_structured_logger
from security.credential_manager import get_credential_manager, initialize_credentials_from_env


class SystemInitializer:
    """Comprehensive system initialization and validation."""
    
    def __init__(self):
        """Initialize the system initializer."""
        self.logger = None
        self.config = None
        self.credential_manager = None
        
        # Track initialization steps
        self.init_steps = [
            ("Configuration", self._init_config),
            ("Logging", self._init_logging),
            ("Security", self._init_security),
            ("Directories", self._init_directories),
            ("Environment", self._validate_environment),
            ("APIs", self._test_api_connectivity),
            ("Health", self._system_health_check)
        ]
    
    async def initialize(self) -> bool:
        """
        Run complete system initialization.
        
        Returns:
            True if initialization successful
        """
        print("🚀 Starting Crypto AI Trading System Initialization...")
        print("=" * 60)
        
        start_time = time.time()
        success_count = 0
        
        for step_name, step_func in self.init_steps:
            print(f"\n📋 Initializing {step_name}...")
            
            try:
                success = await step_func()
                
                if success:
                    print(f"✅ {step_name} initialization completed successfully")
                    success_count += 1
                else:
                    print(f"❌ {step_name} initialization failed")
                    
            except Exception as e:
                print(f"💥 {step_name} initialization crashed: {e}")
                if self.logger:
                    self.logger.error(f"{step_name} initialization failed", error=str(e))
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print(f"🏁 Initialization Complete!")
        print(f"✅ Successful steps: {success_count}/{len(self.init_steps)}")
        print(f"⏱️  Total time: {total_time:.2f} seconds")
        
        if success_count == len(self.init_steps):
            print("🎉 All systems are GO! Ready for trading operations.")
            return True
        else:
            print("⚠️  Some systems failed to initialize. Check logs for details.")
            return False
    
    async def _init_config(self) -> bool:
        """Initialize configuration system."""
        try:
            # Copy example config if config.yaml doesn't exist
            config_path = Path("config/config.yaml")
            example_config_path = Path("config/config.example.yaml")
            
            if not config_path.exists() and example_config_path.exists():
                print("📝 Creating config.yaml from example...")
                config_path.parent.mkdir(exist_ok=True)
                
                # Read example and create config
                with open(example_config_path, 'r') as f:
                    config_content = f.read()
                
                with open(config_path, 'w') as f:
                    f.write(config_content)
                
                print("⚠️  Please update config/config.yaml with your settings!")
            
            # Load configuration
            config_manager = get_config_manager()
            self.config = config_manager.load_config()
            
            # Validate configuration
            env_status = config_manager.validate_environment()
            missing_required = [var for var, status in env_status.items() 
                              if not status and var in ['MASTER_PASSWORD']]
            
            if missing_required:
                print(f"⚠️  Missing required environment variables: {missing_required}")
                print("   Set them in your environment or .env file")
            
            # Print configuration summary
            summary = config_manager.export_config_summary()
            print(f"🔧 Environment: {summary['system']['environment']}")
            print(f"💰 Trading Mode: {summary['trading']['mode']}")
            print(f"🤖 AI Models: {len(summary['ai_models']['enabled_models'])} enabled")
            
            return True
            
        except Exception as e:
            print(f"Configuration initialization failed: {e}")
            return False
    
    async def _init_logging(self) -> bool:
        """Initialize logging system."""
        try:
            logger_system = get_logger_system()
            self.logger = get_structured_logger("system_init")
            
            # Test different log types
            self.logger.info("Logging system initialized successfully")
            
            # Create logs directory
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            
            print(f"📁 Logs directory: {logs_dir.absolute()}")
            return True
            
        except Exception as e:
            print(f"Logging initialization failed: {e}")
            return False
    
    async def _init_security(self) -> bool:
        """Initialize security system."""
        try:
            # Set up master password if not set
            if not os.getenv('MASTER_PASSWORD'):
                print("⚠️  MASTER_PASSWORD not set, using default for testing")
                os.environ['MASTER_PASSWORD'] = 'default_master_password_for_testing_only'
            
            # Initialize credential manager
            self.credential_manager = get_credential_manager()
            
            # Load credentials from environment
            initialize_credentials_from_env()
            
            # Test credential operations
            test_cred = "test_credential_value"
            self.credential_manager.store_credential("test_key", test_cred)
            retrieved = self.credential_manager.get_credential("test_key")
            
            if retrieved != test_cred:
                raise Exception("Credential encryption/decryption test failed")
            
            print("🔐 Security system operational")
            return True
            
        except Exception as e:
            print(f"Security initialization failed: {e}")
            return False
    
    async def _init_directories(self) -> bool:
        """Create required directories."""
        try:
            required_dirs = [
                "logs",
                "data/historical",
                "data/cache",
                "models/saved",
                "reports",
                "backups"
            ]
            
            for dir_path in required_dirs:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
            
            print(f"📁 Created {len(required_dirs)} required directories")
            return True
            
        except Exception as e:
            print(f"Directory initialization failed: {e}")
            return False
    
    async def _validate_environment(self) -> bool:
        """Validate system environment."""
        try:
            # Check Python version
            python_version = sys.version_info
            if python_version < (3, 9):
                print(f"⚠️  Python {python_version.major}.{python_version.minor} detected. Python 3.9+ recommended.")
            else:
                print(f"🐍 Python {python_version.major}.{python_version.minor}.{python_version.micro}")
            
            # Check disk space
            import shutil
            disk_usage = shutil.disk_usage(".")
            free_gb = disk_usage.free / (1024**3)
            
            if free_gb < 1.0:
                print(f"⚠️  Low disk space: {free_gb:.1f}GB free")
            else:
                print(f"💾 Disk space: {free_gb:.1f}GB free")
            
            # Check memory
            try:
                import psutil
                memory = psutil.virtual_memory()
                memory_gb = memory.total / (1024**3)
                print(f"🧠 Memory: {memory_gb:.1f}GB total")
                
                if memory_gb < 4.0:
                    print("⚠️  Less than 4GB RAM detected. Performance may be limited.")
            except ImportError:
                print("ℹ️  psutil not available for memory check")
            
            # Check GPU availability
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    gpu_name = torch.cuda.get_device_name(0)
                    print(f"🎮 GPU: {gpu_count}x {gpu_name}")
                else:
                    print("⚠️  No CUDA GPU detected. CPU-only mode.")
            except ImportError:
                print("ℹ️  PyTorch not available for GPU check")
            
            return True
            
        except Exception as e:
            print(f"Environment validation failed: {e}")
            return False
    
    async def _test_api_connectivity(self) -> bool:
        """Test API connectivity."""
        try:
            print("🌐 Testing API connectivity...")
            
            # Test basic HTTP connectivity
            import aiohttp
            
            test_apis = [
                ("CoinGecko", "https://api.coingecko.com/api/v3/ping"),
                ("GitHub", "https://api.github.com"),
            ]
            
            successful_tests = 0
            
            async with aiohttp.ClientSession() as session:
                for api_name, url in test_apis:
                    try:
                        async with session.get(url, timeout=10) as response:
                            if response.status == 200:
                                print(f"  ✅ {api_name}: Connected")
                                successful_tests += 1
                            else:
                                print(f"  ❌ {api_name}: HTTP {response.status}")
                    except Exception as e:
                        print(f"  ❌ {api_name}: Connection failed ({str(e)[:50]})")
            
            # Test if we have API credentials configured
            if self.credential_manager:
                helius_key = self.credential_manager.get_credential('helius_api_key')
                if helius_key:
                    print("  ✅ Helius API key configured")
                else:
                    print("  ⚠️  Helius API key not configured")
            
            print(f"🌐 API connectivity: {successful_tests}/{len(test_apis)} successful")
            return successful_tests > 0
            
        except Exception as e:
            print(f"API connectivity test failed: {e}")
            return False
    
    async def _system_health_check(self) -> bool:
        """Perform final system health check."""
        try:
            health_checks = []
            
            # Configuration health
            health_checks.append(("Configuration", self.config is not None))
            
            # Logging health
            health_checks.append(("Logging", self.logger is not None))
            
            # Security health
            health_checks.append(("Security", self.credential_manager is not None))
            
            # Directory health
            required_dirs = ["logs", "data", "models"]
            dirs_exist = all(Path(d).exists() for d in required_dirs)
            health_checks.append(("Directories", dirs_exist))
            
            # Import health (test key modules)
            try:
                import numpy, pandas, torch, sklearn
                health_checks.append(("ML Libraries", True))
            except ImportError:
                health_checks.append(("ML Libraries", False))
            
            # Print health summary
            print("🏥 System Health Check:")
            healthy_components = 0
            
            for component, healthy in health_checks:
                status = "✅" if healthy else "❌"
                print(f"  {status} {component}")
                if healthy:
                    healthy_components += 1
            
            health_percentage = (healthy_components / len(health_checks)) * 100
            print(f"🏥 Overall Health: {health_percentage:.0f}% ({healthy_components}/{len(health_checks)})")
            
            return health_percentage >= 80  # 80% minimum health required
            
        except Exception as e:
            print(f"System health check failed: {e}")
            return False


async def main():
    """Main initialization function."""
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print(f"📍 Working directory: {project_root}")
    
    # Initialize system
    initializer = SystemInitializer()
    success = await initializer.initialize()
    
    if success:
        print("\n🚀 System initialization completed successfully!")
        print("You can now start the trading system components.")
        
        # Show next steps
        print("\n📋 Next Steps:")
        print("1. Review and update config/config.yaml with your API keys")
        print("2. Set required environment variables")
        print("3. Run backtesting: python -m tests.backtesting.comprehensive_backtest")
        print("4. Start paper trading: python -m src.trading.paper_trading")
        print("5. Monitor logs in the logs/ directory")
        
        return 0
    else:
        print("\n❌ System initialization failed!")
        print("Please fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))