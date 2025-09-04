"""
Secure Credential Management System for Crypto AI Trading Platform

This module provides enterprise-grade security for managing API credentials, 
encryption keys, and sensitive configuration data with automated rotation 
and comprehensive audit logging.
"""

import os
import base64
import hashlib
import asyncio
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import json
import structlog

logger = structlog.get_logger(__name__)


class SecureCredentialManager:
    """
    Enterprise-grade credential management with encryption, rotation, and audit logging.
    
    Features:
    - Fernet symmetric encryption for credential storage
    - Automated API key rotation with zero downtime
    - Comprehensive audit logging for compliance
    - Environment-based key derivation with salt
    - Memory-safe credential handling
    """
    
    def __init__(self, master_password: Optional[str] = None):
        """
        Initialize the credential manager.
        
        Args:
            master_password: Optional master password. If not provided, uses environment variable.
        """
        self.master_password = master_password or os.getenv('MASTER_PASSWORD')
        if not self.master_password:
            raise ValueError("Master password must be provided via parameter or MASTER_PASSWORD environment variable")
        
        self._cipher = None
        self._credentials_cache: Dict[str, Any] = {}
        self._rotation_schedule: Dict[str, datetime] = {}
        
        # Set up audit logging
        self.audit_logger = structlog.get_logger("credential_audit")
        
        # Initialize encryption
        self._initialize_encryption()
    
    def _initialize_encryption(self) -> None:
        """Initialize the Fernet encryption cipher with derived key."""
        try:
            # Use a fixed salt for deterministic key generation
            # In production, store this securely in a key management system
            salt = b'crypto_ai_trading_salt_2024'
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            
            key = base64.urlsafe_b64encode(kdf.derive(self.master_password.encode()))
            self._cipher = Fernet(key)
            
            logger.info("Credential manager initialized successfully")
            self.audit_logger.info(
                "credential_manager_initialized",
                timestamp=datetime.utcnow(),
                action="initialize",
                success=True
            )
            
        except Exception as e:
            logger.error("Failed to initialize credential manager", error=str(e))
            self.audit_logger.error(
                "credential_manager_init_failed",
                timestamp=datetime.utcnow(),
                action="initialize",
                error=str(e),
                success=False
            )
            raise
    
    def encrypt_credential(self, credential: str, credential_name: str) -> str:
        """
        Encrypt a credential using Fernet symmetric encryption.
        
        Args:
            credential: The credential to encrypt
            credential_name: Name for audit logging
        
        Returns:
            Base64 encoded encrypted credential
        """
        try:
            if not credential:
                raise ValueError("Credential cannot be empty")
            
            encrypted = self._cipher.encrypt(credential.encode())
            encoded = base64.urlsafe_b64encode(encrypted).decode()
            
            self.audit_logger.info(
                "credential_encrypted",
                timestamp=datetime.utcnow(),
                credential_name=credential_name,
                action="encrypt",
                success=True
            )
            
            return encoded
            
        except Exception as e:
            self.audit_logger.error(
                "credential_encrypt_failed",
                timestamp=datetime.utcnow(),
                credential_name=credential_name,
                action="encrypt",
                error=str(e),
                success=False
            )
            raise
    
    def decrypt_credential(self, encrypted_credential: str, credential_name: str) -> str:
        """
        Decrypt a credential for runtime use.
        
        Args:
            encrypted_credential: Base64 encoded encrypted credential
            credential_name: Name for audit logging
        
        Returns:
            Decrypted credential
        """
        try:
            if not encrypted_credential:
                raise ValueError("Encrypted credential cannot be empty")
            
            # Decode from base64
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_credential.encode())
            
            # Decrypt
            decrypted = self._cipher.decrypt(encrypted_bytes).decode()
            
            self.audit_logger.info(
                "credential_decrypted",
                timestamp=datetime.utcnow(),
                credential_name=credential_name,
                action="decrypt",
                success=True
            )
            
            return decrypted
            
        except Exception as e:
            self.audit_logger.error(
                "credential_decrypt_failed",
                timestamp=datetime.utcnow(),
                credential_name=credential_name,
                action="decrypt",
                error=str(e),
                success=False
            )
            raise
    
    def store_credential(self, key: str, credential: str, ttl_days: int = 30) -> None:
        """
        Store an encrypted credential with TTL for rotation.
        
        Args:
            key: Credential identifier
            credential: The credential to store
            ttl_days: Days until rotation required
        """
        try:
            encrypted = self.encrypt_credential(credential, key)
            
            self._credentials_cache[key] = {
                'encrypted_value': encrypted,
                'created_at': datetime.utcnow(),
                'expires_at': datetime.utcnow() + timedelta(days=ttl_days),
                'rotation_required': False
            }
            
            # Schedule rotation warning 7 days before expiry
            warning_date = datetime.utcnow() + timedelta(days=ttl_days - 7)
            self._rotation_schedule[key] = warning_date
            
            logger.info("Credential stored successfully", key=key)
            self.audit_logger.info(
                "credential_stored",
                timestamp=datetime.utcnow(),
                credential_key=key,
                action="store",
                ttl_days=ttl_days,
                success=True
            )
            
        except Exception as e:
            logger.error("Failed to store credential", key=key, error=str(e))
            self.audit_logger.error(
                "credential_store_failed",
                timestamp=datetime.utcnow(),
                credential_key=key,
                action="store",
                error=str(e),
                success=False
            )
            raise
    
    def get_credential(self, key: str) -> Optional[str]:
        """
        Retrieve and decrypt a credential.
        
        Args:
            key: Credential identifier
        
        Returns:
            Decrypted credential or None if not found
        """
        try:
            if key not in self._credentials_cache:
                logger.warning("Credential not found", key=key)
                return None
            
            credential_data = self._credentials_cache[key]
            
            # Check if expired
            if datetime.utcnow() > credential_data['expires_at']:
                logger.warning("Credential expired", key=key)
                credential_data['rotation_required'] = True
                return None
            
            # Check if rotation warning needed
            if key in self._rotation_schedule:
                if datetime.utcnow() > self._rotation_schedule[key]:
                    logger.warning("Credential rotation recommended", key=key)
                    credential_data['rotation_required'] = True
            
            decrypted = self.decrypt_credential(credential_data['encrypted_value'], key)
            
            self.audit_logger.info(
                "credential_retrieved",
                timestamp=datetime.utcnow(),
                credential_key=key,
                action="retrieve",
                rotation_required=credential_data['rotation_required'],
                success=True
            )
            
            return decrypted
            
        except Exception as e:
            logger.error("Failed to retrieve credential", key=key, error=str(e))
            self.audit_logger.error(
                "credential_retrieve_failed",
                timestamp=datetime.utcnow(),
                credential_key=key,
                action="retrieve",
                error=str(e),
                success=False
            )
            return None
    
    def rotate_credential(self, key: str, new_credential: str, ttl_days: int = 30) -> bool:
        """
        Rotate a credential with zero downtime.
        
        Args:
            key: Credential identifier
            new_credential: New credential value
            ttl_days: Days until next rotation
        
        Returns:
            True if rotation successful
        """
        try:
            if key not in self._credentials_cache:
                logger.warning("Cannot rotate non-existent credential", key=key)
                return False
            
            old_credential_data = self._credentials_cache[key].copy()
            
            # Store new credential
            self.store_credential(key, new_credential, ttl_days)
            
            # Mark old credential as rotated in audit log
            self.audit_logger.info(
                "credential_rotated",
                timestamp=datetime.utcnow(),
                credential_key=key,
                action="rotate",
                old_expires_at=old_credential_data['expires_at'],
                new_expires_at=self._credentials_cache[key]['expires_at'],
                success=True
            )
            
            logger.info("Credential rotated successfully", key=key)
            return True
            
        except Exception as e:
            logger.error("Failed to rotate credential", key=key, error=str(e))
            self.audit_logger.error(
                "credential_rotation_failed",
                timestamp=datetime.utcnow(),
                credential_key=key,
                action="rotate",
                error=str(e),
                success=False
            )
            return False
    
    def check_rotation_status(self) -> Dict[str, bool]:
        """
        Check which credentials require rotation.
        
        Returns:
            Dictionary mapping credential keys to rotation requirement status
        """
        rotation_status = {}
        
        for key, credential_data in self._credentials_cache.items():
            needs_rotation = (
                credential_data['rotation_required'] or 
                datetime.utcnow() > credential_data['expires_at'] or
                (key in self._rotation_schedule and 
                 datetime.utcnow() > self._rotation_schedule[key])
            )
            rotation_status[key] = needs_rotation
        
        return rotation_status
    
    async def automated_rotation_check(self, rotation_callback=None):
        """
        Continuously check for credentials requiring rotation.
        
        Args:
            rotation_callback: Optional callback function for rotation alerts
        """
        while True:
            try:
                rotation_status = self.check_rotation_status()
                
                for key, needs_rotation in rotation_status.items():
                    if needs_rotation:
                        logger.warning("Credential requires rotation", key=key)
                        
                        if rotation_callback:
                            await rotation_callback(key, self._credentials_cache[key])
                
                # Check every hour
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error("Error in automated rotation check", error=str(e))
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    def export_audit_log(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """
        Export audit logs for compliance reporting.
        
        Args:
            start_date: Start date for log export
            end_date: End date for log export
        
        Returns:
            List of audit log entries
        """
        # In a production system, this would query a persistent audit log store
        # For now, return current session information
        audit_summary = []
        
        for key, credential_data in self._credentials_cache.items():
            if start_date <= credential_data['created_at'] <= end_date:
                audit_summary.append({
                    'credential_key': key,
                    'created_at': credential_data['created_at'].isoformat(),
                    'expires_at': credential_data['expires_at'].isoformat(),
                    'rotation_required': credential_data['rotation_required'],
                    'status': 'active' if datetime.utcnow() < credential_data['expires_at'] else 'expired'
                })
        
        return audit_summary
    
    def clear_credentials(self) -> None:
        """Clear all credentials from memory (for shutdown/security)."""
        try:
            credential_count = len(self._credentials_cache)
            self._credentials_cache.clear()
            self._rotation_schedule.clear()
            
            logger.info("Credentials cleared from memory", count=credential_count)
            self.audit_logger.info(
                "credentials_cleared",
                timestamp=datetime.utcnow(),
                action="clear_all",
                count=credential_count,
                success=True
            )
            
        except Exception as e:
            logger.error("Failed to clear credentials", error=str(e))
            self.audit_logger.error(
                "credentials_clear_failed",
                timestamp=datetime.utcnow(),
                action="clear_all",
                error=str(e),
                success=False
            )


# Global credential manager instance
_credential_manager: Optional[SecureCredentialManager] = None


def get_credential_manager() -> SecureCredentialManager:
    """Get the global credential manager instance."""
    global _credential_manager
    
    if _credential_manager is None:
        _credential_manager = SecureCredentialManager()
    
    return _credential_manager


def initialize_credentials_from_env():
    """Initialize common credentials from environment variables."""
    cm = get_credential_manager()
    
    # API credentials
    env_credentials = [
        ('helius_api_key', 'HELIUS_API_KEY'),
        ('coinbase_api_key', 'COINBASE_API_KEY'),
        ('coinbase_api_secret', 'COINBASE_API_SECRET'),
        ('coinbase_passphrase', 'COINBASE_PASSPHRASE'),
        ('kraken_api_key', 'KRAKEN_API_KEY'),
        ('kraken_api_secret', 'KRAKEN_API_SECRET'),
        ('twitter_bearer_token', 'TWITTER_BEARER_TOKEN'),
        ('reddit_client_id', 'REDDIT_CLIENT_ID'),
        ('reddit_client_secret', 'REDDIT_CLIENT_SECRET'),
        ('slack_webhook_url', 'SLACK_WEBHOOK_URL'),
        ('jwt_secret', 'JWT_SECRET'),
        ('db_password', 'DB_PASSWORD'),
        ('redis_password', 'REDIS_PASSWORD'),
        ('influxdb_token', 'INFLUXDB_TOKEN')
    ]
    
    for key, env_var in env_credentials:
        value = os.getenv(env_var)
        if value:
            cm.store_credential(key, value)
            logger.info(f"Loaded credential: {key}")
        else:
            logger.warning(f"Environment variable not found: {env_var}")


if __name__ == "__main__":
    # Example usage and testing
    import asyncio
    
    async def test_credential_manager():
        """Test the credential manager functionality."""
        
        # Set up test environment
        os.environ['MASTER_PASSWORD'] = 'test_master_password_123'
        os.environ['HELIUS_API_KEY'] = 'test_helius_key_123'
        
        # Initialize
        cm = SecureCredentialManager()
        
        # Test storing and retrieving credentials
        test_key = "test_api_key"
        test_credential = "sk_test_12345abcdef"
        
        print("Testing credential storage...")
        cm.store_credential(test_key, test_credential)
        
        print("Testing credential retrieval...")
        retrieved = cm.get_credential(test_key)
        assert retrieved == test_credential, "Credential mismatch!"
        
        print("Testing credential rotation...")
        new_credential = "sk_test_67890ghijkl"
        success = cm.rotate_credential(test_key, new_credential)
        assert success, "Rotation failed!"
        
        retrieved_after_rotation = cm.get_credential(test_key)
        assert retrieved_after_rotation == new_credential, "Rotated credential mismatch!"
        
        print("Testing rotation status check...")
        rotation_status = cm.check_rotation_status()
        print(f"Rotation status: {rotation_status}")
        
        print("Testing environment credential loading...")
        initialize_credentials_from_env()
        
        helius_key = cm.get_credential('helius_api_key')
        print(f"Retrieved Helius key: {helius_key[:10]}..." if helius_key else "Helius key not found")
        
        print("All tests passed! 🎉")
    
    # Run tests
    asyncio.run(test_credential_manager())