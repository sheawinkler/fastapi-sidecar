"""
Security module for the Crypto AI Trading System.

This module provides comprehensive security functionality including:
- Credential management and encryption
- API key rotation
- Audit logging
- Authentication and authorization
"""

from .credential_manager import (
    SecureCredentialManager,
    get_credential_manager,
    initialize_credentials_from_env
)

__all__ = [
    "SecureCredentialManager",
    "get_credential_manager", 
    "initialize_credentials_from_env"
]