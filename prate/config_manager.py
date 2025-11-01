"""
Configuration Management System for PRATE.

Provides YAML-based configuration loading, validation, hot-reload support,
and environment-specific configurations.
"""

import os
import yaml
from typing import Any, Dict, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, field
import threading
import time


@dataclass
class ConfigSchema:
    """Schema definition for configuration validation."""
    
    required_fields: list = field(default_factory=list)
    field_types: Dict[str, type] = field(default_factory=dict)
    field_validators: Dict[str, Callable] = field(default_factory=dict)
    nested_schemas: Dict[str, 'ConfigSchema'] = field(default_factory=dict)


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass


class ConfigManager:
    """
    Manages YAML configuration loading, validation, and hot-reload.
    
    Features:
    - YAML configuration loading
    - Schema-based validation
    - Environment-specific configs (dev, staging, prod)
    - Hot-reload support with file watching
    - Default value handling
    """
    
    def __init__(self, config_path: Optional[str] = None, 
                 env: Optional[str] = None,
                 auto_reload: bool = False,
                 reload_interval: float = 5.0):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to main configuration file
            env: Environment name (dev, staging, prod). If None, uses ENV env var
            auto_reload: Enable automatic hot-reload of config file
            reload_interval: Seconds between reload checks (if auto_reload=True)
        """
        self.config_path = config_path
        self.env = env or os.getenv('ENV', 'dev')
        self.auto_reload = auto_reload
        self.reload_interval = reload_interval
        
        self._config: Dict[str, Any] = {}
        self._schema: Optional[ConfigSchema] = None
        self._last_modified: Optional[float] = None
        self._reload_callbacks: list = []
        self._reload_thread: Optional[threading.Thread] = None
        self._stop_reload = threading.Event()
        
        if config_path:
            self.load(config_path)
            
            if auto_reload:
                self.start_auto_reload()
    
    def load(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Loaded configuration dictionary
            
        Raises:
            ConfigurationError: If file not found or invalid YAML
        """
        path = Path(config_path)
        
        if not path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {config_path}: {e}")
        
        if config is None:
            config = {}
        
        # Load environment-specific overrides
        config = self._apply_env_overrides(config, path)
        
        # Store config
        self._config = config
        self._last_modified = path.stat().st_mtime
        self.config_path = str(path)
        
        # Validate if schema is set
        if self._schema:
            self.validate(self._schema)
        
        return config
    
    def _apply_env_overrides(self, config: Dict[str, Any], base_path: Path) -> Dict[str, Any]:
        """
        Apply environment-specific configuration overrides.
        
        Looks for files named:
        - config.{env}.yaml
        - config_{env}.yaml
        
        Args:
            config: Base configuration
            base_path: Path to base config file
            
        Returns:
            Configuration with environment overrides applied
        """
        # Try different naming patterns
        env_patterns = [
            base_path.parent / f"{base_path.stem}.{self.env}.yaml",
            base_path.parent / f"{base_path.stem}_{self.env}.yaml",
        ]
        
        for env_path in env_patterns:
            if env_path.exists():
                try:
                    with open(env_path, 'r') as f:
                        env_config = yaml.safe_load(f) or {}
                    
                    # Deep merge
                    config = self._deep_merge(config, env_config)
                except yaml.YAMLError as e:
                    raise ConfigurationError(f"Invalid YAML in {env_path}: {e}")
        
        return config
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Override dictionary
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def validate(self, schema: ConfigSchema) -> None:
        """
        Validate configuration against schema.
        
        Args:
            schema: Configuration schema
            
        Raises:
            ConfigurationError: If validation fails
        """
        self._schema = schema
        self._validate_dict(self._config, schema, path="config")
    
    def _validate_dict(self, config: Dict[str, Any], schema: ConfigSchema, path: str) -> None:
        """
        Recursively validate configuration dictionary.
        
        Args:
            config: Configuration to validate
            schema: Schema to validate against
            path: Current path in config (for error messages)
            
        Raises:
            ConfigurationError: If validation fails
        """
        # Check required fields
        for field in schema.required_fields:
            if field not in config:
                raise ConfigurationError(f"Missing required field: {path}.{field}")
        
        # Check field types
        for field, expected_type in schema.field_types.items():
            if field in config:
                value = config[field]
                if not isinstance(value, expected_type):
                    raise ConfigurationError(
                        f"Invalid type for {path}.{field}: "
                        f"expected {expected_type.__name__}, got {type(value).__name__}"
                    )
        
        # Run custom validators
        for field, validator in schema.field_validators.items():
            if field in config:
                try:
                    validator(config[field])
                except Exception as e:
                    raise ConfigurationError(f"Validation failed for {path}.{field}: {e}")
        
        # Validate nested schemas
        for field, nested_schema in schema.nested_schemas.items():
            if field in config and isinstance(config[field], dict):
                self._validate_dict(config[field], nested_schema, f"{path}.{field}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Supports dot notation for nested keys (e.g., "primes.M").
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key.
        
        Supports dot notation for nested keys (e.g., "primes.M").
        
        Args:
            key: Configuration key
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get entire configuration dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self._config.copy()
    
    def start_auto_reload(self) -> None:
        """Start automatic configuration reload in background thread."""
        if self._reload_thread and self._reload_thread.is_alive():
            return
        
        self._stop_reload.clear()
        self._reload_thread = threading.Thread(target=self._reload_loop, daemon=True)
        self._reload_thread.start()
    
    def stop_auto_reload(self) -> None:
        """Stop automatic configuration reload."""
        self._stop_reload.set()
        if self._reload_thread:
            self._reload_thread.join(timeout=self.reload_interval + 1)
    
    def _reload_loop(self) -> None:
        """Background thread loop for automatic reload."""
        while not self._stop_reload.is_set():
            try:
                if self.config_path and self._check_modified():
                    self._reload()
            except Exception as e:
                # Log error but don't crash
                print(f"Error during auto-reload: {e}")
            
            self._stop_reload.wait(self.reload_interval)
    
    def _check_modified(self) -> bool:
        """
        Check if configuration file has been modified.
        
        Returns:
            True if file has been modified since last load
        """
        if not self.config_path:
            return False
        
        path = Path(self.config_path)
        if not path.exists():
            return False
        
        current_mtime = path.stat().st_mtime
        return current_mtime != self._last_modified
    
    def _reload(self) -> None:
        """Reload configuration and notify callbacks."""
        old_config = self._config.copy()
        
        try:
            self.load(self.config_path)
            
            # Notify callbacks
            for callback in self._reload_callbacks:
                callback(old_config, self._config)
        except Exception as e:
            # Restore old config on error
            self._config = old_config
            raise ConfigurationError(f"Failed to reload configuration: {e}")
    
    def on_reload(self, callback: Callable[[Dict, Dict], None]) -> None:
        """
        Register callback to be called when configuration is reloaded.
        
        Args:
            callback: Function(old_config, new_config) to call on reload
        """
        self._reload_callbacks.append(callback)
    
    def save(self, output_path: Optional[str] = None) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            output_path: Path to save to. If None, uses original config_path
        """
        path = output_path or self.config_path
        if not path:
            raise ConfigurationError("No output path specified")
        
        with open(path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)


def create_default_schema() -> ConfigSchema:
    """
    Create default schema for PRATE configuration.
    
    Returns:
        Default configuration schema
    """
    # Validators
    def positive_int(value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"Must be positive integer, got {value}")
    
    def positive_float(value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"Must be positive number, got {value}")
    
    def probability(value):
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"Must be in [0, 1], got {value}")
    
    # Nested schemas
    primes_schema = ConfigSchema(
        required_fields=['M'],
        field_types={'M': int},
        field_validators={'M': positive_int}
    )
    
    embedding_schema = ConfigSchema(
        required_fields=['dim'],
        field_types={'dim': int},
        field_validators={'dim': positive_int}
    )
    
    tau_schema = ConfigSchema(
        required_fields=['H_star', 'Kp', 'Ki'],
        field_types={'H_star': (int, float), 'Kp': (int, float), 'Ki': (int, float)},
        field_validators={'H_star': positive_float}
    )
    
    risk_schema = ConfigSchema(
        required_fields=['max_drawdown', 'max_position_size'],
        field_types={'max_drawdown': (int, float), 'max_position_size': (int, float)},
        field_validators={
            'max_drawdown': lambda x: positive_float(x) if x > 0 else None,
            'max_position_size': positive_float
        }
    )
    
    # Main schema
    main_schema = ConfigSchema(
        required_fields=['primes', 'embedding', 'tau', 'risk'],
        nested_schemas={
            'primes': primes_schema,
            'embedding': embedding_schema,
            'tau': tau_schema,
            'risk': risk_schema
        }
    )
    
    return main_schema
