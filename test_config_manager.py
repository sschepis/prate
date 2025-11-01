"""
Tests for Configuration Management System.
"""

import pytest
import os
import yaml
import tempfile
import time
from pathlib import Path

from prate.config_manager import (
    ConfigManager,
    ConfigSchema,
    ConfigurationError,
    create_default_schema
)


def test_load_yaml():
    """Test loading YAML configuration."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'test': 'value', 'number': 42}, f)
        config_path = f.name
    
    try:
        manager = ConfigManager(config_path)
        assert manager.get('test') == 'value'
        assert manager.get('number') == 42
    finally:
        os.unlink(config_path)


def test_missing_file():
    """Test error when config file doesn't exist."""
    with pytest.raises(ConfigurationError, match="not found"):
        ConfigManager('/nonexistent/path.yaml')


def test_invalid_yaml():
    """Test error on invalid YAML."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: content: :")
        config_path = f.name
    
    try:
        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            ConfigManager(config_path)
    finally:
        os.unlink(config_path)


def test_get_nested():
    """Test getting nested configuration values with dot notation."""
    config = {
        'level1': {
            'level2': {
                'value': 123
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        manager = ConfigManager(config_path)
        assert manager.get('level1.level2.value') == 123
        assert manager.get('level1.level2.missing', 'default') == 'default'
    finally:
        os.unlink(config_path)


def test_set_value():
    """Test setting configuration values."""
    manager = ConfigManager()
    manager.set('new.nested.value', 42)
    assert manager.get('new.nested.value') == 42


def test_env_override():
    """Test environment-specific configuration overrides."""
    # Create base config
    base_config = {'value': 1, 'env': 'base', 'nested': {'x': 10}}
    
    # Create dev override
    dev_config = {'value': 2, 'env': 'dev', 'nested': {'y': 20}}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / 'config.yaml'
        dev_path = Path(tmpdir) / 'config.dev.yaml'
        
        with open(base_path, 'w') as f:
            yaml.dump(base_config, f)
        
        with open(dev_path, 'w') as f:
            yaml.dump(dev_config, f)
        
        manager = ConfigManager(str(base_path), env='dev')
        
        assert manager.get('value') == 2  # Overridden
        assert manager.get('env') == 'dev'  # Overridden
        assert manager.get('nested.x') == 10  # From base
        assert manager.get('nested.y') == 20  # From dev


def test_validation_required_fields():
    """Test validation of required fields."""
    schema = ConfigSchema(required_fields=['required_field'])
    
    # Missing required field
    manager = ConfigManager()
    manager.set('other_field', 'value')
    
    with pytest.raises(ConfigurationError, match="Missing required field"):
        manager.validate(schema)
    
    # With required field
    manager.set('required_field', 'value')
    manager.validate(schema)  # Should not raise


def test_validation_field_types():
    """Test validation of field types."""
    schema = ConfigSchema(
        field_types={'int_field': int, 'str_field': str}
    )
    
    # Invalid type
    manager = ConfigManager()
    manager.set('int_field', 'not_an_int')
    
    with pytest.raises(ConfigurationError, match="Invalid type"):
        manager.validate(schema)
    
    # Correct types
    manager = ConfigManager()
    manager.set('int_field', 42)
    manager.set('str_field', 'text')
    manager.validate(schema)  # Should not raise


def test_validation_custom_validator():
    """Test custom field validators."""
    def positive_validator(value):
        if value <= 0:
            raise ValueError("Must be positive")
    
    schema = ConfigSchema(
        field_validators={'positive_field': positive_validator}
    )
    
    # Invalid value
    manager = ConfigManager()
    manager.set('positive_field', -1)
    
    with pytest.raises(ConfigurationError, match="Validation failed"):
        manager.validate(schema)
    
    # Valid value
    manager.set('positive_field', 10)
    manager.validate(schema)  # Should not raise


def test_validation_nested_schema():
    """Test validation of nested schemas."""
    nested_schema = ConfigSchema(required_fields=['nested_required'])
    schema = ConfigSchema(nested_schemas={'nested': nested_schema})
    
    # Missing nested required field
    manager = ConfigManager()
    manager.set('nested.other', 'value')
    
    with pytest.raises(ConfigurationError, match="Missing required field"):
        manager.validate(schema)
    
    # With nested required field
    manager.set('nested.nested_required', 'value')
    manager.validate(schema)  # Should not raise


def test_default_schema():
    """Test default PRATE schema creation and validation."""
    schema = create_default_schema()
    
    # Valid config
    config = {
        'primes': {'M': 100},
        'embedding': {'dim': 64},
        'tau': {'H_star': 2.5, 'Kp': 0.1, 'Ki': 0.01},
        'risk': {'max_drawdown': 0.15, 'max_position_size': 10000}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        manager = ConfigManager(config_path)
        manager.validate(schema)  # Should not raise
    finally:
        os.unlink(config_path)


def test_hot_reload():
    """Test hot-reload functionality."""
    # Create initial config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'value': 1}, f)
        config_path = f.name
    
    try:
        # Track reload callbacks
        reload_called = []
        
        def on_reload(old, new):
            reload_called.append((old, new))
        
        manager = ConfigManager(config_path, auto_reload=True, reload_interval=0.5)
        manager.on_reload(on_reload)
        
        assert manager.get('value') == 1
        
        # Modify config file
        time.sleep(0.1)  # Ensure different mtime
        with open(config_path, 'w') as f:
            yaml.dump({'value': 2}, f)
        
        # Wait for reload
        time.sleep(1.0)
        
        assert manager.get('value') == 2
        assert len(reload_called) > 0
        
        manager.stop_auto_reload()
    finally:
        os.unlink(config_path)


def test_save_config():
    """Test saving configuration to file."""
    manager = ConfigManager()
    manager.set('test', 'value')
    manager.set('nested.key', 42)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        output_path = f.name
    
    try:
        manager.save(output_path)
        
        # Load saved config
        with open(output_path, 'r') as f:
            saved_config = yaml.safe_load(f)
        
        assert saved_config['test'] == 'value'
        assert saved_config['nested']['key'] == 42
    finally:
        os.unlink(output_path)


def test_get_all():
    """Test getting entire configuration."""
    config = {'a': 1, 'b': {'c': 2}}
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        manager = ConfigManager(config_path)
        all_config = manager.get_all()
        
        assert all_config == config
        
        # Ensure it's a copy
        all_config['a'] = 999
        assert manager.get('a') == 1
    finally:
        os.unlink(config_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
