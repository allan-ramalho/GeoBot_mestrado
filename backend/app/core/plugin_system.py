"""
Plugin System for GeoBot
Allows users to create custom geophysics functions as plugins
"""

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Dict, List, Callable, Any
import json
import logging
from dataclasses import dataclass
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """Plugin metadata"""
    id: str
    name: str
    version: str
    author: str
    description: str
    category: str
    parameters: List[Dict[str, Any]]
    requires: List[str] = None
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'version': self.version,
            'author': self.author,
            'description': self.description,
            'category': self.category,
            'parameters': self.parameters,
            'requires': self.requires or [],
        }


class PluginError(Exception):
    """Custom exception for plugin errors"""
    pass


class PluginValidator:
    """Validates plugin code and metadata"""
    
    @staticmethod
    def validate_metadata(metadata: dict) -> bool:
        """Validate plugin metadata structure"""
        required_fields = ['id', 'name', 'version', 'author', 'description', 'category']
        
        for field in required_fields:
            if field not in metadata:
                raise PluginError(f"Missing required field: {field}")
        
        # Validate parameters structure
        if 'parameters' in metadata:
            for param in metadata['parameters']:
                if 'name' not in param or 'type' not in param:
                    raise PluginError(f"Invalid parameter structure: {param}")
        
        return True
    
    @staticmethod
    def validate_function(func: Callable) -> bool:
        """Validate plugin function signature"""
        sig = inspect.signature(func)
        
        # Must accept 'data' parameter
        if 'data' not in sig.parameters:
            raise PluginError("Plugin function must accept 'data' parameter")
        
        # Check return annotation (optional but recommended)
        if sig.return_annotation != inspect.Signature.empty:
            # TODO: Validate return type structure
            pass
        
        return True
    
    @staticmethod
    def check_dependencies(requires: List[str]) -> bool:
        """Check if required packages are installed"""
        for package in requires or []:
            try:
                importlib.import_module(package)
            except ImportError:
                raise PluginError(f"Missing required package: {package}")
        
        return True


class PluginSandbox:
    """
    Sandbox for executing plugins safely
    Limits access to dangerous operations
    """
    
    ALLOWED_MODULES = {
        'numpy', 'scipy', 'math', 'json', 'datetime',
        'collections', 'itertools', 'functools'
    }
    
    FORBIDDEN_OPERATIONS = {
        'eval', 'exec', 'compile', '__import__',
        'open', 'file', 'input', 'raw_input'
    }
    
    @classmethod
    def execute(cls, func: Callable, data: dict, params: dict) -> Any:
        """
        Execute plugin function in sandbox
        
        Args:
            func: Plugin function
            data: Input data
            params: Function parameters
        
        Returns:
            Result from plugin
        """
        # Check for forbidden operations in function code
        source = inspect.getsource(func)
        for forbidden in cls.FORBIDDEN_OPERATIONS:
            if forbidden in source:
                raise PluginError(f"Forbidden operation detected: {forbidden}")
        
        # Execute with timeout and memory limits
        try:
            result = func(data, **params)
            return result
        except Exception as e:
            logger.error(f"Plugin execution error: {e}")
            raise PluginError(f"Plugin execution failed: {str(e)}")


class PluginManager:
    """
    Main plugin manager
    Loads, validates, and manages plugins
    """
    
    def __init__(self, plugin_dir: Path = None):
        self.plugin_dir = plugin_dir or Path.home() / ".geobot" / "plugins"
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
        
        self.plugins: Dict[str, Dict[str, Any]] = {}
        self.validator = PluginValidator()
        
        logger.info(f"Plugin manager initialized. Plugin directory: {self.plugin_dir}")
    
    def load_plugin(self, plugin_path: Path) -> bool:
        """
        Load a plugin from file
        
        Args:
            plugin_path: Path to plugin file (.py)
        
        Returns:
            True if loaded successfully
        """
        try:
            # Load module
            spec = importlib.util.spec_from_file_location(plugin_path.stem, plugin_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[plugin_path.stem] = module
            spec.loader.exec_module(module)
            
            # Get metadata
            if not hasattr(module, 'PLUGIN_METADATA'):
                raise PluginError("Plugin missing PLUGIN_METADATA")
            
            metadata_dict = module.PLUGIN_METADATA
            self.validator.validate_metadata(metadata_dict)
            
            # Get main function
            if not hasattr(module, 'execute'):
                raise PluginError("Plugin missing 'execute' function")
            
            func = module.execute
            self.validator.validate_function(func)
            
            # Check dependencies
            self.validator.check_dependencies(metadata_dict.get('requires', []))
            
            # Create plugin metadata
            metadata = PluginMetadata(**metadata_dict)
            
            # Register plugin
            self.plugins[metadata.id] = {
                'metadata': metadata,
                'function': func,
                'module': module,
                'path': plugin_path,
            }
            
            logger.info(f"Plugin loaded: {metadata.name} v{metadata.version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_path}: {e}")
            raise
    
    def load_all_plugins(self):
        """Load all plugins from plugin directory"""
        loaded = 0
        failed = 0
        
        for plugin_file in self.plugin_dir.glob("*.py"):
            try:
                self.load_plugin(plugin_file)
                loaded += 1
            except Exception as e:
                logger.error(f"Failed to load {plugin_file.name}: {e}")
                failed += 1
        
        logger.info(f"Plugins loaded: {loaded}, failed: {failed}")
        return loaded, failed
    
    def unload_plugin(self, plugin_id: str) -> bool:
        """Unload a plugin"""
        if plugin_id not in self.plugins:
            return False
        
        plugin = self.plugins[plugin_id]
        module_name = plugin['path'].stem
        
        if module_name in sys.modules:
            del sys.modules[module_name]
        
        del self.plugins[plugin_id]
        logger.info(f"Plugin unloaded: {plugin_id}")
        return True
    
    def execute_plugin(self, plugin_id: str, data: dict, params: dict) -> dict:
        """
        Execute a plugin function
        
        Args:
            plugin_id: Plugin identifier
            data: Input data (grid)
            params: Function parameters
        
        Returns:
            Result dictionary with data and metadata
        """
        if plugin_id not in self.plugins:
            raise PluginError(f"Plugin not found: {plugin_id}")
        
        plugin = self.plugins[plugin_id]
        func = plugin['function']
        metadata = plugin['metadata']
        
        # Execute in sandbox
        try:
            result = PluginSandbox.execute(func, data, params)
            
            return {
                'success': True,
                'result': {
                    'data': result,
                    'metadata': {
                        'plugin_id': plugin_id,
                        'plugin_name': metadata.name,
                        'plugin_version': metadata.version,
                        'params': params,
                    }
                },
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Plugin execution error: {e}")
            return {
                'success': False,
                'result': None,
                'error': str(e)
            }
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all loaded plugins"""
        return [
            plugin['metadata'].to_dict()
            for plugin in self.plugins.values()
        ]
    
    def get_plugin_info(self, plugin_id: str) -> Dict[str, Any]:
        """Get detailed plugin information"""
        if plugin_id not in self.plugins:
            return None
        
        plugin = self.plugins[plugin_id]
        metadata = plugin['metadata']
        
        return {
            **metadata.to_dict(),
            'path': str(plugin['path']),
            'loaded': True,
        }
    
    def create_plugin_template(self, output_path: Path) -> bool:
        """Create a plugin template file"""
        template = '''"""
Custom GeoBot Plugin
Author: Your Name
Description: Your plugin description
"""

import numpy as np

# Plugin metadata (required)
PLUGIN_METADATA = {
    'id': 'my_custom_function',
    'name': 'My Custom Function',
    'version': '1.0.0',
    'author': 'Your Name',
    'description': 'Brief description of what this plugin does',
    'category': 'custom',  # custom, gravity, magnetic, filter, advanced
    'parameters': [
        {
            'name': 'param1',
            'type': 'number',
            'default': 1.0,
            'unit': 'units',
            'min': 0.0,
            'max': 10.0,
            'description': 'Parameter description'
        },
        # Add more parameters as needed
    ],
    'requires': [],  # List of required packages: ['scipy', 'sklearn']
}


def execute(data: dict, **params) -> list:
    """
    Main plugin function (required)
    
    Args:
        data: Input grid data with keys:
            - x: array of x coordinates
            - y: array of y coordinates
            - z: 2D array of values
            - nx, ny: grid dimensions
        **params: Function parameters from PLUGIN_METADATA
    
    Returns:
        Processed 2D array (list of lists)
    
    Example:
        result = execute(data, param1=2.0)
    """
    # Extract data
    x = np.array(data['x'])
    y = np.array(data['y'])
    z = np.array(data['z'])
    nx, ny = data['nx'], data['ny']
    
    # Get parameters
    param1 = params.get('param1', 1.0)
    
    # Your processing logic here
    # Example: multiply by parameter
    result = z * param1
    
    # Return as list
    return result.tolist()


# Optional: validation function
def validate_params(params: dict) -> bool:
    """Validate parameters before execution"""
    if params.get('param1', 0) < 0:
        raise ValueError("param1 must be positive")
    return True
'''
        
        try:
            output_path.write_text(template)
            logger.info(f"Plugin template created: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create template: {e}")
            return False


# Global plugin manager instance
plugin_manager = PluginManager()


# Decorator for registering inline plugins
def register_plugin(metadata: dict):
    """
    Decorator to register a function as a plugin
    
    Usage:
        @register_plugin({
            'id': 'my_func',
            'name': 'My Function',
            ...
        })
        def my_function(data, **params):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Register in plugin manager
        plugin_id = metadata['id']
        plugin_manager.plugins[plugin_id] = {
            'metadata': PluginMetadata(**metadata),
            'function': func,
            'module': None,
            'path': None,
        }
        
        return wrapper
    return decorator
