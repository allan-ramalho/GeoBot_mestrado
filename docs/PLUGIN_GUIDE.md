# Plugin System Example

This example demonstrates how to create custom plugins for GeoBot.

## Creating a Plugin

Create a `.py` file with the following structure:

```python
"""
My Custom Plugin
Author: Your Name
"""

import numpy as np

# Plugin metadata (REQUIRED)
PLUGIN_METADATA = {
    'id': 'my_custom_filter',
    'name': 'My Custom Filter',
    'version': '1.0.0',
    'author': 'Your Name',
    'description': 'A custom filtering function',
    'category': 'custom',
    'parameters': [
        {
            'name': 'strength',
            'type': 'number',
            'default': 1.0,
            'min': 0.0,
            'max': 10.0,
            'unit': 'multiplier',
            'description': 'Filter strength'
        },
    ],
    'requires': [],  # List required packages
}

# Main function (REQUIRED)
def execute(data: dict, **params) -> list:
    """
    Process data with custom logic
    
    Args:
        data: Input grid with keys: x, y, z, nx, ny
        **params: Parameters from metadata
    
    Returns:
        Processed 2D array (list of lists)
    """
    # Extract data
    z = np.array(data['z'])
    strength = params.get('strength', 1.0)
    
    # Your processing logic
    result = z * strength
    
    # Return as list
    return result.tolist()
```

## Loading a Plugin

### Method 1: Upload via API

```python
import requests

with open('my_plugin.py', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/plugins/upload',
        files={'file': f}
    )
```

### Method 2: Place in plugins directory

Save your plugin to: `~/.geobot/plugins/my_plugin.py`

Then reload:
```python
response = requests.post('http://localhost:8000/api/plugins/reload')
```

## Executing a Plugin

```python
import requests

response = requests.post(
    'http://localhost:8000/api/plugins/execute/my_custom_filter',
    json={
        'data': {
            'x': [0, 1, 2],
            'y': [0, 1, 2],
            'z': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            'nx': 3,
            'ny': 3,
        },
        'params': {
            'strength': 2.0,
        }
    }
)
```

## Plugin Examples

### Example 1: Custom Derivative

```python
PLUGIN_METADATA = {
    'id': 'custom_derivative',
    'name': 'Custom Derivative',
    'version': '1.0.0',
    'author': 'GeoBot',
    'description': 'Compute custom directional derivative',
    'category': 'custom',
    'parameters': [
        {
            'name': 'azimuth',
            'type': 'number',
            'default': 0,
            'min': 0,
            'max': 360,
            'unit': 'degrees',
            'description': 'Direction angle'
        },
    ],
}

def execute(data, **params):
    import numpy as np
    from scipy.ndimage import sobel
    
    z = np.array(data['z'])
    azimuth = np.radians(params.get('azimuth', 0))
    
    # Compute gradients
    dx = sobel(z, axis=1)
    dy = sobel(z, axis=0)
    
    # Directional derivative
    result = dx * np.cos(azimuth) + dy * np.sin(azimuth)
    
    return result.tolist()
```

### Example 2: Statistical Filter

```python
PLUGIN_METADATA = {
    'id': 'statistical_filter',
    'name': 'Statistical Filter',
    'version': '1.0.0',
    'author': 'GeoBot',
    'description': 'Remove values outside N standard deviations',
    'category': 'filter',
    'parameters': [
        {
            'name': 'n_std',
            'type': 'number',
            'default': 3.0,
            'min': 1.0,
            'max': 5.0,
            'unit': 'sigma',
            'description': 'Number of standard deviations'
        },
    ],
}

def execute(data, **params):
    import numpy as np
    
    z = np.array(data['z'])
    n_std = params.get('n_std', 3.0)
    
    # Compute statistics
    mean = np.mean(z)
    std = np.std(z)
    
    # Remove outliers
    mask = np.abs(z - mean) < n_std * std
    result = np.where(mask, z, mean)
    
    return result.tolist()
```

## Testing Your Plugin

```python
# Test locally before uploading
import numpy as np

# Import your plugin
from my_plugin import execute, PLUGIN_METADATA

# Create test data
data = {
    'x': np.linspace(0, 10, 50),
    'y': np.linspace(0, 10, 50),
    'z': np.random.rand(50, 50).tolist(),
    'nx': 50,
    'ny': 50,
}

# Execute
result = execute(data, strength=2.0)

# Verify
assert len(result) == 50
assert len(result[0]) == 50
print("Test passed!")
```

## Best Practices

1. **Always validate parameters**
   ```python
   def execute(data, **params):
       strength = params.get('strength', 1.0)
       if strength < 0:
           raise ValueError("Strength must be positive")
       # ...
   ```

2. **Handle errors gracefully**
   ```python
   try:
       result = complex_operation(data)
   except Exception as e:
       logger.error(f"Operation failed: {e}")
       return data['z']  # Return original data
   ```

3. **Document your code**
   - Clear docstrings
   - Explain algorithm
   - Cite references if applicable

4. **Test with different grid sizes**
   - Small grids (10x10)
   - Medium grids (100x100)
   - Large grids (1000x1000)

5. **Consider performance**
   - Use NumPy vectorization
   - Avoid Python loops
   - Test with timeit

## Security

Plugins run in a sandbox with limited permissions:
- ✅ Allowed: NumPy, SciPy, math operations
- ❌ Forbidden: File I/O, network access, eval/exec

## Troubleshooting

**Plugin won't load:**
- Check syntax errors
- Verify PLUGIN_METADATA format
- Ensure execute() function exists

**Execution fails:**
- Validate input data format
- Check parameter types
- Review error messages in logs

**Performance issues:**
- Profile your code
- Use NumPy operations
- Consider chunking large data
