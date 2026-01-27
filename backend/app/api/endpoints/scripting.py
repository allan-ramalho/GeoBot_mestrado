"""
Scripting API Endpoint
Execute Python code in sandboxed environment
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging
import sys
from io import StringIO
import contextlib
import ast

router = APIRouter(prefix="/api/scripting", tags=["Scripting"])
logger = logging.getLogger(__name__)


class ScriptExecutor:
    """Execute Python code safely"""
    
    ALLOWED_MODULES = {
        'numpy': 'np',
        'scipy': 'sp',
        'math': 'math',
        'json': 'json',
    }
    
    FORBIDDEN_OPERATIONS = {
        '__import__', 'eval', 'exec', 'compile',
        'open', 'file', 'input', 'raw_input',
        '__builtins__', 'globals', 'locals'
    }
    
    @classmethod
    def validate_code(cls, code: str) -> bool:
        """Validate code for dangerous operations"""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Syntax error: {e}")
        
        for node in ast.walk(tree):
            # Check for forbidden names
            if isinstance(node, ast.Name):
                if node.id in cls.FORBIDDEN_OPERATIONS:
                    raise ValueError(f"Forbidden operation: {node.id}")
            
            # Check for forbidden imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in cls.ALLOWED_MODULES:
                        raise ValueError(f"Import not allowed: {alias.name}")
        
        return True
    
    @classmethod
    def execute(cls, code: str) -> Dict[str, Any]:
        """
        Execute code and capture output
        
        Returns:
            Dict with success, output, error
        """
        # Validate code
        try:
            cls.validate_code(code)
        except ValueError as e:
            return {
                "success": False,
                "output": None,
                "error": str(e),
            }
        
        # Capture output
        stdout = StringIO()
        stderr = StringIO()
        
        # Prepare safe globals
        safe_globals = {
            '__builtins__': {
                'print': print,
                'len': len,
                'range': range,
                'str': str,
                'int': int,
                'float': float,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'abs': abs,
                'min': min,
                'max': max,
                'sum': sum,
                'round': round,
            }
        }
        
        # Add allowed modules
        try:
            import numpy as np
            safe_globals['np'] = np
            safe_globals['numpy'] = np
        except ImportError:
            pass
        
        try:
            import scipy as sp
            safe_globals['sp'] = sp
            safe_globals['scipy'] = sp
        except ImportError:
            pass
        
        import math
        safe_globals['math'] = math
        
        import json
        safe_globals['json'] = json
        
        # Execute code
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(code, safe_globals)
            
            output = stdout.getvalue()
            error = stderr.getvalue()
            
            return {
                "success": True,
                "output": output or "Code executed successfully",
                "error": error or None,
            }
            
        except Exception as e:
            return {
                "success": False,
                "output": stdout.getvalue(),
                "error": f"{type(e).__name__}: {str(e)}",
            }


@router.post("/execute")
async def execute_code(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute Python code
    
    Args:
        code: Python code to execute
    """
    code = request.get('code', '')
    
    if not code:
        raise HTTPException(status_code=400, detail="No code provided")
    
    try:
        result = ScriptExecutor.execute(code)
        return result
        
    except Exception as e:
        logger.error(f"Script execution error: {e}")
        return {
            "success": False,
            "output": None,
            "error": str(e),
        }


@router.get("/help")
async def get_help() -> Dict[str, Any]:
    """Get available functions and modules"""
    help_text = """Available functions:
  - process_data(data, function, params)
  - load_grid(filename)
  - save_grid(data, filename)
  - plot_map(data)
  
Available modules:
  - numpy as np
  - scipy as sp
  - math
  - json

Example:
  import numpy as np
  data = np.random.rand(10, 10)
  result = np.mean(data)
  print(f"Mean: {result}")
"""
    
    return {
        "success": True,
        "help": help_text,
    }


@router.get("/examples")
async def get_examples() -> Dict[str, Any]:
    """Get code examples"""
    examples = [
        {
            "name": "Create synthetic data",
            "code": """import numpy as np
x = np.linspace(0, 10, 100)
y = np.sin(x)
print(f"Max: {np.max(y):.3f}")
print(f"Min: {np.min(y):.3f}")""",
        },
        {
            "name": "Statistics",
            "code": """import numpy as np
data = np.random.normal(0, 1, 1000)
print(f"Mean: {np.mean(data):.3f}")
print(f"Std: {np.std(data):.3f}")""",
        },
        {
            "name": "FFT",
            "code": """import numpy as np
signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 100))
fft = np.fft.fft(signal)
print(f"FFT shape: {fft.shape}")
print(f"Peak frequency: {np.argmax(np.abs(fft))}")""",
        },
    ]
    
    return {
        "success": True,
        "examples": examples,
    }
