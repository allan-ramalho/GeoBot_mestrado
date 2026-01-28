"""
Project Templates
Pre-configured workflows and sample datasets
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from datetime import datetime
import shutil

TEMPLATES_DIR = Path(__file__).parent / "templates"


class ProjectTemplate:
    """Base class for project templates"""
    
    def __init__(self, id: str, name: str, description: str, category: str):
        self.id = id
        self.name = name
        self.description = description
        self.category = category
    
    def create_project(self, output_dir: Path) -> Dict[str, Any]:
        """
        Create project from template
        
        Returns:
            Project metadata
        """
        raise NotImplementedError
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'category': self.category,
        }


class MagneticAnomalyTemplate(ProjectTemplate):
    """Magnetic anomaly analysis template"""
    
    def __init__(self):
        super().__init__(
            id='magnetic_anomaly',
            name='Magnetic Anomaly Analysis',
            description='Complete magnetic data processing workflow',
            category='geophysics'
        )
    
    def create_project(self, output_dir: Path) -> Dict[str, Any]:
        """Create magnetic anomaly project"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate synthetic magnetic data
        nx, ny = 200, 200
        x = np.linspace(0, 10000, nx)
        y = np.linspace(0, 10000, ny)
        X, Y = np.meshgrid(x, y)
        
        # Create anomaly (prism)
        cx, cy = 5000, 5000
        R = np.sqrt((X - cx)**2 + (Y - cy)**2)
        Z = 100 * np.exp(-R**2 / 1000000)
        
        # Add noise
        Z += np.random.normal(0, 5, Z.shape)
        
        # Save data
        data_file = output_dir / "magnetic_data.npz"
        np.savez(data_file, x=x, y=y, z=Z)
        
        # Create workflow
        workflow = {
            'name': 'Magnetic Enhancement',
            'description': 'Reduction to pole, upward continuation, and derivatives',
            'steps': [
                {
                    'function': 'reduction_to_pole',
                    'params': {'inclination': -30, 'declination': 0},
                },
                {
                    'function': 'upward_continuation',
                    'params': {'altitude': 100},
                },
                {
                    'function': 'total_horizontal_derivative',
                    'params': {},
                },
            ],
        }
        
        workflow_file = output_dir / "workflow.json"
        workflow_file.write_text(json.dumps(workflow, indent=2))
        
        # Create README
        readme = """# Magnetic Anomaly Analysis

## Dataset
- **Type**: Synthetic magnetic anomaly
- **Size**: 200 x 200 points
- **Spacing**: 50m
- **Feature**: Central magnetic prism

## Workflow
1. **Reduction to Pole**: Remove effect of magnetic inclination
2. **Upward Continuation**: Suppress short-wavelength noise
3. **Total Horizontal Derivative**: Edge detection

## Next Steps
1. Load data: `magnetic_data.npz`
2. Run workflow: `workflow.json`
3. Visualize results
4. Interpret anomalies
"""
        
        (output_dir / "README.md").write_text(readme)
        
        return {
            'id': f"magnetic_template_{datetime.now().timestamp()}",
            'name': 'Magnetic Anomaly Analysis',
            'description': self.description,
            'data_files': [str(data_file)],
            'workflow_file': str(workflow_file),
            'created_at': datetime.now().isoformat(),
        }


class GravityReductionTemplate(ProjectTemplate):
    """Gravity reduction workflow template"""
    
    def __init__(self):
        super().__init__(
            id='gravity_reduction',
            name='Gravity Data Reduction',
            description='Complete Bouguer gravity reduction',
            category='geophysics'
        )
    
    def create_project(self, output_dir: Path) -> Dict[str, Any]:
        """Create gravity reduction project"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate synthetic gravity data
        nx, ny = 150, 150
        x = np.linspace(0, 15000, nx)
        y = np.linspace(0, 15000, ny)
        X, Y = np.meshgrid(x, y)
        
        # Regional trend + local anomaly
        regional = 0.0001 * X + 0.00005 * Y + 50
        
        # Gravity anomaly (density contrast)
        cx, cy = 7500, 7500
        R = np.sqrt((X - cx)**2 + (Y - cy)**2)
        anomaly = 30 * np.exp(-R**2 / 5000000)
        
        Z = regional + anomaly + np.random.normal(0, 1, X.shape)
        
        # Create elevation data
        elevation = 100 + 50 * np.sin(X / 2000) + np.random.normal(0, 5, X.shape)
        
        # Save data
        data_file = output_dir / "gravity_data.npz"
        np.savez(data_file, x=x, y=y, z=Z, elevation=elevation)
        
        # Create workflow
        workflow = {
            'name': 'Bouguer Reduction',
            'description': 'Complete gravity reduction workflow',
            'steps': [
                {
                    'function': 'free_air_correction',
                    'params': {'use_elevation': True},
                },
                {
                    'function': 'bouguer_correction',
                    'params': {'density': 2.67},
                },
                {
                    'function': 'terrain_correction',
                    'params': {'radius': 2000},
                },
                {
                    'function': 'regional_residual_separation',
                    'params': {'method': 'polynomial', 'order': 2},
                },
            ],
        }
        
        workflow_file = output_dir / "workflow.json"
        workflow_file.write_text(json.dumps(workflow, indent=2))
        
        # Create README
        readme = """# Gravity Data Reduction

## Dataset
- **Type**: Synthetic Bouguer gravity
- **Size**: 150 x 150 points
- **Spacing**: 100m
- **Features**: Regional trend + local anomaly

## Workflow
1. **Free-Air Correction**: Elevation effect
2. **Bouguer Correction**: Topography mass
3. **Terrain Correction**: Local topography
4. **Regional-Residual Separation**: Isolate local anomalies

## Parameters
- Density: 2.67 g/cm³ (typical crust)
- Terrain radius: 2 km

## Expected Results
- Bouguer anomaly: ~30 mGal
- Regional gradient removed
- Local anomaly enhanced
"""
        
        (output_dir / "README.md").write_text(readme)
        
        return {
            'id': f"gravity_template_{datetime.now().timestamp()}",
            'name': 'Gravity Data Reduction',
            'description': self.description,
            'data_files': [str(data_file)],
            'workflow_file': str(workflow_file),
            'created_at': datetime.now().isoformat(),
        }


class FilteringTutorialTemplate(ProjectTemplate):
    """Filtering techniques tutorial"""
    
    def __init__(self):
        super().__init__(
            id='filtering_tutorial',
            name='Filtering Techniques',
            description='Learn different filtering methods',
            category='tutorial'
        )
    
    def create_project(self, output_dir: Path) -> Dict[str, Any]:
        """Create filtering tutorial project"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate noisy data
        nx, ny = 100, 100
        x = np.linspace(0, 5000, nx)
        y = np.linspace(0, 5000, ny)
        X, Y = np.meshgrid(x, y)
        
        # Signal: multiple anomalies
        signal = (
            50 * np.sin(X / 500) * np.cos(Y / 500) +
            30 * np.exp(-((X - 2500)**2 + (Y - 2500)**2) / 500000)
        )
        
        # Add noise
        noise = np.random.normal(0, 10, signal.shape)
        Z = signal + noise
        
        # Save data
        data_file = output_dir / "noisy_data.npz"
        np.savez(data_file, x=x, y=y, z=Z, signal=signal, noise=noise)
        
        # Create multiple workflows
        workflows = [
            {
                'name': 'Low-pass Filter',
                'steps': [{'function': 'gaussian_filter', 'params': {'sigma': 2.0}}],
            },
            {
                'name': 'High-pass Filter',
                'steps': [{'function': 'butterworth_filter', 'params': {'cutoff': 0.1, 'order': 2, 'filter_type': 'high'}}],
            },
            {
                'name': 'Median Filter',
                'steps': [{'function': 'median_filter', 'params': {'size': 5}}],
            },
        ]
        
        for i, wf in enumerate(workflows):
            wf_file = output_dir / f"workflow_{i+1}.json"
            wf_file.write_text(json.dumps(wf, indent=2))
        
        # Create README
        readme = """# Filtering Techniques Tutorial

## Dataset
Synthetic data with:
- Signal: sinusoidal pattern + Gaussian anomaly
- Noise: Gaussian white noise (SNR ~5)

## Workflows

### 1. Low-pass Filter (Gaussian)
- **Purpose**: Remove high-frequency noise
- **Result**: Smooth signal, preserves broad features
- **Best for**: Noisy data, regional trends

### 2. High-pass Filter (Butterworth)
- **Purpose**: Enhance local anomalies
- **Result**: Removes regional trend, highlights edges
- **Best for**: Residual separation, edge detection

### 3. Median Filter
- **Purpose**: Remove outliers
- **Result**: Preserves edges, removes spikes
- **Best for**: Spike noise, non-Gaussian noise

## Exercise
1. Load `noisy_data.npz`
2. Apply each filter
3. Compare results
4. Which filter works best for this data?
5. Try adjusting parameters
"""
        
        (output_dir / "README.md").write_text(readme)
        
        return {
            'id': f"filtering_template_{datetime.now().timestamp()}",
            'name': 'Filtering Techniques',
            'description': self.description,
            'data_files': [str(data_file)],
            'workflow_files': [str(output_dir / f"workflow_{i+1}.json") for i in range(3)],
            'created_at': datetime.now().isoformat(),
        }


class TemplateManager:
    """Manage project templates"""
    
    def __init__(self):
        self.templates: Dict[str, ProjectTemplate] = {
            'magnetic_anomaly': MagneticAnomalyTemplate(),
            'gravity_reduction': GravityReductionTemplate(),
            'filtering_tutorial': FilteringTutorialTemplate(),
        }
    
    def list_templates(self) -> List[Dict]:
        """List all available templates"""
        return [t.to_dict() for t in self.templates.values()]
    
    def get_template(self, template_id: str) -> ProjectTemplate:
        """Get template by ID"""
        return self.templates.get(template_id)
    
    def create_from_template(self, template_id: str, output_dir: Path) -> Dict:
        """Create project from template"""
        template = self.get_template(template_id)
        
        if not template:
            raise ValueError(f"Template not found: {template_id}")
        
        return template.create_project(output_dir)
    
    def add_custom_template(self, template: ProjectTemplate):
        """Add custom template"""
        self.templates[template.id] = template


# Global template manager
template_manager = TemplateManager()
