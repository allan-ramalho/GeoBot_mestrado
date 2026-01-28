"""
Workflow System for Geophysical Data Processing
Create and execute complex processing workflows with dependencies
"""

import numpy as np
from typing import Dict, Any, List, Optional, Callable, Set
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from datetime import datetime
import networkx as nx

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Status of a workflow step"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """Represents a single step in a workflow"""
    step_id: str
    function_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'step_id': self.step_id,
            'function_name': self.function_name,
            'parameters': self.parameters,
            'depends_on': self.depends_on,
            'status': self.status.value,
            'error': self.error,
            'execution_time': self.execution_time
        }


class Workflow:
    """
    Workflow for geophysical data processing
    
    Features:
    - Dependency management
    - Step ordering with topological sort
    - Intermediate result caching
    - Error handling and recovery
    - Workflow serialization
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize workflow
        
        Args:
            name: Workflow name
            description: Workflow description
        """
        self.name = name
        self.description = description
        self.steps: Dict[str, WorkflowStep] = {}
        self.execution_order: List[str] = []
        self.results_cache: Dict[str, Dict[str, Any]] = {}
        logger.info(f"🔧 Workflow created: {name}")
    
    def add_step(
        self,
        step_id: str,
        function_name: str,
        parameters: Dict[str, Any] = None,
        depends_on: List[str] = None
    ) -> WorkflowStep:
        """
        Add a step to the workflow
        
        Args:
            step_id: Unique step identifier
            function_name: Name of processing function
            parameters: Function parameters
            depends_on: List of step IDs this step depends on
            
        Returns:
            Created WorkflowStep
        """
        if step_id in self.steps:
            raise ValueError(f"Step '{step_id}' already exists")
        
        # Validate dependencies
        if depends_on:
            for dep in depends_on:
                if dep not in self.steps:
                    raise ValueError(f"Dependency '{dep}' not found for step '{step_id}'")
        
        step = WorkflowStep(
            step_id=step_id,
            function_name=function_name,
            parameters=parameters or {},
            depends_on=depends_on or []
        )
        
        self.steps[step_id] = step
        self._update_execution_order()
        
        logger.info(f"➕ Step added: {step_id} ({function_name})")
        if depends_on:
            logger.info(f"   Dependencies: {', '.join(depends_on)}")
        
        return step
    
    def _update_execution_order(self):
        """Update execution order using topological sort"""
        # Build dependency graph
        graph = nx.DiGraph()
        
        for step_id, step in self.steps.items():
            graph.add_node(step_id)
            for dep in step.depends_on:
                graph.add_edge(dep, step_id)
        
        # Check for cycles
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Workflow contains circular dependencies")
        
        # Topological sort
        self.execution_order = list(nx.topological_sort(graph))
        logger.info(f"📋 Execution order: {' → '.join(self.execution_order)}")
    
    def remove_step(self, step_id: str):
        """
        Remove a step from the workflow
        
        Args:
            step_id: Step to remove
        """
        if step_id not in self.steps:
            raise ValueError(f"Step '{step_id}' not found")
        
        # Check if other steps depend on this
        dependents = [s for s in self.steps.values() if step_id in s.depends_on]
        if dependents:
            dep_names = [s.step_id for s in dependents]
            raise ValueError(f"Cannot remove '{step_id}': steps depend on it: {dep_names}")
        
        del self.steps[step_id]
        self._update_execution_order()
        logger.info(f"➖ Step removed: {step_id}")
    
    def execute(
        self,
        input_data: Dict[str, Any],
        function_registry: Dict[str, Callable],
        skip_on_error: bool = False,
        cache_results: bool = True
    ) -> Dict[str, Any]:
        """
        Execute the workflow
        
        Args:
            input_data: Initial input data
            function_registry: Dictionary of available functions
            skip_on_error: Continue execution if a step fails
            cache_results: Cache intermediate results
            
        Returns:
            Final workflow result
        """
        logger.info(f"🚀 Executing workflow: {self.name}")
        logger.info(f"   Steps: {len(self.steps)}, Order: {' → '.join(self.execution_order)}")
        
        current_data = input_data
        failed_steps = []
        
        for step_id in self.execution_order:
            step = self.steps[step_id]
            
            # Check if dependencies failed
            failed_deps = [dep for dep in step.depends_on if self.steps[dep].status == StepStatus.FAILED]
            if failed_deps and not skip_on_error:
                step.status = StepStatus.SKIPPED
                logger.warning(f"⏭️ Skipping step '{step_id}': dependencies failed: {failed_deps}")
                continue
            
            # Get function
            func = function_registry.get(step.function_name)
            if not func:
                step.status = StepStatus.FAILED
                step.error = f"Function '{step.function_name}' not found"
                logger.error(f"❌ {step.error}")
                
                if not skip_on_error:
                    raise ValueError(step.error)
                failed_steps.append(step_id)
                continue
            
            # Execute step
            step.status = StepStatus.RUNNING
            logger.info(f"▶️ Executing step: {step_id} ({step.function_name})")
            
            import time
            start_time = time.time()
            
            try:
                # Execute function
                result = func(current_data, **step.parameters)
                
                # Store result
                step.result = result
                step.status = StepStatus.COMPLETED
                step.execution_time = time.time() - start_time
                
                # Cache if enabled
                if cache_results:
                    self.results_cache[step_id] = result
                
                # Update current data for next step
                current_data = result
                
                logger.info(f"✅ Step completed: {step_id} ({step.execution_time:.2f}s)")
            
            except Exception as e:
                step.status = StepStatus.FAILED
                step.error = str(e)
                step.execution_time = time.time() - start_time
                
                logger.error(f"❌ Step failed: {step_id} - {e}")
                failed_steps.append(step_id)
                
                if not skip_on_error:
                    raise
        
        # Summary
        completed = sum(1 for s in self.steps.values() if s.status == StepStatus.COMPLETED)
        failed = len(failed_steps)
        skipped = sum(1 for s in self.steps.values() if s.status == StepStatus.SKIPPED)
        
        logger.info(f"✅ Workflow completed: {completed} completed, {failed} failed, {skipped} skipped")
        
        if failed_steps:
            logger.warning(f"⚠️ Failed steps: {', '.join(failed_steps)}")
        
        return current_data
    
    def get_step_result(self, step_id: str) -> Optional[Dict[str, Any]]:
        """
        Get cached result from a specific step
        
        Args:
            step_id: Step identifier
            
        Returns:
            Step result or None
        """
        return self.results_cache.get(step_id)
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary"""
        status_counts = {}
        for status in StepStatus:
            status_counts[status.value] = sum(
                1 for s in self.steps.values() if s.status == status
            )
        
        total_time = sum(
            s.execution_time for s in self.steps.values() 
            if s.execution_time is not None
        )
        
        return {
            'workflow_name': self.name,
            'total_steps': len(self.steps),
            'status_counts': status_counts,
            'total_execution_time': total_time,
            'steps': [step.to_dict() for step in self.steps.values()]
        }
    
    def reset(self):
        """Reset all step statuses and clear cache"""
        for step in self.steps.values():
            step.status = StepStatus.PENDING
            step.result = None
            step.error = None
            step.execution_time = None
        
        self.results_cache.clear()
        logger.info("🔄 Workflow reset")
    
    def save(self, output_path: str):
        """
        Save workflow definition to JSON
        
        Args:
            output_path: Path to output file
        """
        workflow_def = {
            'name': self.name,
            'description': self.description,
            'steps': [step.to_dict() for step in self.steps.values()],
            'execution_order': self.execution_order
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(workflow_def, f, indent=2)
        
        logger.info(f"💾 Workflow saved: {output_path}")
    
    @classmethod
    def load(cls, input_path: str) -> 'Workflow':
        """
        Load workflow from JSON file
        
        Args:
            input_path: Path to workflow JSON
            
        Returns:
            Loaded Workflow object
        """
        with open(input_path, 'r') as f:
            workflow_def = json.load(f)
        
        workflow = cls(
            name=workflow_def['name'],
            description=workflow_def.get('description', '')
        )
        
        # Add steps
        for step_dict in workflow_def['steps']:
            workflow.add_step(
                step_id=step_dict['step_id'],
                function_name=step_dict['function_name'],
                parameters=step_dict.get('parameters', {}),
                depends_on=step_dict.get('depends_on', [])
            )
        
        logger.info(f"📂 Workflow loaded: {input_path}")
        return workflow


class WorkflowBuilder:
    """
    Builder for creating common geophysical processing workflows
    """
    
    @staticmethod
    def create_magnetic_enhancement_workflow(name: str = "magnetic_enhancement") -> Workflow:
        """
        Create standard magnetic data enhancement workflow
        
        Steps:
        1. Reduction to pole
        2. Upward continuation
        3. Total horizontal derivative
        4. Tilt derivative
        
        Returns:
            Configured Workflow
        """
        workflow = Workflow(
            name=name,
            description="Standard magnetic enhancement: RTP → UC → THD → Tilt"
        )
        
        workflow.add_step(
            step_id="rtp",
            function_name="reduction_to_pole",
            parameters={'inclination': -30.0, 'declination': 0.0}
        )
        
        workflow.add_step(
            step_id="upward_continuation",
            function_name="upward_continuation",
            parameters={'height': 500.0},
            depends_on=["rtp"]
        )
        
        workflow.add_step(
            step_id="thd",
            function_name="total_horizontal_derivative",
            depends_on=["upward_continuation"]
        )
        
        workflow.add_step(
            step_id="tilt",
            function_name="tilt_derivative",
            depends_on=["upward_continuation"]
        )
        
        logger.info("✅ Magnetic enhancement workflow created")
        return workflow
    
    @staticmethod
    def create_gravity_reduction_workflow(name: str = "gravity_reduction") -> Workflow:
        """
        Create gravity data reduction workflow
        
        Steps:
        1. Free-air correction
        2. Bouguer correction
        3. Terrain correction
        4. Regional-residual separation
        
        Returns:
            Configured Workflow
        """
        workflow = Workflow(
            name=name,
            description="Gravity reduction: FA → Bouguer → Terrain → Regional/Residual"
        )
        
        workflow.add_step(
            step_id="free_air",
            function_name="free_air_correction",
            parameters={'reference_elevation': 0.0}
        )
        
        workflow.add_step(
            step_id="bouguer",
            function_name="bouguer_correction",
            parameters={'density': 2.67},
            depends_on=["free_air"]
        )
        
        workflow.add_step(
            step_id="terrain",
            function_name="terrain_correction",
            depends_on=["bouguer"]
        )
        
        workflow.add_step(
            step_id="regional_residual",
            function_name="regional_residual_separation",
            parameters={'method': 'polynomial', 'order': 2},
            depends_on=["terrain"]
        )
        
        logger.info("✅ Gravity reduction workflow created")
        return workflow
    
    @staticmethod
    def create_depth_estimation_workflow(name: str = "depth_estimation") -> Workflow:
        """
        Create depth estimation workflow
        
        Steps:
        1. Calculate analytic signal
        2. Euler deconvolution
        3. Tilt-depth method
        4. Source parameter imaging
        
        Returns:
            Configured Workflow
        """
        workflow = Workflow(
            name=name,
            description="Multi-method depth estimation"
        )
        
        workflow.add_step(
            step_id="analytic_signal",
            function_name="analytic_signal"
        )
        
        workflow.add_step(
            step_id="euler",
            function_name="euler_deconvolution",
            parameters={'structural_index': 1.0, 'window_size': 10},
            depends_on=["analytic_signal"]
        )
        
        workflow.add_step(
            step_id="tilt_depth",
            function_name="tilt_depth_method",
            depends_on=["analytic_signal"]
        )
        
        workflow.add_step(
            step_id="spi",
            function_name="source_parameter_imaging",
            depends_on=["analytic_signal"]
        )
        
        logger.info("✅ Depth estimation workflow created")
        return workflow
    
    @staticmethod
    def create_filtering_workflow(name: str = "data_filtering") -> Workflow:
        """
        Create data filtering workflow
        
        Steps:
        1. Median filter (remove spikes)
        2. Gaussian smoothing
        3. Directional filter
        
        Returns:
            Configured Workflow
        """
        workflow = Workflow(
            name=name,
            description="Multi-stage filtering: Median → Gaussian → Directional"
        )
        
        workflow.add_step(
            step_id="median",
            function_name="median_filter",
            parameters={'size': 3, 'threshold': 3.0}
        )
        
        workflow.add_step(
            step_id="gaussian",
            function_name="gaussian_filter",
            parameters={'sigma': 2.0},
            depends_on=["median"]
        )
        
        workflow.add_step(
            step_id="directional",
            function_name="directional_filter",
            parameters={'azimuth': 45.0, 'width': 30.0},
            depends_on=["gaussian"]
        )
        
        logger.info("✅ Filtering workflow created")
        return workflow


class WorkflowLibrary:
    """
    Library for managing multiple workflows
    """
    
    def __init__(self, library_path: Optional[str] = None):
        """
        Initialize workflow library
        
        Args:
            library_path: Path to workflow library directory
        """
        self.workflows: Dict[str, Workflow] = {}
        self.library_path = Path(library_path) if library_path else None
        
        if self.library_path:
            self.library_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("📚 Workflow library initialized")
    
    def add_workflow(self, workflow: Workflow):
        """Add workflow to library"""
        self.workflows[workflow.name] = workflow
        logger.info(f"➕ Workflow added to library: {workflow.name}")
    
    def get_workflow(self, name: str) -> Optional[Workflow]:
        """Get workflow by name"""
        return self.workflows.get(name)
    
    def list_workflows(self) -> List[str]:
        """List all workflow names"""
        return list(self.workflows.keys())
    
    def save_all(self):
        """Save all workflows to library path"""
        if not self.library_path:
            raise ValueError("Library path not set")
        
        for workflow in self.workflows.values():
            output_path = self.library_path / f"{workflow.name}.json"
            workflow.save(str(output_path))
        
        logger.info(f"💾 All workflows saved to: {self.library_path}")
    
    def load_all(self):
        """Load all workflows from library path"""
        if not self.library_path:
            raise ValueError("Library path not set")
        
        for workflow_file in self.library_path.glob("*.json"):
            workflow = Workflow.load(str(workflow_file))
            self.workflows[workflow.name] = workflow
        
        logger.info(f"📂 Loaded {len(self.workflows)} workflows from library")


# Initialize
logger.info("🔀 Workflow system initialized")
