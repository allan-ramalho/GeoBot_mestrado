"""
Project Manager
Manages geophysical projects and their structure
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import uuid
import logging
from datetime import datetime

from app.core.config import settings

logger = logging.getLogger(__name__)


class ProjectManager:
    """
    Manages projects and their file structure
    """
    
    def __init__(self):
        self.projects_dir = settings.PROJECTS_DIR
        self.projects_dir.mkdir(parents=True, exist_ok=True)
    
    async def create_project(
        self,
        name: str,
        description: Optional[str] = None,
        project_type: str = "magnetic",
        location: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new project
        """
        project_id = uuid.uuid4().hex[:12]
        project_dir = self.projects_dir / project_id
        
        # Create project structure
        structure = {
            "raw_data": project_dir / "raw_data",
            "processed_data": project_dir / "processed_data",
            "interpretations": project_dir / "interpretations",
            "maps": project_dir / "maps",
            "exports": project_dir / "exports",
            "features": project_dir / "features"
        }
        
        for path in structure.values():
            path.mkdir(parents=True, exist_ok=True)
        
        # Project metadata
        metadata = {
            "id": project_id,
            "name": name,
            "description": description,
            "project_type": project_type,
            "location": location,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "structure": {k: str(v) for k, v in structure.items()}
        }
        
        # Save metadata
        metadata_file = project_dir / "project.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Project created: {name} ({project_id})")
        
        return metadata
    
    async def list_projects(self) -> List[Dict[str, Any]]:
        """
        List all projects
        """
        projects = []
        
        for project_dir in self.projects_dir.iterdir():
            if project_dir.is_dir():
                metadata_file = project_dir / "project.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    projects.append(metadata)
        
        return projects
    
    async def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """
        Get project metadata
        """
        project_dir = self.projects_dir / project_id
        metadata_file = project_dir / "project.json"
        
        if not metadata_file.exists():
            return None
        
        with open(metadata_file, 'r') as f:
            return json.load(f)
    
    async def update_project(
        self,
        project_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update project metadata
        """
        project = await self.get_project(project_id)
        
        if not project:
            raise ValueError(f"Project not found: {project_id}")
        
        # Update fields
        project.update(updates)
        project["updated_at"] = datetime.now().isoformat()
        
        # Save
        project_dir = self.projects_dir / project_id
        metadata_file = project_dir / "project.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(project, f, indent=2)
        
        logger.info(f"✅ Project updated: {project_id}")
        
        return project
    
    async def delete_project(self, project_id: str):
        """
        Delete a project
        """
        project_dir = self.projects_dir / project_id
        
        if project_dir.exists():
            import shutil
            shutil.rmtree(project_dir)
            logger.info(f"🗑️ Project deleted: {project_id}")
    
    async def get_project_tree(self, project_id: str) -> Dict[str, Any]:
        """
        Get project file tree
        """
        project_dir = self.projects_dir / project_id
        
        if not project_dir.exists():
            raise ValueError(f"Project not found: {project_id}")
        
        def build_tree(path: Path) -> Dict[str, Any]:
            """Recursively build tree"""
            if path.is_file():
                return {
                    "type": "file",
                    "name": path.name,
                    "size": path.stat().st_size
                }
            
            children = []
            for child in sorted(path.iterdir()):
                children.append(build_tree(child))
            
            return {
                "type": "directory",
                "name": path.name,
                "children": children
            }
        
        return build_tree(project_dir)
