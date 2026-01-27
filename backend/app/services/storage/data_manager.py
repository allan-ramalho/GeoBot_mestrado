"""
Data Manager
Handles file uploads, storage, and data management
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import shutil
import uuid
import logging
from datetime import datetime

from fastapi import UploadFile
import pandas as pd
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class DataManager:
    """
    Manages geophysical data files and storage
    """
    
    def __init__(self):
        self.data_dir = settings.DATA_DIR
    
    async def save_file(
        self,
        file: UploadFile,
        project_id: Optional[str] = None,
        data_type: str = "raw"
    ) -> Path:
        """
        Save uploaded file
        
        Args:
            file: Uploaded file
            project_id: Project ID (optional)
            data_type: Type of data (raw, processed, interpretation)
        
        Returns:
            Path to saved file
        """
        # Create directory structure
        if project_id:
            file_dir = self.data_dir / project_id / data_type
        else:
            file_dir = self.data_dir / "uploads" / data_type
        
        file_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        file_id = uuid.uuid4().hex[:8]
        file_ext = Path(file.filename).suffix
        file_path = file_dir / f"{file_id}_{file.filename}"
        
        # Save file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"💾 File saved: {file_path}")
        
        return file_path
    
    async def parse_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse file metadata
        """
        stat = file_path.stat()
        
        return {
            "filename": file_path.name,
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "extension": file_path.suffix
        }
    
    async def list_project_files(self, project_id: str) -> List[Dict[str, Any]]:
        """
        List all files in a project
        """
        project_dir = self.data_dir / project_id
        
        if not project_dir.exists():
            return []
        
        files = []
        for file_path in project_dir.rglob("*"):
            if file_path.is_file():
                metadata = await self.parse_file_metadata(file_path)
                files.append(metadata)
        
        return files
    
    async def get_file_path(self, file_id: str) -> Path:
        """
        Get file path by ID
        """
        # Search for file
        for file_path in self.data_dir.rglob("*"):
            if file_id in file_path.name:
                return file_path
        
        raise FileNotFoundError(f"File not found: {file_id}")
    
    async def delete_file(self, file_id: str):
        """
        Delete a file
        """
        file_path = await self.get_file_path(file_id)
        file_path.unlink()
        logger.info(f"🗑️ File deleted: {file_path}")
    
    async def import_data(
        self,
        file: UploadFile,
        format: str,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Import and parse geophysical data
        """
        # Save file first
        file_path = await self.save_file(file, project_id, "raw")
        
        # Parse based on format
        if format == "xyz":
            data = await self._parse_xyz(file_path)
        elif format == "csv":
            data = await self._parse_csv(file_path)
        elif format == "grd":
            data = await self._parse_grd(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        data_id = uuid.uuid4().hex[:8]
        
        return {
            "id": data_id,
            "file_path": str(file_path),
            "format": format,
            "summary": await self._summarize_data(data)
        }
    
    async def _parse_xyz(self, file_path: Path) -> Dict[str, Any]:
        """Parse XYZ format"""
        df = pd.read_csv(file_path, delim_whitespace=True, header=None, names=['x', 'y', 'z'])
        
        return {
            "x": df['x'].values,
            "y": df['y'].values,
            "z": df['z'].values,
            "format": "xyz"
        }
    
    async def _parse_csv(self, file_path: Path) -> Dict[str, Any]:
        """Parse CSV format"""
        df = pd.read_csv(file_path)
        
        return {
            "data": df,
            "columns": df.columns.tolist(),
            "format": "csv"
        }
    
    async def _parse_grd(self, file_path: Path) -> Dict[str, Any]:
        """Parse GRD format"""
        # TODO: Implement GRD parsing
        logger.warning("GRD parsing not yet implemented")
        return {"format": "grd"}
    
    async def _summarize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create data summary"""
        if data["format"] == "xyz":
            return {
                "points": len(data["x"]),
                "x_range": [float(np.min(data["x"])), float(np.max(data["x"]))],
                "y_range": [float(np.min(data["y"])), float(np.max(data["y"]))],
                "z_range": [float(np.min(data["z"])), float(np.max(data["z"]))]
            }
        
        return {"format": data["format"]}
