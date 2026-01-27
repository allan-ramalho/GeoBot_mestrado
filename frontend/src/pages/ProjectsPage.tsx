/**
 * Projects Page - Complete Project Management
 * Features: tree view, CRUD, tags, metadata, import/export
 */

import React, { useState, useMemo } from 'react';
import {
  Folder,
  FolderOpen,
  File,
  ChevronRight,
  ChevronDown,
  Plus,
  Search,
  Filter,
  Download,
  Upload,
  Trash2,
  Edit,
  Tag,
  Calendar,
  User,
  FileText,
  Database,
  MoreVertical,
} from 'lucide-react';

interface ProjectFile {
  id: string;
  name: string;
  type: 'file' | 'folder';
  size?: number;
  dataType?: 'magnetic' | 'gravity' | 'processed';
  createdAt: Date;
  modifiedAt: Date;
  tags: string[];
  metadata: Record<string, any>;
  children?: ProjectFile[];
}

interface Project {
  id: string;
  name: string;
  description: string;
  createdAt: Date;
  modifiedAt: Date;
  author: string;
  tags: string[];
  files: ProjectFile[];
  metadata: Record<string, any>;
}

// Mock projects data
const MOCK_PROJECTS: Project[] = [
  {
    id: 'proj_001',
    name: 'Serra do Carajás Survey',
    description: 'Magnetic and gravity survey of Carajás region',
    createdAt: new Date('2025-12-01'),
    modifiedAt: new Date('2026-01-15'),
    author: 'Allan Ramalho',
    tags: ['magnetic', 'gravity', 'iron-ore', 'brazil'],
    metadata: {
      location: 'Pará, Brazil',
      surveyDate: '2025',
      grid: '1000x1000',
    },
    files: [
      {
        id: 'file_001',
        name: 'Raw Data',
        type: 'folder',
        createdAt: new Date('2025-12-01'),
        modifiedAt: new Date('2025-12-15'),
        tags: ['raw'],
        metadata: {},
        children: [
          {
            id: 'file_002',
            name: 'magnetic_survey.xyz',
            type: 'file',
            size: 2048576,
            dataType: 'magnetic',
            createdAt: new Date('2025-12-01'),
            modifiedAt: new Date('2025-12-01'),
            tags: ['raw', 'magnetic'],
            metadata: { rows: 1000, cols: 1000, unit: 'nT' },
          },
          {
            id: 'file_003',
            name: 'gravity_survey.xyz',
            type: 'file',
            size: 1536000,
            dataType: 'gravity',
            createdAt: new Date('2025-12-02'),
            modifiedAt: new Date('2025-12-02'),
            tags: ['raw', 'gravity'],
            metadata: { rows: 800, cols: 800, unit: 'mGal' },
          },
        ],
      },
      {
        id: 'file_004',
        name: 'Processed',
        type: 'folder',
        createdAt: new Date('2026-01-10'),
        modifiedAt: new Date('2026-01-15'),
        tags: ['processed'],
        metadata: {},
        children: [
          {
            id: 'file_005',
            name: 'rtp_result.xyz',
            type: 'file',
            size: 2048576,
            dataType: 'processed',
            createdAt: new Date('2026-01-10'),
            modifiedAt: new Date('2026-01-10'),
            tags: ['rtp', 'processed'],
            metadata: { function: 'reduction_to_pole', inclination: -30, declination: 0 },
          },
        ],
      },
    ],
  },
  {
    id: 'proj_002',
    name: 'Amazon Basin Study',
    description: 'Regional gravity study of Amazon sedimentary basin',
    createdAt: new Date('2026-01-05'),
    modifiedAt: new Date('2026-01-20'),
    author: 'Allan Ramalho',
    tags: ['gravity', 'basin', 'regional'],
    metadata: {
      location: 'Amazon Basin, Brazil',
      area: '500000 km²',
    },
    files: [],
  },
];

const ALL_TAGS = ['magnetic', 'gravity', 'processed', 'raw', 'rtp', 'iron-ore', 'brazil', 'basin', 'regional'];

export default function ProjectsPage() {
  const [projects, setProjects] = useState<Project[]>(MOCK_PROJECTS);
  const [selectedProject, setSelectedProject] = useState<Project | null>(MOCK_PROJECTS[0]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set(['file_001', 'file_004']));
  const [selectedFile, setSelectedFile] = useState<ProjectFile | null>(null);
  const [showNewProjectDialog, setShowNewProjectDialog] = useState(false);

  // Filter projects
  const filteredProjects = useMemo(() => {
    return projects.filter(project => {
      const matchesSearch = !searchTerm ||
        project.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        project.description.toLowerCase().includes(searchTerm.toLowerCase());

      const matchesTags = selectedTags.length === 0 ||
        selectedTags.every(tag => project.tags.includes(tag));

      return matchesSearch && matchesTags;
    });
  }, [projects, searchTerm, selectedTags]);

  // Toggle folder expansion
  const toggleFolder = (fileId: string) => {
    setExpandedFolders(prev => {
      const next = new Set(prev);
      if (next.has(fileId)) {
        next.delete(fileId);
      } else {
        next.add(fileId);
      }
      return next;
    });
  };

  // Toggle tag filter
  const toggleTag = (tag: string) => {
    setSelectedTags(prev =>
      prev.includes(tag) ? prev.filter(t => t !== tag) : [...prev, tag]
    );
  };

  // Render file tree recursively
  const renderFileTree = (files: ProjectFile[], level = 0) => {
    return files.map(file => (
      <div key={file.id} style={{ marginLeft: `${level * 20}px` }}>
        <button
          onClick={() => {
            if (file.type === 'folder') {
              toggleFolder(file.id);
            } else {
              setSelectedFile(file);
            }
          }}
          className={`w-full flex items-center gap-2 px-3 py-2 hover:bg-gray-100 rounded transition ${
            selectedFile?.id === file.id ? 'bg-blue-50 border-l-4 border-blue-500' : ''
          }`}
        >
          {file.type === 'folder' ? (
            expandedFolders.has(file.id) ? (
              <>
                <ChevronDown className="w-4 h-4" />
                <FolderOpen className="w-5 h-5 text-yellow-500" />
              </>
            ) : (
              <>
                <ChevronRight className="w-4 h-4" />
                <Folder className="w-5 h-5 text-yellow-500" />
              </>
            )
          ) : (
            <>
              <File className="w-5 h-5 text-blue-500 ml-5" />
            </>
          )}
          <span className="text-sm flex-1 text-left">{file.name}</span>
          {file.type === 'file' && file.size && (
            <span className="text-xs text-gray-500">
              {(file.size / 1024 / 1024).toFixed(2)} MB
            </span>
          )}
        </button>

        {file.type === 'folder' && expandedFolders.has(file.id) && file.children && (
          <div className="mt-1">
            {renderFileTree(file.children, level + 1)}
          </div>
        )}
      </div>
    ));
  };

  // Format date
  const formatDate = (date: Date) => {
    return new Intl.DateTimeFormat('pt-BR', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    }).format(date);
  };

  return (
    <div className="h-full flex gap-4 p-4">
      {/* Projects List */}
      <div className="w-80 bg-white rounded-lg shadow overflow-hidden flex flex-col">
        {/* Header */}
        <div className="p-4 border-b">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold">Projects</h2>
            <button
              onClick={() => setShowNewProjectDialog(true)}
              className="p-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition"
              title="New Project"
            >
              <Plus className="w-5 h-5" />
            </button>
          </div>

          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search projects..."
              className="w-full pl-10 pr-4 py-2 border rounded text-sm"
            />
          </div>

          {/* Tag filters */}
          <div className="mt-3 flex flex-wrap gap-2">
            {ALL_TAGS.map(tag => (
              <button
                key={tag}
                onClick={() => toggleTag(tag)}
                className={`px-2 py-1 text-xs rounded transition ${
                  selectedTags.includes(tag)
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 hover:bg-gray-200'
                }`}
              >
                {tag}
              </button>
            ))}
          </div>
        </div>

        {/* Project Cards */}
        <div className="flex-1 overflow-y-auto">
          {filteredProjects.map(project => (
            <button
              key={project.id}
              onClick={() => setSelectedProject(project)}
              className={`w-full p-4 border-b text-left hover:bg-gray-50 transition ${
                selectedProject?.id === project.id ? 'bg-blue-50 border-l-4 border-blue-500' : ''
              }`}
            >
              <div className="font-medium mb-1">{project.name}</div>
              <div className="text-xs text-gray-600 mb-2">{project.description}</div>
              <div className="flex flex-wrap gap-1 mb-2">
                {project.tags.slice(0, 3).map(tag => (
                  <span key={tag} className="px-2 py-0.5 bg-gray-100 text-xs rounded">
                    {tag}
                  </span>
                ))}
                {project.tags.length > 3 && (
                  <span className="px-2 py-0.5 bg-gray-100 text-xs rounded">
                    +{project.tags.length - 3}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-4 text-xs text-gray-500">
                <span className="flex items-center gap-1">
                  <Calendar className="w-3 h-3" />
                  {formatDate(project.modifiedAt)}
                </span>
                <span className="flex items-center gap-1">
                  <Database className="w-3 h-3" />
                  {project.files.length} items
                </span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* File Tree */}
      {selectedProject && (
        <div className="w-96 bg-white rounded-lg shadow overflow-hidden flex flex-col">
          {/* Header */}
          <div className="p-4 border-b">
            <h3 className="font-semibold mb-1">{selectedProject.name}</h3>
            <p className="text-sm text-gray-600">{selectedProject.description}</p>
            
            {/* Actions */}
            <div className="flex gap-2 mt-3">
              <button className="px-3 py-1.5 text-sm border rounded hover:bg-gray-50 transition flex items-center gap-1">
                <Upload className="w-4 h-4" />
                Import
              </button>
              <button className="px-3 py-1.5 text-sm border rounded hover:bg-gray-50 transition flex items-center gap-1">
                <Download className="w-4 h-4" />
                Export
              </button>
              <button className="px-3 py-1.5 text-sm border rounded hover:bg-gray-50 transition flex items-center gap-1">
                <Plus className="w-4 h-4" />
                Add File
              </button>
            </div>
          </div>

          {/* Tree */}
          <div className="flex-1 overflow-y-auto p-2">
            {selectedProject.files.length > 0 ? (
              renderFileTree(selectedProject.files)
            ) : (
              <div className="text-center py-12 text-gray-400">
                <Database className="w-12 h-12 mx-auto mb-2 opacity-30" />
                <p>No files in this project</p>
                <button className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 text-sm">
                  Add First File
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* File Details */}
      {selectedFile && (
        <div className="flex-1 bg-white rounded-lg shadow overflow-hidden">
          {/* Header */}
          <div className="p-6 border-b">
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-3">
                <File className="w-8 h-8 text-blue-500" />
                <div>
                  <h3 className="text-xl font-semibold">{selectedFile.name}</h3>
                  <p className="text-sm text-gray-600 mt-1">
                    {selectedFile.dataType && (
                      <span className="px-2 py-0.5 bg-blue-100 text-blue-700 rounded text-xs mr-2">
                        {selectedFile.dataType}
                      </span>
                    )}
                    {selectedFile.size && `${(selectedFile.size / 1024 / 1024).toFixed(2)} MB`}
                  </p>
                </div>
              </div>
              <div className="flex gap-2">
                <button className="p-2 hover:bg-gray-100 rounded">
                  <Edit className="w-5 h-5" />
                </button>
                <button className="p-2 hover:bg-gray-100 rounded">
                  <Trash2 className="w-5 h-5 text-red-500" />
                </button>
                <button className="p-2 hover:bg-gray-100 rounded">
                  <MoreVertical className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>

          {/* Metadata */}
          <div className="p-6 space-y-6">
            {/* Tags */}
            <div>
              <h4 className="text-sm font-semibold mb-2 flex items-center gap-2">
                <Tag className="w-4 h-4" />
                Tags
              </h4>
              <div className="flex flex-wrap gap-2">
                {selectedFile.tags.map(tag => (
                  <span key={tag} className="px-3 py-1 bg-gray-100 text-sm rounded-full">
                    {tag}
                  </span>
                ))}
                <button className="px-3 py-1 border-2 border-dashed border-gray-300 text-sm rounded-full hover:border-gray-400 transition">
                  + Add Tag
                </button>
              </div>
            </div>

            {/* Timestamps */}
            <div>
              <h4 className="text-sm font-semibold mb-2 flex items-center gap-2">
                <Calendar className="w-4 h-4" />
                Timeline
              </h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Created:</span>
                  <span>{formatDate(selectedFile.createdAt)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Modified:</span>
                  <span>{formatDate(selectedFile.modifiedAt)}</span>
                </div>
              </div>
            </div>

            {/* Custom Metadata */}
            {Object.keys(selectedFile.metadata).length > 0 && (
              <div>
                <h4 className="text-sm font-semibold mb-2 flex items-center gap-2">
                  <FileText className="w-4 h-4" />
                  Metadata
                </h4>
                <div className="space-y-2">
                  {Object.entries(selectedFile.metadata).map(([key, value]) => (
                    <div key={key} className="flex justify-between text-sm">
                      <span className="text-gray-600">{key}:</span>
                      <span className="font-mono">{JSON.stringify(value)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Actions */}
            <div className="pt-4 border-t flex gap-3">
              <button className="flex-1 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition">
                Open in Viewer
              </button>
              <button className="px-4 py-2 border rounded hover:bg-gray-50 transition">
                Download
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
