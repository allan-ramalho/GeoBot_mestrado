/**
 * Processing Panel - Interface for geophysical data processing
 * Features: function list, search, dynamic forms, preview, comparison, queue
 */

import React, { useState, useMemo, useCallback } from 'react';
import {
  Search,
  Filter,
  Play,
  Pause,
  X,
  ChevronDown,
  ChevronRight,
  Layers,
  Zap,
  Info,
  Clock,
  CheckCircle,
  AlertCircle,
  Loader,
  SplitSquareHorizontal,
} from 'lucide-react';

// Function categories and definitions
const FUNCTION_REGISTRY = {
  gravity: {
    name: 'Gravity',
    icon: '🌍',
    functions: [
      {
        id: 'bouguer_correction',
        name: 'Bouguer Correction',
        description: 'Remove gravitational effect of topography using slab formula',
        params: [
          { name: 'density', type: 'number', default: 2.67, unit: 'g/cm³', min: 2.0, max: 3.5 },
        ],
        keywords: ['bouguer', 'correction', 'gravity', 'topography'],
      },
      {
        id: 'free_air_correction',
        name: 'Free-Air Correction',
        description: 'Account for elevation differences (0.3086 mGal/m gradient)',
        params: [
          { name: 'reference_elevation', type: 'number', default: 0, unit: 'm' },
          { name: 'gradient', type: 'number', default: 0.3086, unit: 'mGal/m' },
        ],
        keywords: ['free-air', 'elevation', 'correction'],
      },
      {
        id: 'terrain_correction',
        name: 'Terrain Correction',
        description: 'Correct for irregular topography using DEM',
        params: [
          { name: 'density', type: 'number', default: 2.67, unit: 'g/cm³' },
          { name: 'radius', type: 'number', default: 5000, unit: 'm' },
        ],
        keywords: ['terrain', 'topography', 'dem'],
      },
      {
        id: 'regional_residual_separation',
        name: 'Regional-Residual Separation',
        description: 'Separate regional and residual components',
        params: [
          { name: 'method', type: 'select', options: ['polynomial', 'upward_continuation'], default: 'polynomial' },
          { name: 'order', type: 'number', default: 2, min: 1, max: 4 },
        ],
        keywords: ['regional', 'residual', 'separation', 'trend'],
      },
      {
        id: 'isostatic_correction',
        name: 'Isostatic Correction',
        description: 'Apply Airy-Heiskanen isostatic compensation model',
        params: [
          { name: 'crustal_density', type: 'number', default: 2.67, unit: 'g/cm³' },
          { name: 'mantle_density', type: 'number', default: 3.27, unit: 'g/cm³' },
          { name: 'normal_crustal_thickness', type: 'number', default: 35000, unit: 'm' },
        ],
        keywords: ['isostatic', 'airy', 'compensation'],
      },
    ],
  },
  magnetic: {
    name: 'Magnetic',
    icon: '🧲',
    functions: [
      {
        id: 'reduction_to_pole',
        name: 'Reduction to Pole',
        description: 'Transform magnetic data to polar equivalent',
        params: [
          { name: 'inclination', type: 'number', default: -30, unit: '°', min: -90, max: 90 },
          { name: 'declination', type: 'number', default: 0, unit: '°', min: -180, max: 180 },
          { name: 'dx', type: 'number', default: 100, unit: 'm' },
          { name: 'dy', type: 'number', default: 100, unit: 'm' },
        ],
        keywords: ['rtp', 'reduction', 'pole', 'magnetic'],
      },
      {
        id: 'upward_continuation',
        name: 'Upward Continuation',
        description: 'Continue field upward to attenuate high frequencies',
        params: [
          { name: 'height', type: 'number', default: 500, unit: 'm', min: 0 },
        ],
        keywords: ['upward', 'continuation', 'regional', 'smooth'],
      },
      {
        id: 'analytic_signal',
        name: 'Analytic Signal',
        description: 'Calculate 3D analytic signal amplitude for edge detection',
        params: [
          { name: 'dx', type: 'number', default: 100, unit: 'm' },
          { name: 'dy', type: 'number', default: 100, unit: 'm' },
        ],
        keywords: ['analytic', 'signal', 'amplitude', 'edge'],
      },
      {
        id: 'total_horizontal_derivative',
        name: 'Total Horizontal Derivative',
        description: 'Calculate THD for edge detection',
        params: [
          { name: 'dx', type: 'number', default: 100, unit: 'm' },
          { name: 'dy', type: 'number', default: 100, unit: 'm' },
        ],
        keywords: ['thd', 'horizontal', 'derivative', 'edge'],
      },
      {
        id: 'vertical_derivative',
        name: 'Vertical Derivative',
        description: 'Calculate vertical derivatives (1st, 2nd, or 3rd order)',
        params: [
          { name: 'order', type: 'select', options: [1, 2, 3], default: 1 },
        ],
        keywords: ['vertical', 'derivative', 'gradient'],
      },
      {
        id: 'tilt_derivative',
        name: 'Tilt Derivative',
        description: 'Normalized edge detector independent of source depth',
        params: [],
        keywords: ['tilt', 'angle', 'normalized', 'edge'],
      },
      {
        id: 'pseudogravity',
        name: 'Pseudo-Gravity',
        description: 'Transform magnetic to equivalent gravity field',
        params: [
          { name: 'inclination', type: 'number', default: -30, unit: '°' },
          { name: 'declination', type: 'number', default: 0, unit: '°' },
          { name: 'mag_to_dens_ratio', type: 'number', default: 0.03 },
        ],
        keywords: ['pseudo-gravity', 'poisson', 'transformation'],
      },
      {
        id: 'matched_filter',
        name: 'Matched Filter',
        description: 'Enhance anomalies from specific depth range',
        params: [
          { name: 'target_depth', type: 'number', default: 1000, unit: 'm' },
          { name: 'depth_range', type: 'number', default: 500, unit: 'm' },
        ],
        keywords: ['matched', 'filter', 'depth', 'selective'],
      },
    ],
  },
  filters: {
    name: 'Filters',
    icon: '🔧',
    functions: [
      {
        id: 'butterworth_filter',
        name: 'Butterworth Filter',
        description: 'Frequency domain low/high/band-pass filter',
        params: [
          { name: 'cutoff_wavelength', type: 'number', default: 1000, unit: 'm' },
          { name: 'filter_type', type: 'select', options: ['low-pass', 'high-pass', 'band-pass'], default: 'low-pass' },
          { name: 'order', type: 'number', default: 4, min: 1, max: 8 },
        ],
        keywords: ['butterworth', 'filter', 'frequency', 'lowpass', 'highpass'],
      },
      {
        id: 'gaussian_filter',
        name: 'Gaussian Filter',
        description: 'Spatial smoothing with Gaussian kernel',
        params: [
          { name: 'sigma', type: 'number', default: 2.0, min: 0.1, max: 10 },
        ],
        keywords: ['gaussian', 'smooth', 'spatial', 'blur'],
      },
      {
        id: 'median_filter',
        name: 'Median Filter',
        description: 'Robust spike and outlier removal',
        params: [
          { name: 'size', type: 'number', default: 3, min: 3, max: 11, step: 2 },
          { name: 'threshold', type: 'number', default: 3.0, min: 1, max: 5 },
        ],
        keywords: ['median', 'spike', 'outlier', 'robust'],
      },
      {
        id: 'directional_filter',
        name: 'Directional Filter',
        description: 'Enhance features at specific azimuth',
        params: [
          { name: 'azimuth', type: 'number', default: 45, unit: '°', min: 0, max: 360 },
          { name: 'width', type: 'number', default: 30, unit: '°', min: 10, max: 90 },
        ],
        keywords: ['directional', 'azimuth', 'orientation'],
      },
      {
        id: 'wiener_filter',
        name: 'Wiener Filter',
        description: 'Optimal noise reduction filter',
        params: [
          { name: 'noise_variance', type: 'number', default: 0.1, min: 0.01, max: 1.0 },
        ],
        keywords: ['wiener', 'noise', 'optimal', 'reduction'],
      },
    ],
  },
  advanced: {
    name: 'Advanced',
    icon: '🎯',
    functions: [
      {
        id: 'euler_deconvolution',
        name: 'Euler Deconvolution',
        description: 'Automated depth and location estimation',
        params: [
          { name: 'window_size', type: 'number', default: 10, min: 5, max: 20 },
          { name: 'structural_index', type: 'select', options: [0, 1, 2, 3], default: 1 },
          { name: 'max_depth_uncertainty', type: 'number', default: 0.15, min: 0.05, max: 0.5 },
        ],
        keywords: ['euler', 'depth', 'source', 'location'],
      },
      {
        id: 'source_parameter_imaging',
        name: 'Source Parameter Imaging (SPI)',
        description: 'Simultaneous depth and structural index estimation',
        params: [
          { name: 'min_depth', type: 'number', default: 0, unit: 'm' },
          { name: 'max_depth', type: 'number', default: 5000, unit: 'm' },
          { name: 'n_depth_tests', type: 'number', default: 20, min: 10, max: 50 },
        ],
        keywords: ['spi', 'depth', 'structural', 'wavenumber'],
      },
      {
        id: 'werner_deconvolution',
        name: 'Werner Deconvolution',
        description: 'Contact and thin dike depth estimation',
        params: [
          { name: 'profile_direction', type: 'select', options: ['x', 'y'], default: 'x' },
          { name: 'window_size', type: 'number', default: 5, min: 3, max: 15 },
        ],
        keywords: ['werner', 'contact', 'dike', 'depth'],
      },
      {
        id: 'tilt_depth_method',
        name: 'Tilt-Depth Method',
        description: 'Depth estimation using tilt angle zero-crossing',
        params: [
          { name: 'dx', type: 'number', default: 100, unit: 'm' },
          { name: 'dy', type: 'number', default: 100, unit: 'm' },
        ],
        keywords: ['tilt', 'depth', 'zero', 'crossing'],
      },
    ],
  },
};

interface ProcessingJob {
  id: string;
  functionId: string;
  functionName: string;
  params: Record<string, any>;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  startTime?: Date;
  endTime?: Date;
  error?: string;
  resultPreview?: any;
}

interface ProcessingPanelProps {
  currentData?: any;
  onExecute?: (functionId: string, params: Record<string, any>) => Promise<any>;
  onCompare?: (original: any, processed: any) => void;
}

export const ProcessingPanel: React.FC<ProcessingPanelProps> = ({
  currentData,
  onExecute,
  onCompare,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set(['magnetic']));
  const [selectedFunction, setSelectedFunction] = useState<any>(null);
  const [params, setParams] = useState<Record<string, any>>({});
  const [processingQueue, setProcessingQueue] = useState<ProcessingJob[]>([]);
  const [showComparison, setShowComparison] = useState(false);

  // Filter functions based on search
  const filteredRegistry = useMemo(() => {
    if (!searchTerm) return FUNCTION_REGISTRY;

    const term = searchTerm.toLowerCase();
    const filtered: any = {};

    Object.entries(FUNCTION_REGISTRY).forEach(([catId, category]) => {
      const matchingFuncs = category.functions.filter(func =>
        func.name.toLowerCase().includes(term) ||
        func.description.toLowerCase().includes(term) ||
        func.keywords.some(kw => kw.includes(term))
      );

      if (matchingFuncs.length > 0) {
        filtered[catId] = {
          ...category,
          functions: matchingFuncs,
        };
      }
    });

    return filtered;
  }, [searchTerm]);

  // Toggle category expansion
  const toggleCategory = useCallback((catId: string) => {
    setExpandedCategories(prev => {
      const next = new Set(prev);
      if (next.has(catId)) {
        next.delete(catId);
      } else {
        next.add(catId);
      }
      return next;
    });
  }, []);

  // Select function
  const selectFunction = useCallback((func: any) => {
    setSelectedFunction(func);
    // Initialize params with defaults
    const defaultParams: Record<string, any> = {};
    func.params.forEach((p: any) => {
      defaultParams[p.name] = p.default;
    });
    setParams(defaultParams);
  }, []);

  // Execute function
  const executeFunction = useCallback(async () => {
    if (!selectedFunction || !onExecute) return;

    const job: ProcessingJob = {
      id: `job_${Date.now()}`,
      functionId: selectedFunction.id,
      functionName: selectedFunction.name,
      params: { ...params },
      status: 'pending',
      progress: 0,
    };

    setProcessingQueue(prev => [...prev, job]);

    // Update to running
    setProcessingQueue(prev =>
      prev.map(j => j.id === job.id ? { ...j, status: 'running', startTime: new Date(), progress: 50 } : j)
    );

    try {
      const result = await onExecute(selectedFunction.id, params);
      
      // Update to completed
      setProcessingQueue(prev =>
        prev.map(j => 
          j.id === job.id 
            ? { ...j, status: 'completed', progress: 100, endTime: new Date(), resultPreview: result }
            : j
        )
      );
    } catch (error: any) {
      // Update to failed
      setProcessingQueue(prev =>
        prev.map(j =>
          j.id === job.id
            ? { ...j, status: 'failed', endTime: new Date(), error: error.message }
            : j
        )
      );
    }
  }, [selectedFunction, params, onExecute]);

  // Render parameter input
  const renderParamInput = (param: any) => {
    const value = params[param.name] ?? param.default;

    switch (param.type) {
      case 'number':
        return (
          <input
            type="number"
            value={value}
            onChange={(e) => setParams(prev => ({ ...prev, [param.name]: parseFloat(e.target.value) }))}
            min={param.min}
            max={param.max}
            step={param.step || 'any'}
            className="w-full px-3 py-2 border rounded"
          />
        );

      case 'select':
        return (
          <select
            value={value}
            onChange={(e) => setParams(prev => ({ 
              ...prev, 
              [param.name]: isNaN(Number(e.target.value)) ? e.target.value : Number(e.target.value)
            }))}
            className="w-full px-3 py-2 border rounded"
          >
            {param.options.map((opt: any) => (
              <option key={opt} value={opt}>{opt}</option>
            ))}
          </select>
        );

      default:
        return (
          <input
            type="text"
            value={value}
            onChange={(e) => setParams(prev => ({ ...prev, [param.name]: e.target.value }))}
            className="w-full px-3 py-2 border rounded"
          />
        );
    }
  };

  return (
    <div className="flex h-full gap-4">
      {/* Function List */}
      <div className="w-80 bg-white rounded-lg shadow overflow-hidden flex flex-col">
        {/* Search */}
        <div className="p-4 border-b">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search functions..."
              className="w-full pl-10 pr-4 py-2 border rounded"
            />
          </div>
        </div>

        {/* Function Categories */}
        <div className="flex-1 overflow-y-auto">
          {Object.entries(filteredRegistry).map(([catId, category]) => (
            <div key={catId} className="border-b">
              <button
                onClick={() => toggleCategory(catId)}
                className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition"
              >
                <div className="flex items-center gap-2">
                  <span className="text-2xl">{category.icon}</span>
                  <span className="font-medium">{category.name}</span>
                  <span className="text-sm text-gray-500">({category.functions.length})</span>
                </div>
                {expandedCategories.has(catId) ? (
                  <ChevronDown className="w-5 h-5" />
                ) : (
                  <ChevronRight className="w-5 h-5" />
                )}
              </button>

              {expandedCategories.has(catId) && (
                <div className="bg-gray-50">
                  {category.functions.map((func: any) => (
                    <button
                      key={func.id}
                      onClick={() => selectFunction(func)}
                      className={`w-full px-6 py-3 text-left hover:bg-white transition border-l-4 ${
                        selectedFunction?.id === func.id
                          ? 'border-blue-500 bg-blue-50'
                          : 'border-transparent'
                      }`}
                    >
                      <div className="font-medium text-sm">{func.name}</div>
                      <div className="text-xs text-gray-600 mt-1">{func.description}</div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Parameter Form & Execution */}
      <div className="flex-1 bg-white rounded-lg shadow overflow-hidden flex flex-col">
        {selectedFunction ? (
          <>
            {/* Function Info */}
            <div className="p-6 border-b">
              <h2 className="text-2xl font-semibold flex items-center gap-2">
                <Zap className="w-6 h-6 text-blue-500" />
                {selectedFunction.name}
              </h2>
              <p className="text-gray-600 mt-2">{selectedFunction.description}</p>
              <div className="flex flex-wrap gap-2 mt-3">
                {selectedFunction.keywords.map((kw: string) => (
                  <span key={kw} className="px-2 py-1 bg-gray-100 text-xs rounded">
                    {kw}
                  </span>
                ))}
              </div>
            </div>

            {/* Parameters */}
            <div className="flex-1 overflow-y-auto p-6">
              {selectedFunction.params.length > 0 ? (
                <div className="space-y-4">
                  {selectedFunction.params.map((param: any) => (
                    <div key={param.name}>
                      <label className="block text-sm font-medium mb-1">
                        {param.name.replace(/_/g, ' ')}
                        {param.unit && <span className="text-gray-500 ml-1">({param.unit})</span>}
                      </label>
                      {renderParamInput(param)}
                      {(param.min !== undefined || param.max !== undefined) && (
                        <div className="text-xs text-gray-500 mt-1">
                          Range: {param.min ?? '–∞'} to {param.max ?? '∞'}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center text-gray-500 py-8">
                  <Info className="w-12 h-12 mx-auto mb-2 opacity-50" />
                  <p>No parameters required for this function</p>
                </div>
              )}
            </div>

            {/* Execute Button */}
            <div className="p-6 border-t bg-gray-50 flex gap-3">
              <button
                onClick={executeFunction}
                disabled={!currentData}
                className="flex-1 bg-blue-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-blue-600 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                <Play className="w-5 h-5" />
                Execute
              </button>
              {processingQueue.length > 0 && (
                <button
                  onClick={() => setShowComparison(!showComparison)}
                  className="px-6 py-3 border-2 border-blue-500 text-blue-500 rounded-lg font-medium hover:bg-blue-50 transition flex items-center gap-2"
                >
                  <SplitSquareHorizontal className="w-5 h-5" />
                  Compare
                </button>
              )}
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-gray-400">
            <div className="text-center">
              <Layers className="w-16 h-16 mx-auto mb-4 opacity-30" />
              <p className="text-lg">Select a function to get started</p>
              <p className="text-sm mt-2">Choose from {Object.values(FUNCTION_REGISTRY).reduce((sum, cat) => sum + cat.functions.length, 0)} available functions</p>
            </div>
          </div>
        )}
      </div>

      {/* Processing Queue */}
      {processingQueue.length > 0 && (
        <div className="w-80 bg-white rounded-lg shadow overflow-hidden flex flex-col">
          <div className="p-4 border-b flex items-center justify-between">
            <h3 className="font-semibold">Processing Queue</h3>
            <button
              onClick={() => setProcessingQueue([])}
              className="text-sm text-red-500 hover:text-red-700"
            >
              Clear All
            </button>
          </div>

          <div className="flex-1 overflow-y-auto">
            {processingQueue.slice().reverse().map((job) => (
              <div key={job.id} className="p-4 border-b">
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1">
                    <div className="font-medium text-sm">{job.functionName}</div>
                    <div className="text-xs text-gray-500 mt-1">
                      {job.startTime && new Date(job.startTime).toLocaleTimeString()}
                    </div>
                  </div>
                  {job.status === 'running' && <Loader className="w-5 h-5 animate-spin text-blue-500" />}
                  {job.status === 'completed' && <CheckCircle className="w-5 h-5 text-green-500" />}
                  {job.status === 'failed' && <AlertCircle className="w-5 h-5 text-red-500" />}
                  {job.status === 'pending' && <Clock className="w-5 h-5 text-gray-400" />}
                </div>

                {job.status === 'running' && (
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-blue-500 h-2 rounded-full transition-all"
                      style={{ width: `${job.progress}%` }}
                    />
                  </div>
                )}

                {job.status === 'failed' && (
                  <div className="text-xs text-red-600 mt-2">{job.error}</div>
                )}

                {job.status === 'completed' && job.endTime && job.startTime && (
                  <div className="text-xs text-gray-500 mt-2">
                    Completed in {((job.endTime.getTime() - job.startTime.getTime()) / 1000).toFixed(2)}s
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ProcessingPanel;
