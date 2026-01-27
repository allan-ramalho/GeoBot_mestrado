/**
 * Advanced Map Viewer with Plotly
 * Features: contour, heatmap, 3D, colormaps, cross-sections, profiles, export
 */

import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { 
  Settings, 
  Download, 
  Maximize2, 
  Minimize2, 
  RefreshCw,
  Grid3x3,
  Mountain,
  Map as MapIcon,
  Layers,
  Ruler,
  Palette
} from 'lucide-react';

// Colorscale options
const COLORSCALES = [
  { value: 'Viridis', label: 'Viridis', type: 'sequential' },
  { value: 'Plasma', label: 'Plasma', type: 'sequential' },
  { value: 'Jet', label: 'Jet', type: 'sequential' },
  { value: 'Rainbow', label: 'Rainbow', type: 'sequential' },
  { value: 'Hot', label: 'Hot', type: 'sequential' },
  { value: 'Cool', label: 'Cool', type: 'sequential' },
  { value: 'RdBu', label: 'Red-Blue', type: 'diverging' },
  { value: 'RdYlGn', label: 'Red-Yellow-Green', type: 'diverging' },
  { value: 'Picnic', label: 'Picnic', type: 'diverging' },
  { value: 'Portland', label: 'Portland', type: 'sequential' },
  { value: 'Earth', label: 'Earth', type: 'sequential' },
  { value: 'Electric', label: 'Electric', type: 'sequential' },
];

type PlotType = 'contour' | 'heatmap' | 'surface' | 'contourf';

interface MapData {
  x: number[];
  y: number[];
  z: number[][];
  title?: string;
  xlabel?: string;
  ylabel?: string;
  zlabel?: string;
  unit?: string;
}

interface MapViewerProps {
  data: MapData;
  width?: number;
  height?: number;
  showControls?: boolean;
  onProfileSelect?: (x1: number, y1: number, x2: number, y2: number) => void;
}

interface PlotSettings {
  plotType: PlotType;
  colorscale: string;
  showContourLines: boolean;
  contourLevels: number;
  showColorbar: boolean;
  reverseColorscale: boolean;
  zmin: number | null;
  zmax: number | null;
  aspectRatio: 'auto' | 'equal';
  showGrid: boolean;
}

export const MapViewer: React.FC<MapViewerProps> = ({
  data,
  width = 800,
  height = 600,
  showControls = true,
  onProfileSelect,
}) => {
  const plotRef = useRef<any>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [profileMode, setProfileMode] = useState(false);
  const [profilePoints, setProfilePoints] = useState<Array<{x: number, y: number}>>([]);

  // Default settings
  const [settings, setSettings] = useState<PlotSettings>({
    plotType: 'contourf',
    colorscale: 'Viridis',
    showContourLines: true,
    contourLevels: 20,
    showColorbar: true,
    reverseColorscale: false,
    zmin: null,
    zmax: null,
    aspectRatio: 'auto',
    showGrid: true,
  });

  // Calculate data statistics
  const dataStats = useMemo(() => {
    if (!data.z || data.z.length === 0) return { min: 0, max: 1, mean: 0.5, std: 0.5 };

    const flatZ = data.z.flat();
    const validZ = flatZ.filter(v => !isNaN(v) && isFinite(v));
    
    const min = Math.min(...validZ);
    const max = Math.max(...validZ);
    const mean = validZ.reduce((a, b) => a + b, 0) / validZ.length;
    const std = Math.sqrt(
      validZ.reduce((sq, v) => sq + Math.pow(v - mean, 2), 0) / validZ.length
    );

    return { min, max, mean, std };
  }, [data.z]);

  // Prepare plot data
  const plotData = useMemo(() => {
    const zmin = settings.zmin ?? dataStats.min;
    const zmax = settings.zmax ?? dataStats.max;

    const baseTrace: any = {
      x: data.x,
      y: data.y,
      z: data.z,
      colorscale: settings.colorscale,
      reversescale: settings.reverseColorscale,
      zmin,
      zmax,
      colorbar: settings.showColorbar ? {
        title: data.unit || 'Value',
        titleside: 'right',
        tickmode: 'auto',
        nticks: 10,
        thickness: 20,
        len: 0.9,
      } : undefined,
    };

    switch (settings.plotType) {
      case 'contour':
        return [{
          ...baseTrace,
          type: 'contour',
          contours: {
            coloring: 'lines',
            showlabels: true,
            labelfont: { size: 10, color: 'white' },
            start: zmin,
            end: zmax,
            size: (zmax - zmin) / settings.contourLevels,
          },
          line: { width: 1 },
        }];

      case 'contourf':
        const traces: any[] = [{
          ...baseTrace,
          type: 'contour',
          contours: {
            coloring: 'heatmap',
            start: zmin,
            end: zmax,
            size: (zmax - zmin) / settings.contourLevels,
          },
        }];

        if (settings.showContourLines) {
          traces.push({
            x: data.x,
            y: data.y,
            z: data.z,
            type: 'contour',
            showscale: false,
            contours: {
              coloring: 'none',
              showlabels: true,
              labelfont: { size: 9, color: 'black' },
              start: zmin,
              end: zmax,
              size: (zmax - zmin) / 10,
            },
            line: { color: 'black', width: 0.5 },
          });
        }

        return traces;

      case 'heatmap':
        return [{
          ...baseTrace,
          type: 'heatmap',
          hovertemplate: 'x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>',
        }];

      case 'surface':
        return [{
          ...baseTrace,
          type: 'surface',
          contours: {
            z: {
              show: settings.showContourLines,
              usecolormap: true,
              highlightcolor: '#42f462',
              project: { z: true },
            },
          },
        }];

      default:
        return [baseTrace];
    }
  }, [data, settings, dataStats]);

  // Layout configuration
  const layout = useMemo(() => {
    const is3D = settings.plotType === 'surface';

    const baseLayout: any = {
      title: data.title || 'Geophysical Data Map',
      autosize: true,
      width: isFullscreen ? window.innerWidth : width,
      height: isFullscreen ? window.innerHeight : height,
      margin: { l: 60, r: 60, t: 80, b: 60 },
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#f8f9fa',
      hovermode: 'closest',
      dragmode: profileMode ? 'select' : 'zoom',
    };

    if (is3D) {
      baseLayout.scene = {
        xaxis: { title: data.xlabel || 'X (m)', gridcolor: '#ddd' },
        yaxis: { title: data.ylabel || 'Y (m)', gridcolor: '#ddd' },
        zaxis: { title: data.zlabel || data.unit || 'Z', gridcolor: '#ddd' },
        camera: {
          eye: { x: 1.5, y: 1.5, z: 1.3 },
          center: { x: 0, y: 0, z: 0 },
        },
        aspectmode: settings.aspectRatio === 'equal' ? 'cube' : 'auto',
      };
    } else {
      baseLayout.xaxis = {
        title: data.xlabel || 'X (m)',
        gridcolor: settings.showGrid ? '#e5e5e5' : 'transparent',
        showgrid: settings.showGrid,
      };
      baseLayout.yaxis = {
        title: data.ylabel || 'Y (m)',
        gridcolor: settings.showGrid ? '#e5e5e5' : 'transparent',
        showgrid: settings.showGrid,
        scaleanchor: settings.aspectRatio === 'equal' ? 'x' : undefined,
      };
    }

    return baseLayout;
  }, [data, settings, width, height, isFullscreen, profileMode]);

  // Config for Plotly
  const config = useMemo(() => ({
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
    modeBarButtonsToAdd: [
      {
        name: 'Download as PNG',
        icon: {
          width: 500,
          height: 600,
          path: 'M16 16v4a2 2 0 002 2h8a2 2 0 002-2v-4M12 12l-4 4m0 0l4 4m-4-4h12'
        },
        click: () => exportPlot('png'),
      },
    ],
    toImageButtonOptions: {
      format: 'png',
      filename: data.title?.replace(/\s+/g, '_') || 'geobot_map',
      width: 1920,
      height: 1080,
      scale: 2,
    },
    responsive: true,
  }), [data.title]);

  // Export functions
  const exportPlot = useCallback((format: 'png' | 'svg' | 'pdf' | 'json') => {
    const plotElement = plotRef.current?.el;
    if (!plotElement) return;

    if (format === 'json') {
      // Export data as JSON
      const jsonData = {
        data: data,
        settings: settings,
        timestamp: new Date().toISOString(),
      };
      const blob = new Blob([JSON.stringify(jsonData, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${data.title || 'map'}_data.json`;
      a.click();
      URL.revokeObjectURL(url);
    } else {
      // Use Plotly's built-in export
      import('plotly.js').then((Plotly) => {
        Plotly.downloadImage(plotElement, {
          format: format as any,
          width: 1920,
          height: 1080,
          filename: data.title?.replace(/\s+/g, '_') || 'geobot_map',
        });
      });
    }
  }, [data, settings]);

  // Handle plot click for profile selection
  const handlePlotClick = useCallback((event: any) => {
    if (!profileMode || !event.points || event.points.length === 0) return;

    const point = event.points[0];
    const newPoint = { x: point.x, y: point.y };

    setProfilePoints(prev => {
      const updated = [...prev, newPoint];
      if (updated.length === 2) {
        // Call callback with two points
        if (onProfileSelect) {
          onProfileSelect(updated[0].x, updated[0].y, updated[1].x, updated[1].y);
        }
        return []; // Reset
      }
      return updated;
    });
  }, [profileMode, onProfileSelect]);

  // Reset view
  const resetView = useCallback(() => {
    if (plotRef.current) {
      const plotElement = plotRef.current.el;
      import('plotly.js').then((Plotly) => {
        Plotly.relayout(plotElement, {
          'xaxis.autorange': true,
          'yaxis.autorange': true,
          'scene.camera': {
            eye: { x: 1.5, y: 1.5, z: 1.3 },
          },
        });
      });
    }
  }, []);

  return (
    <div className={`relative ${isFullscreen ? 'fixed inset-0 z-50 bg-white' : ''}`}>
      {/* Toolbar */}
      {showControls && (
        <div className="absolute top-2 right-2 z-10 flex gap-2">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="p-2 bg-white rounded-lg shadow hover:bg-gray-50 transition"
            title="Settings"
          >
            <Settings className="w-5 h-5" />
          </button>

          <button
            onClick={() => setProfileMode(!profileMode)}
            className={`p-2 rounded-lg shadow transition ${
              profileMode ? 'bg-blue-500 text-white' : 'bg-white hover:bg-gray-50'
            }`}
            title="Profile Mode"
          >
            <Ruler className="w-5 h-5" />
          </button>

          <button
            onClick={resetView}
            className="p-2 bg-white rounded-lg shadow hover:bg-gray-50 transition"
            title="Reset View"
          >
            <RefreshCw className="w-5 h-5" />
          </button>

          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="p-2 bg-white rounded-lg shadow hover:bg-gray-50 transition"
            title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
          >
            {isFullscreen ? <Minimize2 className="w-5 h-5" /> : <Maximize2 className="w-5 h-5" />}
          </button>

          <div className="relative group">
            <button
              className="p-2 bg-white rounded-lg shadow hover:bg-gray-50 transition"
              title="Export"
            >
              <Download className="w-5 h-5" />
            </button>
            <div className="absolute right-0 mt-2 w-40 bg-white rounded-lg shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all">
              <button
                onClick={() => exportPlot('png')}
                className="block w-full px-4 py-2 text-left text-sm hover:bg-gray-50"
              >
                Export PNG
              </button>
              <button
                onClick={() => exportPlot('svg')}
                className="block w-full px-4 py-2 text-left text-sm hover:bg-gray-50"
              >
                Export SVG
              </button>
              <button
                onClick={() => exportPlot('json')}
                className="block w-full px-4 py-2 text-left text-sm hover:bg-gray-50"
              >
                Export Data (JSON)
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Settings Panel */}
      {showSettings && (
        <div className="absolute top-14 right-2 z-10 w-80 bg-white rounded-lg shadow-xl p-4 max-h-[600px] overflow-y-auto">
          <h3 className="font-semibold mb-4 flex items-center gap-2">
            <Settings className="w-5 h-5" />
            Map Settings
          </h3>

          {/* Plot Type */}
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">Plot Type</label>
            <div className="grid grid-cols-2 gap-2">
              {[
                { value: 'contourf', icon: MapIcon, label: 'Filled Contour' },
                { value: 'contour', icon: Layers, label: 'Contour Lines' },
                { value: 'heatmap', icon: Grid3x3, label: 'Heatmap' },
                { value: 'surface', icon: Mountain, label: '3D Surface' },
              ].map(({ value, icon: Icon, label }) => (
                <button
                  key={value}
                  onClick={() => setSettings(s => ({ ...s, plotType: value as PlotType }))}
                  className={`p-2 rounded border flex flex-col items-center gap-1 text-xs ${
                    settings.plotType === value
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  {label}
                </button>
              ))}
            </div>
          </div>

          {/* Colorscale */}
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2 flex items-center gap-2">
              <Palette className="w-4 h-4" />
              Colorscale
            </label>
            <select
              value={settings.colorscale}
              onChange={(e) => setSettings(s => ({ ...s, colorscale: e.target.value }))}
              className="w-full px-3 py-2 border rounded"
            >
              {COLORSCALES.map(cs => (
                <option key={cs.value} value={cs.value}>
                  {cs.label} ({cs.type})
                </option>
              ))}
            </select>
            <label className="flex items-center gap-2 mt-2 text-sm">
              <input
                type="checkbox"
                checked={settings.reverseColorscale}
                onChange={(e) => setSettings(s => ({ ...s, reverseColorscale: e.target.checked }))}
              />
              Reverse colorscale
            </label>
          </div>

          {/* Z Range */}
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">Value Range</label>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-xs text-gray-600">Min</label>
                <input
                  type="number"
                  value={settings.zmin ?? ''}
                  onChange={(e) => setSettings(s => ({ 
                    ...s, 
                    zmin: e.target.value ? parseFloat(e.target.value) : null 
                  }))}
                  placeholder={dataStats.min.toFixed(2)}
                  className="w-full px-2 py-1 border rounded text-sm"
                />
              </div>
              <div>
                <label className="text-xs text-gray-600">Max</label>
                <input
                  type="number"
                  value={settings.zmax ?? ''}
                  onChange={(e) => setSettings(s => ({ 
                    ...s, 
                    zmax: e.target.value ? parseFloat(e.target.value) : null 
                  }))}
                  placeholder={dataStats.max.toFixed(2)}
                  className="w-full px-2 py-1 border rounded text-sm"
                />
              </div>
            </div>
            <div className="text-xs text-gray-500 mt-1">
              Data range: {dataStats.min.toFixed(2)} to {dataStats.max.toFixed(2)}
            </div>
          </div>

          {/* Contour Options */}
          {(settings.plotType === 'contour' || settings.plotType === 'contourf') && (
            <div className="mb-4">
              <label className="block text-sm font-medium mb-2">Contour Levels</label>
              <input
                type="range"
                min="5"
                max="50"
                value={settings.contourLevels}
                onChange={(e) => setSettings(s => ({ ...s, contourLevels: parseInt(e.target.value) }))}
                className="w-full"
              />
              <div className="text-sm text-gray-600">{settings.contourLevels} levels</div>
              
              {settings.plotType === 'contourf' && (
                <label className="flex items-center gap-2 mt-2 text-sm">
                  <input
                    type="checkbox"
                    checked={settings.showContourLines}
                    onChange={(e) => setSettings(s => ({ ...s, showContourLines: e.target.checked }))}
                  />
                  Show contour lines
                </label>
              )}
            </div>
          )}

          {/* Display Options */}
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">Display Options</label>
            <label className="flex items-center gap-2 text-sm mb-2">
              <input
                type="checkbox"
                checked={settings.showColorbar}
                onChange={(e) => setSettings(s => ({ ...s, showColorbar: e.target.checked }))}
              />
              Show colorbar
            </label>
            <label className="flex items-center gap-2 text-sm mb-2">
              <input
                type="checkbox"
                checked={settings.showGrid}
                onChange={(e) => setSettings(s => ({ ...s, showGrid: e.target.checked }))}
              />
              Show grid
            </label>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={settings.aspectRatio === 'equal'}
                onChange={(e) => setSettings(s => ({ 
                  ...s, 
                  aspectRatio: e.target.checked ? 'equal' : 'auto' 
                }))}
              />
              Equal aspect ratio
            </label>
          </div>

          {/* Statistics */}
          <div className="pt-4 border-t">
            <h4 className="text-sm font-medium mb-2">Statistics</h4>
            <div className="text-xs space-y-1 text-gray-600">
              <div>Min: {dataStats.min.toFixed(4)}</div>
              <div>Max: {dataStats.max.toFixed(4)}</div>
              <div>Mean: {dataStats.mean.toFixed(4)}</div>
              <div>Std Dev: {dataStats.std.toFixed(4)}</div>
              <div>Points: {data.x.length} × {data.y.length}</div>
            </div>
          </div>
        </div>
      )}

      {/* Profile Mode Indicator */}
      {profileMode && (
        <div className="absolute top-2 left-2 z-10 bg-blue-500 text-white px-4 py-2 rounded-lg shadow">
          <div className="flex items-center gap-2">
            <Ruler className="w-5 h-5" />
            <span className="text-sm font-medium">
              Profile Mode: Click {profilePoints.length === 0 ? 'start' : 'end'} point
            </span>
          </div>
        </div>
      )}

      {/* Plot */}
      <Plot
        ref={plotRef}
        data={plotData}
        layout={layout}
        config={config}
        onClick={handlePlotClick}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler
      />
    </div>
  );
};

export default MapViewer;
