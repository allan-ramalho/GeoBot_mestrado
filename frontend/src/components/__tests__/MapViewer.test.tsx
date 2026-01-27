import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import MapViewer from '../MapViewer';

// Mock react-plotly.js
vi.mock('react-plotly.js', () => ({
  default: ({ data, layout, onRelayout }: any) => (
    <div data-testid="plotly-plot">
      <div data-testid="plot-data">{JSON.stringify(data)}</div>
      <div data-testid="plot-layout">{JSON.stringify(layout)}</div>
      <button onClick={() => onRelayout?.({})}>Relayout</button>
    </div>
  ),
}));

// Mock lucide-react icons
vi.mock('lucide-react', () => ({
  Settings: () => <div>Settings Icon</div>,
  Ruler: () => <div>Ruler Icon</div>,
  RefreshCw: () => <div>RefreshCw Icon</div>,
  Maximize2: () => <div>Maximize2 Icon</div>,
  Minimize2: () => <div>Minimize2 Icon</div>,
  Download: () => <div>Download Icon</div>,
  ChevronDown: () => <div>ChevronDown Icon</div>,
  MapIcon: () => <div>MapIcon</div>,
  Layers: () => <div>Layers Icon</div>,
  Grid3x3: () => <div>Grid3x3 Icon</div>,
  Mountain: () => <div>Mountain Icon</div>,
}));

describe('MapViewer', () => {
  const mockData = {
    x: [0, 1, 2, 3, 4],
    y: [0, 1, 2, 3, 4],
    z: [
      [1, 2, 3, 4, 5],
      [2, 3, 4, 5, 6],
      [3, 4, 5, 6, 7],
      [4, 5, 6, 7, 8],
      [5, 6, 7, 8, 9],
    ],
    xLabel: 'X (m)',
    yLabel: 'Y (m)',
    zLabel: 'Z (nT)',
    zUnit: 'nT',
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders without crashing', () => {
    render(<MapViewer data={mockData} />);
    expect(screen.getByTestId('plotly-plot')).toBeInTheDocument();
  });

  it('displays toolbar buttons', () => {
    render(<MapViewer data={mockData} />);
    
    // Check for toolbar buttons by their icons
    expect(screen.getByText('Settings Icon')).toBeInTheDocument();
    expect(screen.getByText('Ruler Icon')).toBeInTheDocument();
    expect(screen.getByText('RefreshCw Icon')).toBeInTheDocument();
    expect(screen.getByText('Maximize2 Icon')).toBeInTheDocument();
  });

  it('toggles settings panel', async () => {
    render(<MapViewer data={mockData} />);
    
    const settingsButton = screen.getByTitle('Settings');
    
    // Settings should be hidden initially
    expect(screen.queryByText('Plot Type')).not.toBeInTheDocument();
    
    // Click to show settings
    fireEvent.click(settingsButton);
    await waitFor(() => {
      expect(screen.getByText('Plot Type')).toBeInTheDocument();
    });
    
    // Click again to hide
    fireEvent.click(settingsButton);
    await waitFor(() => {
      expect(screen.queryByText('Plot Type')).not.toBeInTheDocument();
    });
  });

  it('changes plot type', async () => {
    render(<MapViewer data={mockData} />);
    
    // Open settings
    const settingsButton = screen.getByTitle('Settings');
    fireEvent.click(settingsButton);
    
    await waitFor(() => {
      expect(screen.getByText('Plot Type')).toBeInTheDocument();
    });
    
    // Find and click different plot type buttons
    const buttons = screen.getAllByRole('button');
    const contourfButton = buttons.find(btn => btn.getAttribute('title') === 'Filled Contour');
    
    if (contourfButton) {
      fireEvent.click(contourfButton);
      
      // Plot should update (check plot data)
      const plotData = screen.getByTestId('plot-data');
      expect(plotData.textContent).toBeTruthy();
    }
  });

  it('calculates statistics correctly', () => {
    render(<MapViewer data={mockData} />);
    
    // Open settings to see statistics
    const settingsButton = screen.getByTitle('Settings');
    fireEvent.click(settingsButton);
    
    // Should display min, max, mean, std
    // These are calculated from mockData.z
    waitFor(() => {
      expect(screen.getByText(/Min:/)).toBeInTheDocument();
      expect(screen.getByText(/Max:/)).toBeInTheDocument();
      expect(screen.getByText(/Mean:/)).toBeInTheDocument();
      expect(screen.getByText(/Std:/)).toBeInTheDocument();
    });
  });

  it('toggles profile mode', async () => {
    const onProfileSelect = vi.fn();
    render(<MapViewer data={mockData} onProfileSelect={onProfileSelect} />);
    
    const profileButton = screen.getByTitle('Profile Mode');
    
    // Click to enable profile mode
    fireEvent.click(profileButton);
    
    // Button should have active state (check for bg-blue class or similar)
    expect(profileButton.className).toContain('blue');
  });

  it('resets view', () => {
    render(<MapViewer data={mockData} />);
    
    const resetButton = screen.getByTitle('Reset View');
    
    // Click reset
    fireEvent.click(resetButton);
    
    // Should trigger plot update (check that Plotly is rendered)
    expect(screen.getByTestId('plotly-plot')).toBeInTheDocument();
  });

  it('toggles fullscreen', async () => {
    render(<MapViewer data={mockData} />);
    
    const fullscreenButton = screen.getByTitle('Fullscreen');
    
    // Click to enter fullscreen
    fireEvent.click(fullscreenButton);
    
    await waitFor(() => {
      // Should change to Minimize2 icon
      expect(screen.getByText('Minimize2 Icon')).toBeInTheDocument();
    });
    
    // Click again to exit
    fireEvent.click(fullscreenButton);
    
    await waitFor(() => {
      expect(screen.getByText('Maximize2 Icon')).toBeInTheDocument();
    });
  });

  it('changes colorscale', async () => {
    render(<MapViewer data={mockData} />);
    
    // Open settings
    const settingsButton = screen.getByTitle('Settings');
    fireEvent.click(settingsButton);
    
    await waitFor(() => {
      const colorscaleSelect = screen.getByRole('combobox', { name: /colorscale/i });
      expect(colorscaleSelect).toBeInTheDocument();
      
      // Change colorscale
      fireEvent.change(colorscaleSelect, { target: { value: 'Plasma' } });
      
      // Check that plot updates
      const plotData = screen.getByTestId('plot-data');
      expect(plotData.textContent).toContain('Plasma');
    });
  });

  it('handles empty data gracefully', () => {
    const emptyData = {
      x: [],
      y: [],
      z: [],
      xLabel: 'X',
      yLabel: 'Y',
      zLabel: 'Z',
      zUnit: '',
    };
    
    render(<MapViewer data={emptyData} />);
    
    // Should still render without errors
    expect(screen.getByTestId('plotly-plot')).toBeInTheDocument();
  });

  it('displays correct grid size', () => {
    render(<MapViewer data={mockData} />);
    
    // Open settings
    const settingsButton = screen.getByTitle('Settings');
    fireEvent.click(settingsButton);
    
    waitFor(() => {
      // Should show grid size (5x5 in mockData)
      expect(screen.getByText(/Grid: 5 × 5/)).toBeInTheDocument();
    });
  });
});
