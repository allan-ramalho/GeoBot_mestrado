import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import ProcessingPanel from '../ProcessingPanel';

// Mock lucide-react
vi.mock('lucide-react', () => ({
  Search: () => <div>Search Icon</div>,
  ChevronDown: () => <div>ChevronDown Icon</div>,
  ChevronRight: () => <div>ChevronRight Icon</div>,
  Play: () => <div>Play Icon</div>,
  Loader: () => <div>Loader Icon</div>,
  CheckCircle: () => <div>CheckCircle Icon</div>,
  AlertCircle: () => <div>AlertCircle Icon</div>,
  Clock: () => <div>Clock Icon</div>,
  Trash2: () => <div>Trash2 Icon</div>,
}));

describe('ProcessingPanel', () => {
  const mockOnExecute = vi.fn();
  const mockJobs = [
    {
      id: 'job1',
      functionId: 'reduction_to_pole',
      functionName: 'Reduction to Pole',
      params: { inclination: -30, declination: 0 },
      status: 'completed' as const,
      progress: 100,
      startTime: Date.now() - 5000,
      endTime: Date.now(),
    },
    {
      id: 'job2',
      functionId: 'upward_continuation',
      functionName: 'Upward Continuation',
      params: { altitude: 500 },
      status: 'running' as const,
      progress: 50,
      startTime: Date.now() - 2000,
    },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders without crashing', () => {
    render(<ProcessingPanel onExecute={mockOnExecute} jobs={[]} />);
    expect(screen.getByText('Search Icon')).toBeInTheDocument();
  });

  it('displays function categories', () => {
    render(<ProcessingPanel onExecute={mockOnExecute} jobs={[]} />);
    
    expect(screen.getByText('🌍')).toBeInTheDocument(); // Gravity icon
    expect(screen.getByText('Gravity')).toBeInTheDocument();
    expect(screen.getByText('Magnetic')).toBeInTheDocument();
    expect(screen.getByText('Filters')).toBeInTheDocument();
    expect(screen.getByText('Advanced')).toBeInTheDocument();
  });

  it('expands and collapses categories', async () => {
    render(<ProcessingPanel onExecute={mockOnExecute} jobs={[]} />);
    
    // Initially, gravity category should be expanded
    expect(screen.getByText('Bouguer Correction')).toBeInTheDocument();
    
    // Click to collapse
    const gravityHeader = screen.getByText('Gravity').closest('button');
    if (gravityHeader) {
      fireEvent.click(gravityHeader);
      
      await waitFor(() => {
        expect(screen.queryByText('Bouguer Correction')).not.toBeInTheDocument();
      });
    }
  });

  it('searches functions by keyword', async () => {
    render(<ProcessingPanel onExecute={mockOnExecute} jobs={[]} />);
    
    const searchInput = screen.getByPlaceholderText(/search functions/i);
    
    // Search for "pole"
    fireEvent.change(searchInput, { target: { value: 'pole' } });
    
    await waitFor(() => {
      // Should show Reduction to Pole
      expect(screen.getByText('Reduction to Pole')).toBeInTheDocument();
      // Should hide unrelated functions
      expect(screen.queryByText('Gaussian Filter')).not.toBeInTheDocument();
    });
  });

  it('selects a function and shows parameters', async () => {
    render(<ProcessingPanel onExecute={mockOnExecute} jobs={[]} />);
    
    // Click on a function
    const rtpFunction = screen.getByText('Reduction to Pole');
    fireEvent.click(rtpFunction);
    
    await waitFor(() => {
      // Should show parameter form
      expect(screen.getByText(/inclination/i)).toBeInTheDocument();
      expect(screen.getByText(/declination/i)).toBeInTheDocument();
    });
  });

  it('updates parameter values', async () => {
    render(<ProcessingPanel onExecute={mockOnExecute} jobs={[]} />);
    
    // Select function
    const rtpFunction = screen.getByText('Reduction to Pole');
    fireEvent.click(rtpFunction);
    
    await waitFor(() => {
      const inclinationInput = screen.getByLabelText(/inclination/i);
      
      // Change value
      fireEvent.change(inclinationInput, { target: { value: '-45' } });
      
      expect(inclinationInput).toHaveValue(-45);
    });
  });

  it('executes function with parameters', async () => {
    render(<ProcessingPanel onExecute={mockOnExecute} jobs={[]} />);
    
    // Select function
    const rtpFunction = screen.getByText('Reduction to Pole');
    fireEvent.click(rtpFunction);
    
    await waitFor(async () => {
      // Click execute button
      const executeButton = screen.getByText(/execute/i);
      fireEvent.click(executeButton);
      
      // Should call onExecute with correct params
      expect(mockOnExecute).toHaveBeenCalledWith(
        expect.objectContaining({
          functionId: 'reduction_to_pole',
          params: expect.any(Object),
        })
      );
    });
  });

  it('displays processing queue', () => {
    render(<ProcessingPanel onExecute={mockOnExecute} jobs={mockJobs} />);
    
    // Should show both jobs
    expect(screen.getByText('Reduction to Pole')).toBeInTheDocument();
    expect(screen.getByText('Upward Continuation')).toBeInTheDocument();
  });

  it('shows job status icons', () => {
    render(<ProcessingPanel onExecute={mockOnExecute} jobs={mockJobs} />);
    
    // Should show completed and running icons
    const checkIcons = screen.getAllByText('CheckCircle Icon');
    const loaderIcons = screen.getAllByText('Loader Icon');
    
    expect(checkIcons.length).toBeGreaterThan(0);
    expect(loaderIcons.length).toBeGreaterThan(0);
  });

  it('displays progress bars for running jobs', () => {
    render(<ProcessingPanel onExecute={mockOnExecute} jobs={mockJobs} />);
    
    // Should show progress bar (50%)
    const progressBars = screen.getAllByRole('progressbar');
    expect(progressBars.length).toBeGreaterThan(0);
  });

  it('clears processing queue', async () => {
    const mockOnClearQueue = vi.fn();
    render(
      <ProcessingPanel
        onExecute={mockOnExecute}
        jobs={mockJobs}
        onClearQueue={mockOnClearQueue}
      />
    );
    
    const clearButton = screen.getByText(/clear/i);
    fireEvent.click(clearButton);
    
    await waitFor(() => {
      expect(mockOnClearQueue).toHaveBeenCalled();
    });
  });

  it('filters functions by category', async () => {
    render(<ProcessingPanel onExecute={mockOnExecute} jobs={[]} />);
    
    // Collapse all and expand only Filters
    const filterHeader = screen.getByText('Filters').closest('button');
    if (filterHeader) {
      fireEvent.click(filterHeader);
      
      await waitFor(() => {
        expect(screen.getByText('Butterworth Filter')).toBeInTheDocument();
        expect(screen.getByText('Gaussian Filter')).toBeInTheDocument();
      });
    }
  });

  it('validates parameter ranges', async () => {
    render(<ProcessingPanel onExecute={mockOnExecute} jobs={[]} />);
    
    // Select Bouguer Correction
    const bouguerFunction = screen.getByText('Bouguer Correction');
    fireEvent.click(bouguerFunction);
    
    await waitFor(() => {
      const densityInput = screen.getByLabelText(/density/i) as HTMLInputElement;
      
      // Check min/max attributes
      expect(densityInput.min).toBe('2');
      expect(densityInput.max).toBe('3.5');
    });
  });

  it('displays function descriptions', () => {
    render(<ProcessingPanel onExecute={mockOnExecute} jobs={[]} />);
    
    // Should show descriptions
    expect(screen.getByText(/Remove gravitational effect of topography/i)).toBeInTheDocument();
  });

  it('shows execution time for completed jobs', () => {
    render(<ProcessingPanel onExecute={mockOnExecute} jobs={mockJobs} />);
    
    // Should show duration (5 seconds for job1)
    expect(screen.getByText(/5\.0s/)).toBeInTheDocument();
  });

  it('handles comparison mode toggle', async () => {
    const mockOnToggleComparison = vi.fn();
    render(
      <ProcessingPanel
        onExecute={mockOnExecute}
        jobs={[]}
        onToggleComparison={mockOnToggleComparison}
      />
    );
    
    // Find and click comparison checkbox
    const comparisonCheckbox = screen.getByLabelText(/compare with original/i);
    fireEvent.click(comparisonCheckbox);
    
    await waitFor(() => {
      expect(mockOnToggleComparison).toHaveBeenCalled();
    });
  });
});
