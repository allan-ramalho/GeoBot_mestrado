/**
 * Undo/Redo System
 * History management for processing operations
 */

import React, { createContext, useContext, useCallback, useReducer } from 'react';
import { Undo, Redo, History } from 'lucide-react';

interface HistoryState {
  id: string;
  timestamp: Date;
  type: 'processing' | 'project' | 'settings';
  description: string;
  data: any;
}

interface HistoryContextType {
  past: HistoryState[];
  present: HistoryState | null;
  future: HistoryState[];
  canUndo: boolean;
  canRedo: boolean;
  undo: () => void;
  redo: () => void;
  pushState: (state: Omit<HistoryState, 'id' | 'timestamp'>) => void;
  clearHistory: () => void;
  getHistory: () => HistoryState[];
}

type HistoryAction =
  | { type: 'PUSH'; payload: HistoryState }
  | { type: 'UNDO' }
  | { type: 'REDO' }
  | { type: 'CLEAR' }
  | { type: 'RESTORE'; payload: HistoryState[] };

interface HistoryReducerState {
  past: HistoryState[];
  present: HistoryState | null;
  future: HistoryState[];
}

const MAX_HISTORY_SIZE = 50;

// History reducer
function historyReducer(state: HistoryReducerState, action: HistoryAction): HistoryReducerState {
  switch (action.type) {
    case 'PUSH': {
      const newPresent = action.payload;
      const newPast = state.present
        ? [...state.past, state.present].slice(-MAX_HISTORY_SIZE)
        : state.past;

      return {
        past: newPast,
        present: newPresent,
        future: [], // Clear future on new action
      };
    }

    case 'UNDO': {
      if (state.past.length === 0) return state;

      const previous = state.past[state.past.length - 1];
      const newPast = state.past.slice(0, -1);
      const newFuture = state.present ? [state.present, ...state.future] : state.future;

      return {
        past: newPast,
        present: previous,
        future: newFuture.slice(0, MAX_HISTORY_SIZE),
      };
    }

    case 'REDO': {
      if (state.future.length === 0) return state;

      const next = state.future[0];
      const newFuture = state.future.slice(1);
      const newPast = state.present ? [...state.past, state.present] : state.past;

      return {
        past: newPast.slice(-MAX_HISTORY_SIZE),
        present: next,
        future: newFuture,
      };
    }

    case 'CLEAR': {
      return {
        past: [],
        present: null,
        future: [],
      };
    }

    case 'RESTORE': {
      const states = action.payload;
      return {
        past: states.slice(0, -1),
        present: states[states.length - 1] || null,
        future: [],
      };
    }

    default:
      return state;
  }
}

// Create context
const HistoryContext = createContext<HistoryContextType | undefined>(undefined);

// Provider component
export function HistoryProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(historyReducer, {
    past: [],
    present: null,
    future: [],
  });

  // Load history from localStorage on mount
  React.useEffect(() => {
    const savedHistory = localStorage.getItem('geobot-history');
    if (savedHistory) {
      try {
        const parsed = JSON.parse(savedHistory);
        dispatch({ type: 'RESTORE', payload: parsed });
      } catch (error) {
        console.error('Failed to load history:', error);
      }
    }
  }, []);

  // Save history to localStorage
  React.useEffect(() => {
    const allStates = [...state.past];
    if (state.present) {
      allStates.push(state.present);
    }

    localStorage.setItem('geobot-history', JSON.stringify(allStates));
  }, [state.past, state.present]);

  const pushState = useCallback((newState: Omit<HistoryState, 'id' | 'timestamp'>) => {
    const fullState: HistoryState = {
      ...newState,
      id: `${Date.now()}-${Math.random()}`,
      timestamp: new Date(),
    };
    dispatch({ type: 'PUSH', payload: fullState });
  }, []);

  const undo = useCallback(() => {
    dispatch({ type: 'UNDO' });
  }, []);

  const redo = useCallback(() => {
    dispatch({ type: 'REDO' });
  }, []);

  const clearHistory = useCallback(() => {
    dispatch({ type: 'CLEAR' });
    localStorage.removeItem('geobot-history');
  }, []);

  const getHistory = useCallback(() => {
    const all = [...state.past];
    if (state.present) all.push(state.present);
    return all;
  }, [state.past, state.present]);

  // Keyboard shortcuts
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.key === 'z' && !e.shiftKey) {
        e.preventDefault();
        undo();
      }
      if ((e.ctrlKey && e.key === 'y') || (e.ctrlKey && e.shiftKey && e.key === 'z')) {
        e.preventDefault();
        redo();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [undo, redo]);

  const value: HistoryContextType = {
    past: state.past,
    present: state.present,
    future: state.future,
    canUndo: state.past.length > 0,
    canRedo: state.future.length > 0,
    undo,
    redo,
    pushState,
    clearHistory,
    getHistory,
  };

  return <HistoryContext.Provider value={value}>{children}</HistoryContext.Provider>;
}

// Hook to use history
export function useHistory() {
  const context = useContext(HistoryContext);
  if (!context) {
    throw new Error('useHistory must be used within HistoryProvider');
  }
  return context;
}

// History controls component
export function HistoryControls() {
  const { canUndo, canRedo, undo, redo, past, present, future } = useHistory();
  const [showPanel, setShowPanel] = React.useState(false);

  const allHistory = [...past];
  if (present) allHistory.push(present);

  return (
    <>
      {/* Controls */}
      <div className="flex items-center gap-2">
        <button
          onClick={undo}
          disabled={!canUndo}
          className="p-2 rounded hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed transition"
          title="Undo (Ctrl+Z)"
        >
          <Undo className="w-4 h-4" />
        </button>

        <button
          onClick={redo}
          disabled={!canRedo}
          className="p-2 rounded hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed transition"
          title="Redo (Ctrl+Y)"
        >
          <Redo className="w-4 h-4" />
        </button>

        <button
          onClick={() => setShowPanel(!showPanel)}
          className="p-2 rounded hover:bg-gray-100 transition relative"
          title="History"
        >
          <History className="w-4 h-4" />
          {allHistory.length > 0 && (
            <span className="absolute -top-1 -right-1 bg-blue-500 text-white text-xs rounded-full w-4 h-4 flex items-center justify-center">
              {allHistory.length}
            </span>
          )}
        </button>
      </div>

      {/* History Panel */}
      {showPanel && (
        <div className="absolute top-12 right-0 bg-white rounded-lg shadow-lg border w-80 max-h-96 overflow-hidden z-50">
          <div className="p-3 border-b bg-gray-50 font-semibold">
            History
          </div>

          <div className="overflow-y-auto max-h-80">
            {allHistory.length === 0 ? (
              <div className="p-4 text-center text-gray-500">
                No history yet
              </div>
            ) : (
              <div className="divide-y">
                {allHistory.reverse().map((item, idx) => (
                  <div
                    key={item.id}
                    className={`p-3 hover:bg-gray-50 cursor-pointer ${
                      idx === 0 ? 'bg-blue-50' : ''
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1">
                        <div className="font-medium text-sm">{item.description}</div>
                        <div className="text-xs text-gray-500 mt-1">
                          {new Date(item.timestamp).toLocaleString()}
                        </div>
                      </div>
                      {idx === 0 && (
                        <span className="text-xs bg-blue-500 text-white px-2 py-0.5 rounded">
                          Current
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}

// Hook for tracking processing operations
export function useProcessingHistory() {
  const { pushState } = useHistory();

  const trackProcessing = useCallback(
    (functionId: string, params: any, result: any) => {
      pushState({
        type: 'processing',
        description: `Processed: ${functionId}`,
        data: {
          functionId,
          params,
          result,
        },
      });
    },
    [pushState]
  );

  return { trackProcessing };
}

// Hook for tracking project changes
export function useProjectHistory() {
  const { pushState } = useHistory();

  const trackProjectChange = useCallback(
    (action: string, projectId: string, data: any) => {
      pushState({
        type: 'project',
        description: `Project ${action}: ${projectId}`,
        data: {
          action,
          projectId,
          data,
        },
      });
    },
    [pushState]
  );

  return { trackProjectChange };
}
