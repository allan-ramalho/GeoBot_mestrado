/**
 * Keyboard Shortcuts System
 * Global hotkeys and command palette
 */

import React, { useState, useEffect, useCallback } from 'react';
import { Search, Command, Keyboard } from 'lucide-react';

interface Shortcut {
  id: string;
  keys: string;
  description: string;
  action: () => void;
  category: string;
  enabled?: boolean;
}

interface KeyboardShortcutsProps {
  shortcuts?: Shortcut[];
  onShortcut?: (id: string) => void;
}

const DEFAULT_SHORTCUTS: Omit<Shortcut, 'action'>[] = [
  // Global
  { id: 'new-conversation', keys: 'Ctrl+N', description: 'New conversation', category: 'Global' },
  { id: 'save', keys: 'Ctrl+S', description: 'Save project', category: 'Global' },
  { id: 'settings', keys: 'Ctrl+,', description: 'Open settings', category: 'Global' },
  { id: 'command-palette', keys: 'Ctrl+K', description: 'Command palette', category: 'Global' },
  { id: 'fullscreen', keys: 'F11', description: 'Toggle fullscreen', category: 'Global' },
  
  // Chat
  { id: 'clear-chat', keys: 'Ctrl+L', description: 'Clear conversation', category: 'Chat' },
  { id: 'toggle-rag', keys: 'Ctrl+R', description: 'Toggle RAG', category: 'Chat' },
  
  // Processing
  { id: 'execute', keys: 'Ctrl+Enter', description: 'Execute function', category: 'Processing' },
  { id: 'search-functions', keys: 'Ctrl+F', description: 'Search functions', category: 'Processing' },
  
  // Map
  { id: 'reset-view', keys: 'Ctrl+0', description: 'Reset map view', category: 'Map' },
  { id: 'toggle-settings', keys: 'Ctrl+H', description: 'Toggle settings', category: 'Map' },
  { id: 'export', keys: 'Ctrl+E', description: 'Export map', category: 'Map' },
  
  // Navigation
  { id: 'go-chat', keys: 'Alt+1', description: 'Go to Chat', category: 'Navigation' },
  { id: 'go-processing', keys: 'Alt+2', description: 'Go to Processing', category: 'Navigation' },
  { id: 'go-projects', keys: 'Alt+3', description: 'Go to Projects', category: 'Navigation' },
  { id: 'go-settings', keys: 'Alt+4', description: 'Go to Settings', category: 'Navigation' },
];

export default function KeyboardShortcuts({ shortcuts, onShortcut }: KeyboardShortcutsProps) {
  const [showPalette, setShowPalette] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [showHelp, setShowHelp] = useState(false);

  // Parse key combination
  const parseKeys = (keys: string): { ctrl: boolean; alt: boolean; shift: boolean; key: string } => {
    const parts = keys.toLowerCase().split('+');
    return {
      ctrl: parts.includes('ctrl'),
      alt: parts.includes('alt'),
      shift: parts.includes('shift'),
      key: parts[parts.length - 1],
    };
  };

  // Check if key combination matches
  const matchesKeys = (e: KeyboardEvent, keys: string): boolean => {
    const parsed = parseKeys(keys);
    
    return (
      e.ctrlKey === parsed.ctrl &&
      e.altKey === parsed.alt &&
      e.shiftKey === parsed.shift &&
      e.key.toLowerCase() === parsed.key
    );
  };

  // Handle keyboard events
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Command palette (Ctrl+K)
      if (e.ctrlKey && e.key === 'k') {
        e.preventDefault();
        setShowPalette(prev => !prev);
        return;
      }

      // Help (Ctrl+/)
      if (e.ctrlKey && e.key === '/') {
        e.preventDefault();
        setShowHelp(prev => !prev);
        return;
      }

      // Close palette with Escape
      if (e.key === 'Escape') {
        setShowPalette(false);
        setShowHelp(false);
        return;
      }

      // Check shortcuts
      const allShortcuts = shortcuts || DEFAULT_SHORTCUTS;
      
      for (const shortcut of allShortcuts) {
        if (matchesKeys(e, shortcut.keys)) {
          e.preventDefault();
          
          if ('action' in shortcut && typeof shortcut.action === 'function') {
            shortcut.action();
          } else if (onShortcut) {
            onShortcut(shortcut.id);
          }
          
          break;
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [shortcuts, onShortcut]);

  // Filter shortcuts by search
  const filteredShortcuts = DEFAULT_SHORTCUTS.filter(s =>
    s.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
    s.category.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Group by category
  const groupedShortcuts = filteredShortcuts.reduce((acc, shortcut) => {
    if (!acc[shortcut.category]) {
      acc[shortcut.category] = [];
    }
    acc[shortcut.category].push(shortcut);
    return acc;
  }, {} as Record<string, typeof filteredShortcuts>);

  return (
    <>
      {/* Command Palette */}
      {showPalette && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-start justify-center pt-20">
          <div className="bg-white rounded-lg shadow-2xl w-full max-w-2xl overflow-hidden">
            {/* Search */}
            <div className="p-4 border-b">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="text"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  placeholder="Search commands..."
                  className="w-full pl-10 pr-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  autoFocus
                />
              </div>
            </div>

            {/* Results */}
            <div className="max-h-96 overflow-y-auto">
              {Object.entries(groupedShortcuts).map(([category, items]) => (
                <div key={category} className="border-b last:border-b-0">
                  <div className="px-4 py-2 bg-gray-50 text-sm font-semibold text-gray-600">
                    {category}
                  </div>
                  {items.map(shortcut => (
                    <button
                      key={shortcut.id}
                      onClick={() => {
                        onShortcut?.(shortcut.id);
                        setShowPalette(false);
                        setSearchTerm('');
                      }}
                      className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition"
                    >
                      <span className="text-gray-700">{shortcut.description}</span>
                      <kbd className="px-2 py-1 bg-gray-200 rounded text-xs font-mono">
                        {shortcut.keys}
                      </kbd>
                    </button>
                  ))}
                </div>
              ))}
            </div>

            {/* Footer */}
            <div className="px-4 py-3 bg-gray-50 text-xs text-gray-500 flex items-center justify-between">
              <span>Press ESC to close</span>
              <span>Ctrl+K to toggle</span>
            </div>
          </div>
        </div>
      )}

      {/* Help Modal */}
      {showHelp && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg shadow-2xl w-full max-w-4xl max-h-[80vh] overflow-hidden flex flex-col">
            {/* Header */}
            <div className="p-6 border-b flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Keyboard className="w-6 h-6 text-blue-500" />
                <h2 className="text-2xl font-bold">Keyboard Shortcuts</h2>
              </div>
              <button
                onClick={() => setShowHelp(false)}
                className="text-gray-500 hover:text-gray-700"
              >
                ✕
              </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {Object.entries(
                  DEFAULT_SHORTCUTS.reduce((acc, s) => {
                    if (!acc[s.category]) acc[s.category] = [];
                    acc[s.category].push(s);
                    return acc;
                  }, {} as Record<string, typeof DEFAULT_SHORTCUTS>)
                ).map(([category, items]) => (
                  <div key={category}>
                    <h3 className="text-lg font-semibold mb-3 text-gray-800">
                      {category}
                    </h3>
                    <div className="space-y-2">
                      {items.map(shortcut => (
                        <div
                          key={shortcut.id}
                          className="flex items-center justify-between py-2"
                        >
                          <span className="text-gray-700">{shortcut.description}</span>
                          <kbd className="px-3 py-1.5 bg-gray-100 border border-gray-300 rounded text-sm font-mono">
                            {shortcut.keys}
                          </kbd>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Footer */}
            <div className="p-4 border-t bg-gray-50 text-sm text-gray-600">
              <p>Press <kbd className="px-2 py-1 bg-gray-200 rounded">Ctrl+/</kbd> anytime to show this help</p>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

// Hook for using shortcuts in components
export function useKeyboardShortcut(keys: string, callback: () => void, deps: any[] = []) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const parts = keys.toLowerCase().split('+');
      const ctrl = parts.includes('ctrl');
      const alt = parts.includes('alt');
      const shift = parts.includes('shift');
      const key = parts[parts.length - 1];

      if (
        e.ctrlKey === ctrl &&
        e.altKey === alt &&
        e.shiftKey === shift &&
        e.key.toLowerCase() === key
      ) {
        e.preventDefault();
        callback();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, deps);
}
