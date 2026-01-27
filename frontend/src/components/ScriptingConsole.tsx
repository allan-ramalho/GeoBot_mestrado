/**
 * Scripting Console - Python REPL
 * Allows users to execute custom Python code
 */

import React, { useState, useRef, useEffect } from 'react';
import {
  Play,
  Trash2,
  Download,
  Upload,
  Code,
  Terminal,
  ChevronRight,
  AlertCircle,
  CheckCircle,
} from 'lucide-react';

interface ConsoleOutput {
  id: string;
  type: 'input' | 'output' | 'error';
  content: string;
  timestamp: Date;
}

interface ScriptingConsoleProps {
  onExecute?: (code: string) => Promise<any>;
}

export default function ScriptingConsole({ onExecute }: ScriptingConsoleProps) {
  const [code, setCode] = useState('');
  const [history, setHistory] = useState<ConsoleOutput[]>([]);
  const [isExecuting, setIsExecuting] = useState(false);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [commandHistory, setCommandHistory] = useState<string[]>([]);
  
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const outputRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [history]);

  // Add welcome message
  useEffect(() => {
    addOutput({
      type: 'output',
      content: 'GeoBot Python Console\nType Python code and press Ctrl+Enter to execute\nType "help()" for available functions',
    });
  }, []);

  const addOutput = (output: Omit<ConsoleOutput, 'id' | 'timestamp'>) => {
    setHistory(prev => [
      ...prev,
      {
        ...output,
        id: `${Date.now()}-${Math.random()}`,
        timestamp: new Date(),
      },
    ]);
  };

  const executeCode = async () => {
    if (!code.trim() || isExecuting) return;

    // Add to command history
    setCommandHistory(prev => [...prev, code]);
    setHistoryIndex(-1);

    // Add input to console
    addOutput({
      type: 'input',
      content: code,
    });

    setIsExecuting(true);

    try {
      // Execute code (call backend API)
      const result = await onExecute?.(code) || await mockExecute(code);

      // Add output
      if (result.success) {
        addOutput({
          type: 'output',
          content: result.output || 'Execution successful',
        });
      } else {
        addOutput({
          type: 'error',
          content: result.error || 'Execution failed',
        });
      }
    } catch (error: any) {
      addOutput({
        type: 'error',
        content: error.message || 'Unexpected error',
      });
    } finally {
      setIsExecuting(false);
      setCode('');
    }
  };

  // Mock execution for demo
  const mockExecute = async (code: string) => {
    await new Promise(resolve => setTimeout(resolve, 500));
    
    if (code.includes('print')) {
      const match = code.match(/print\((.*)\)/);
      return { success: true, output: eval(match?.[1] || '""') };
    }
    
    if (code.includes('help()')) {
      return {
        success: true,
        output: `Available functions:
  - process_data(data, function, params)
  - load_grid(filename)
  - save_grid(data, filename)
  - plot_map(data)
  - numpy as np
  - scipy as sp`,
      };
    }
    
    return { success: true, output: 'Code executed' };
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Ctrl+Enter to execute
    if (e.ctrlKey && e.key === 'Enter') {
      e.preventDefault();
      executeCode();
    }
    
    // Arrow up/down for history
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (historyIndex < commandHistory.length - 1) {
        const newIndex = historyIndex + 1;
        setHistoryIndex(newIndex);
        setCode(commandHistory[commandHistory.length - 1 - newIndex]);
      }
    }
    
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (historyIndex > 0) {
        const newIndex = historyIndex - 1;
        setHistoryIndex(newIndex);
        setCode(commandHistory[commandHistory.length - 1 - newIndex]);
      } else if (historyIndex === 0) {
        setHistoryIndex(-1);
        setCode('');
      }
    }
  };

  const clearConsole = () => {
    setHistory([]);
    addOutput({
      type: 'output',
      content: 'Console cleared',
    });
  };

  const exportHistory = () => {
    const text = history
      .map(h => {
        const prefix = h.type === 'input' ? '>>> ' : '';
        return `${prefix}${h.content}`;
      })
      .join('\n\n');
    
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'console-history.txt';
    a.click();
  };

  const loadScript = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.py,.txt';
    input.onchange = (e: any) => {
      const file = e.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          setCode(e.target?.result as string);
        };
        reader.readAsText(file);
      }
    };
    input.click();
  };

  return (
    <div className="h-full flex flex-col bg-gray-900 text-gray-100">
      {/* Header */}
      <div className="p-4 border-b border-gray-700 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Terminal className="w-5 h-5 text-green-400" />
          <h2 className="font-semibold">Python Console</h2>
        </div>
        
        <div className="flex gap-2">
          <button
            onClick={loadScript}
            className="p-2 hover:bg-gray-800 rounded transition"
            title="Load Script"
          >
            <Upload className="w-4 h-4" />
          </button>
          <button
            onClick={exportHistory}
            className="p-2 hover:bg-gray-800 rounded transition"
            title="Export History"
          >
            <Download className="w-4 h-4" />
          </button>
          <button
            onClick={clearConsole}
            className="p-2 hover:bg-gray-800 rounded transition"
            title="Clear Console"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Output Area */}
      <div
        ref={outputRef}
        className="flex-1 overflow-y-auto p-4 space-y-3 font-mono text-sm"
      >
        {history.map(item => (
          <div key={item.id} className="flex gap-2">
            {item.type === 'input' && (
              <>
                <ChevronRight className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <div className="text-green-400">{item.content}</div>
              </>
            )}
            
            {item.type === 'output' && (
              <>
                <CheckCircle className="w-4 h-4 text-blue-400 flex-shrink-0 mt-0.5" />
                <div className="text-gray-300 whitespace-pre-wrap">{item.content}</div>
              </>
            )}
            
            {item.type === 'error' && (
              <>
                <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
                <div className="text-red-400 whitespace-pre-wrap">{item.content}</div>
              </>
            )}
          </div>
        ))}
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-700 p-4">
        <div className="flex items-start gap-3">
          <Code className="w-5 h-5 text-green-400 mt-2 flex-shrink-0" />
          
          <div className="flex-1">
            <textarea
              ref={inputRef}
              value={code}
              onChange={(e) => setCode(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Enter Python code... (Ctrl+Enter to execute)"
              className="w-full bg-gray-800 text-gray-100 px-3 py-2 rounded border border-gray-700 focus:border-green-400 focus:outline-none font-mono text-sm resize-none"
              rows={3}
              disabled={isExecuting}
            />
            
            <div className="flex items-center justify-between mt-2 text-xs text-gray-500">
              <span>Ctrl+Enter to execute • ↑↓ for history</span>
              <button
                onClick={executeCode}
                disabled={!code.trim() || isExecuting}
                className="px-4 py-1.5 bg-green-500 text-white rounded hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed transition flex items-center gap-2"
              >
                <Play className="w-3 h-3" />
                {isExecuting ? 'Executing...' : 'Execute'}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Status Bar */}
      <div className="px-4 py-2 bg-gray-800 border-t border-gray-700 flex items-center justify-between text-xs text-gray-400">
        <span>Python 3.11 • GeoBot Environment</span>
        <span>{history.length} outputs</span>
      </div>
    </div>
  );
}
