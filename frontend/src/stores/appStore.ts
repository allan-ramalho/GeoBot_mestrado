/**
 * Global Application Store
 * Manages application-wide state
 */

import { create } from 'zustand';
import { apiClient } from '@/services/api';

interface AppState {
  isLoading: boolean;
  isInitialized: boolean;
  theme: 'light' | 'dark';
  backendUrl: string;
  backendHealthy: boolean;
  
  // Actions
  initialize: () => Promise<void>;
  setTheme: (theme: 'light' | 'dark') => void;
  checkBackendHealth: () => Promise<boolean>;
}

export const useAppStore = create<AppState>((set, get) => ({
  isLoading: true,
  isInitialized: false,
  theme: 'dark',
  backendUrl: 'http://localhost:8000',
  backendHealthy: false,

  initialize: async () => {
    try {
      // Get backend URL from Electron
      if (window.electron) {
        const url = await window.electron.getBackendUrl();
        set({ backendUrl: url });
      }

      // Check backend health
      const healthy = await get().checkBackendHealth();
      
      // Load system config
      const config = await apiClient.get('/config/system');
      set({ 
        theme: config.data.theme || 'dark',
        isInitialized: true,
        backendHealthy: healthy
      });

    } catch (error) {
      console.error('Failed to initialize app:', error);
    } finally {
      set({ isLoading: false });
    }
  },

  setTheme: (theme) => {
    set({ theme });
    document.documentElement.classList.toggle('dark', theme === 'dark');
    
    // Save to backend
    apiClient.put('/config/system', { theme }).catch(console.error);
  },

  checkBackendHealth: async () => {
    try {
      if (window.electron) {
        const healthy = await window.electron.checkBackendHealth();
        set({ backendHealthy: healthy });
        return healthy;
      }
      
      const response = await apiClient.get('/health');
      const healthy = response.status === 200;
      set({ backendHealthy: healthy });
      return healthy;
    } catch {
      set({ backendHealthy: false });
      return false;
    }
  },
}));
