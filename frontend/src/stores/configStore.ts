/**
 * Configuration Store
 * Manages AI provider and system configuration
 */

import { create } from 'zustand';
import { apiClient } from '@/services/api';

interface AIConfig {
  provider: string;
  model: string;
  apiKey: string;
  temperature: number;
}

interface ConfigState {
  isConfigured: boolean;
  aiConfig: AIConfig | null;
  availableProviders: string[];
  availableModels: any[];
  
  // Actions
  checkConfiguration: () => Promise<void>;
  configureAI: (config: AIConfig) => Promise<void>;
  listModels: (provider: string, apiKey: string) => Promise<any[]>;
  clearConfiguration: () => Promise<void>;
}

export const useConfigStore = create<ConfigState>((set) => ({
  isConfigured: false,
  aiConfig: null,
  availableProviders: ['groq', 'openai', 'claude', 'gemini'],
  availableModels: [],

  checkConfiguration: async () => {
    try {
      const response = await apiClient.get('/ai/config/current');
      const configured = response.data.configured !== false;
      
      set({ 
        isConfigured: configured,
        aiConfig: configured ? response.data : null
      });
    } catch (error) {
      console.error('Failed to check configuration:', error);
      set({ isConfigured: false });
    }
  },

  configureAI: async (config: AIConfig) => {
    try {
      await apiClient.post('/ai/providers/configure', config);
      
      set({ 
        isConfigured: true,
        aiConfig: config
      });
    } catch (error) {
      console.error('Failed to configure AI:', error);
      throw error;
    }
  },

  listModels: async (provider: string, apiKey: string) => {
    try {
      const response = await apiClient.get(`/ai/providers/${provider}/models`, {
        params: { api_key: apiKey }
      });
      
      set({ availableModels: response.data });
      return response.data;
    } catch (error) {
      console.error('Failed to list models:', error);
      throw error;
    }
  },

  clearConfiguration: async () => {
    try {
      await apiClient.delete('/ai/config');
      set({ 
        isConfigured: false,
        aiConfig: null
      });
    } catch (error) {
      console.error('Failed to clear configuration:', error);
      throw error;
    }
  },
}));
