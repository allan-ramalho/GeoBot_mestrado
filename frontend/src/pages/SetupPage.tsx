/**
 * Setup Page - AI Provider Configuration
 * Mandatory first-time setup screen
 */

import { useState, useEffect } from 'react';
import { useConfigStore } from '@/stores/configStore';

export default function SetupPage() {
  const { configureAI, listModels, availableProviders } = useConfigStore();
  
  const [step, setStep] = useState(1);
  const [provider, setProvider] = useState('groq');
  const [apiKey, setApiKey] = useState('');
  const [models, setModels] = useState<any[]>([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleProviderSelect = (selectedProvider: string) => {
    setProvider(selectedProvider);
    setStep(2);
  };

  const handleApiKeySubmit = async () => {
    if (!apiKey.trim()) {
      setError('API Key is required');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const modelList = await listModels(provider, apiKey);
      setModels(modelList);
      setStep(3);
    } catch (err: any) {
      setError(err.message || 'Failed to validate API key');
    } finally {
      setLoading(false);
    }
  };

  const handleComplete = async () => {
    if (!selectedModel) {
      setError('Please select a model');
      return;
    }

    setLoading(true);
    setError('');

    try {
      await configureAI({
        provider,
        apiKey,
        model: selectedModel,
        temperature: 0.7,
      });
      
      // Configuration successful, app will reload
    } catch (err: any) {
      setError(err.message || 'Configuration failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-background">
      {/* Left side - Branding */}
      <div className="hidden lg:flex lg:w-1/2 bg-primary items-center justify-center p-12">
        <div className="text-center text-primary-foreground">
          <h1 className="text-6xl font-bold mb-4">🌍 GeoBot</h1>
          <p className="text-xl mb-8">AI-Powered Geophysical Data Processing</p>
          <div className="space-y-4 text-left max-w-md">
            <div className="flex items-start space-x-3">
              <span className="text-2xl">🤖</span>
              <div>
                <h3 className="font-semibold">AI Assistant</h3>
                <p className="text-sm opacity-90">Natural language processing commands</p>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <span className="text-2xl">📚</span>
              <div>
                <h3 className="font-semibold">Scientific Literature</h3>
                <p className="text-sm opacity-90">RAG-powered knowledge base</p>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <span className="text-2xl">🗺️</span>
              <div>
                <h3 className="font-semibold">Advanced Processing</h3>
                <p className="text-sm opacity-90">Magnetic & gravity data analysis</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Right side - Configuration */}
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="w-full max-w-md">
          <div className="mb-8">
            <h2 className="text-3xl font-bold mb-2">Welcome to GeoBot</h2>
            <p className="text-muted-foreground">Let's configure your AI assistant</p>
          </div>

          {/* Step 1: Provider Selection */}
          {step === 1 && (
            <div className="space-y-4">
              <h3 className="text-xl font-semibold mb-4">1. Select AI Provider</h3>
              {availableProviders.map((p) => (
                <button
                  key={p}
                  onClick={() => handleProviderSelect(p)}
                  className="w-full p-4 border-2 border-border hover:border-primary rounded-lg text-left transition-colors"
                >
                  <div className="font-semibold capitalize">{p}</div>
                  <div className="text-sm text-muted-foreground">
                    {p === 'groq' && 'Fast inference with automatic fallback'}
                    {p === 'openai' && 'GPT-4 and GPT-3.5 models'}
                    {p === 'claude' && 'Anthropic Claude models'}
                    {p === 'gemini' && 'Google Gemini models'}
                  </div>
                </button>
              ))}
            </div>
          )}

          {/* Step 2: API Key */}
          {step === 2 && (
            <div className="space-y-4">
              <button
                onClick={() => setStep(1)}
                className="text-primary hover:underline mb-4"
              >
                ← Back
              </button>
              <h3 className="text-xl font-semibold mb-4">
                2. Enter {provider.charAt(0).toUpperCase() + provider.slice(1)} API Key
              </h3>
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="Enter your API key"
                className="w-full p-3 border border-border rounded-lg bg-background"
              />
              {error && <p className="text-destructive text-sm">{error}</p>}
              <button
                onClick={handleApiKeySubmit}
                disabled={loading || !apiKey.trim()}
                className="w-full p-3 bg-primary text-primary-foreground rounded-lg font-semibold hover:opacity-90 disabled:opacity-50"
              >
                {loading ? 'Validating...' : 'Continue'}
              </button>
            </div>
          )}

          {/* Step 3: Model Selection */}
          {step === 3 && (
            <div className="space-y-4">
              <button
                onClick={() => setStep(2)}
                className="text-primary hover:underline mb-4"
              >
                ← Back
              </button>
              <h3 className="text-xl font-semibold mb-4">3. Select Model</h3>
              {models.length === 0 ? (
                <p className="text-muted-foreground">No models available</p>
              ) : (
                <div className="space-y-2">
                  {models.map((model) => (
                    <button
                      key={model.id}
                      onClick={() => setSelectedModel(model.id)}
                      className={`w-full p-3 border-2 rounded-lg text-left transition-colors ${
                        selectedModel === model.id
                          ? 'border-primary bg-primary/10'
                          : 'border-border hover:border-primary'
                      }`}
                    >
                      <div className="font-semibold">{model.name}</div>
                      <div className="text-sm text-muted-foreground">
                        Context: {model.context_window.toLocaleString()} tokens
                      </div>
                    </button>
                  ))}
                </div>
              )}
              {error && <p className="text-destructive text-sm">{error}</p>}
              <button
                onClick={handleComplete}
                disabled={loading || !selectedModel}
                className="w-full p-3 bg-primary text-primary-foreground rounded-lg font-semibold hover:opacity-90 disabled:opacity-50"
              >
                {loading ? 'Configuring...' : 'Complete Setup'}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
