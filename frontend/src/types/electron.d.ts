/**
 * TypeScript declarations for Electron API
 */

interface ElectronAPI {
  getBackendUrl: () => Promise<string>;
  checkBackendHealth: () => Promise<boolean>;
}

declare global {
  interface Window {
    electron?: ElectronAPI;
  }
}

export {};
