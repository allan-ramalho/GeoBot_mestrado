/**
 * API Client
 * Axios instance configured for backend communication
 */

import axios from 'axios';

const baseURL = import.meta.env.DEV 
  ? 'http://localhost:8000/api/v1'
  : `${window.location.origin}/api/v1`;

export const apiClient = axios.create({
  baseURL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if needed
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);
