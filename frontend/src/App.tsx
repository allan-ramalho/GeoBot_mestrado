import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { useEffect } from 'react';
import { useAppStore } from '@/stores/appStore';
import { useConfigStore } from '@/stores/configStore';

// Pages
import SetupPage from '@/pages/SetupPage';
import MainLayout from '@/pages/MainLayout';
import ProjectsPage from '@/pages/ProjectsPage';
import MapViewPage from '@/pages/MapViewPage';
import ProcessingPage from '@/pages/ProcessingPage';
import ChatPage from '@/pages/ChatPage';

// Components
import LoadingScreen from '@/components/LoadingScreen';

import './index.css';

function App() {
  const { isLoading, initialize } = useAppStore();
  const { isConfigured } = useConfigStore();

  useEffect(() => {
    initialize();
  }, []);

  if (isLoading) {
    return <LoadingScreen />;
  }

  return (
    <BrowserRouter>
      <Routes>
        {/* Setup route - shown when AI not configured */}
        {!isConfigured && (
          <Route path="*" element={<SetupPage />} />
        )}

        {/* Main application routes */}
        {isConfigured && (
          <Route path="/" element={<MainLayout />}>
            <Route index element={<Navigate to="/projects" replace />} />
            <Route path="projects" element={<ProjectsPage />} />
            <Route path="map" element={<MapViewPage />} />
            <Route path="processing" element={<ProcessingPage />} />
            <Route path="chat" element={<ChatPage />} />
          </Route>
        )}
      </Routes>
    </BrowserRouter>
  );
}

export default App;
