/**
 * Electron Main Process
 * Manages application lifecycle, window creation, and backend integration
 */

const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let backendProcess;

// Backend server configuration
const BACKEND_PORT = 8000;
const BACKEND_HOST = '127.0.0.1';

/**
 * Create main application window
 */
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1024,
    minHeight: 768,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      preload: path.join(__dirname, 'preload.js'),
    },
    icon: path.join(__dirname, '../build/icon.png'),
    show: false, // Don't show until ready
  });

  // Load app
  if (process.env.NODE_ENV === 'development') {
    mainWindow.loadURL('http://localhost:5173');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
  }

  // Show window when ready
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

/**
 * Start FastAPI backend server
 */
function startBackend() {
  return new Promise((resolve, reject) => {
    const isDev = process.env.NODE_ENV === 'development';
    
    let backendPath;
    let pythonPath;

    if (isDev) {
      // Development: use local Python
      backendPath = path.join(__dirname, '../../backend');
      pythonPath = 'python'; // Assumes Python in PATH
    } else {
      // Production: use bundled Python
      backendPath = path.join(process.resourcesPath, 'backend');
      pythonPath = path.join(process.resourcesPath, 'backend', 'venv', 'Scripts', 'python.exe');
    }

    console.log('🚀 Starting backend server...');
    console.log('Backend path:', backendPath);

    backendProcess = spawn(
      pythonPath,
      [
        '-m',
        'uvicorn',
        'app.main:app',
        '--host',
        BACKEND_HOST,
        '--port',
        BACKEND_PORT.toString(),
      ],
      {
        cwd: backendPath,
        env: { ...process.env },
      }
    );

    backendProcess.stdout.on('data', (data) => {
      console.log(`[Backend] ${data.toString()}`);
    });

    backendProcess.stderr.on('data', (data) => {
      console.error(`[Backend Error] ${data.toString()}`);
    });

    backendProcess.on('error', (error) => {
      console.error('Failed to start backend:', error);
      reject(error);
    });

    // Wait for backend to be ready
    const maxAttempts = 30;
    let attempts = 0;

    const checkBackend = setInterval(async () => {
      attempts++;
      
      try {
        const response = await fetch(`http://${BACKEND_HOST}:${BACKEND_PORT}/health`);
        if (response.ok) {
          clearInterval(checkBackend);
          console.log('✅ Backend server is ready');
          resolve();
        }
      } catch (error) {
        if (attempts >= maxAttempts) {
          clearInterval(checkBackend);
          reject(new Error('Backend failed to start within timeout'));
        }
      }
    }, 1000);
  });
}

/**
 * Stop backend server
 */
function stopBackend() {
  if (backendProcess) {
    console.log('🛑 Stopping backend server...');
    backendProcess.kill();
    backendProcess = null;
  }
}

/**
 * Application ready event
 */
app.whenReady().then(async () => {
  try {
    // Start backend first
    await startBackend();
    
    // Create window
    createWindow();

    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
      }
    });
  } catch (error) {
    console.error('Failed to start application:', error);
    app.quit();
  }
});

/**
 * Application quit event
 */
app.on('window-all-closed', () => {
  stopBackend();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  stopBackend();
});

/**
 * IPC Handlers
 */

ipcMain.handle('get-backend-url', () => {
  return `http://${BACKEND_HOST}:${BACKEND_PORT}`;
});

ipcMain.handle('check-backend-health', async () => {
  try {
    const response = await fetch(`http://${BACKEND_HOST}:${BACKEND_PORT}/health`);
    return response.ok;
  } catch {
    return false;
  }
});
