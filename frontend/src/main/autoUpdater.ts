/**
 * Auto-Updater Configuration
 * Handles automatic updates using electron-updater
 */

import { autoUpdater } from 'electron-updater';
import { app, BrowserWindow, ipcMain } from 'electron';
import log from 'electron-log';

// Configure logging
log.transports.file.level = 'info';
autoUpdater.logger = log;

export class AutoUpdater {
  private mainWindow: BrowserWindow | null = null;
  private updateCheckInterval: NodeJS.Timeout | null = null;

  constructor(mainWindow: BrowserWindow) {
    this.mainWindow = mainWindow;
    this.setupAutoUpdater();
    this.setupIPC();
  }

  private setupAutoUpdater() {
    // Configure update server
    autoUpdater.setFeedURL({
      provider: 'github',
      owner: 'yourusername',
      repo: 'geobot',
      private: false,
    });

    // Auto-download updates
    autoUpdater.autoDownload = false;
    autoUpdater.autoInstallOnAppQuit = true;

    // Events
    autoUpdater.on('checking-for-update', () => {
      log.info('Checking for updates...');
      this.sendToRenderer('update-checking');
    });

    autoUpdater.on('update-available', (info) => {
      log.info('Update available:', info.version);
      this.sendToRenderer('update-available', {
        version: info.version,
        releaseNotes: info.releaseNotes,
        releaseDate: info.releaseDate,
      });
    });

    autoUpdater.on('update-not-available', (info) => {
      log.info('Update not available');
      this.sendToRenderer('update-not-available', {
        version: info.version,
      });
    });

    autoUpdater.on('error', (err) => {
      log.error('Update error:', err);
      this.sendToRenderer('update-error', {
        message: err.message,
      });
    });

    autoUpdater.on('download-progress', (progressObj) => {
      log.info(`Download progress: ${progressObj.percent}%`);
      this.sendToRenderer('update-download-progress', {
        percent: progressObj.percent,
        bytesPerSecond: progressObj.bytesPerSecond,
        transferred: progressObj.transferred,
        total: progressObj.total,
      });
    });

    autoUpdater.on('update-downloaded', (info) => {
      log.info('Update downloaded:', info.version);
      this.sendToRenderer('update-downloaded', {
        version: info.version,
        releaseNotes: info.releaseNotes,
      });
    });
  }

  private setupIPC() {
    // Check for updates manually
    ipcMain.handle('check-for-updates', async () => {
      try {
        const result = await autoUpdater.checkForUpdates();
        return {
          updateInfo: result?.updateInfo,
          cancellationToken: result?.cancellationToken,
        };
      } catch (error: any) {
        log.error('Error checking for updates:', error);
        return { error: error.message };
      }
    });

    // Download update
    ipcMain.handle('download-update', async () => {
      try {
        await autoUpdater.downloadUpdate();
        return { success: true };
      } catch (error: any) {
        log.error('Error downloading update:', error);
        return { error: error.message };
      }
    });

    // Install update and restart
    ipcMain.handle('quit-and-install', () => {
      autoUpdater.quitAndInstall(false, true);
    });

    // Get current version
    ipcMain.handle('get-app-version', () => {
      return app.getVersion();
    });

    // Enable/disable auto-update
    ipcMain.handle('set-auto-update', (event, enabled: boolean) => {
      if (enabled) {
        this.startPeriodicCheck();
      } else {
        this.stopPeriodicCheck();
      }
      return { success: true };
    });
  }

  private sendToRenderer(channel: string, data?: any) {
    if (this.mainWindow && !this.mainWindow.isDestroyed()) {
      this.mainWindow.webContents.send(channel, data);
    }
  }

  public checkForUpdates() {
    autoUpdater.checkForUpdates().catch((err) => {
      log.error('Error checking for updates:', err);
    });
  }

  public startPeriodicCheck(intervalHours: number = 6) {
    // Check immediately
    this.checkForUpdates();

    // Then check every N hours
    this.updateCheckInterval = setInterval(() => {
      this.checkForUpdates();
    }, intervalHours * 60 * 60 * 1000);

    log.info(`Auto-update check scheduled every ${intervalHours} hours`);
  }

  public stopPeriodicCheck() {
    if (this.updateCheckInterval) {
      clearInterval(this.updateCheckInterval);
      this.updateCheckInterval = null;
      log.info('Auto-update check stopped');
    }
  }

  public downloadUpdate() {
    return autoUpdater.downloadUpdate();
  }

  public quitAndInstall() {
    autoUpdater.quitAndInstall(false, true);
  }
}
