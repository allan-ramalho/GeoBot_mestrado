import { test, expect } from '@playwright/test';

test.describe('GeoBot E2E Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Launch Electron app (configure in playwright.config.ts)
    await page.goto('http://localhost:3000');
  });

  test.describe('Chat Interface', () => {
    test('should send message and receive response', async ({ page }) => {
      // Navigate to chat
      await page.click('text=Chat');
      
      // Type message
      await page.fill('textarea[placeholder*="message"]', 'What is reduction to pole?');
      
      // Send message
      await page.click('button:has-text("Send")');
      
      // Wait for response
      await page.waitForSelector('.assistant-message', { timeout: 10000 });
      
      // Check response exists
      const response = await page.textContent('.assistant-message');
      expect(response).toBeTruthy();
      expect(response?.length).toBeGreaterThan(10);
    });

    test('should toggle RAG', async ({ page }) => {
      await page.click('text=Chat');
      
      // Find and click RAG toggle
      const ragToggle = page.locator('text=Use RAG').first();
      await ragToggle.click();
      
      // Verify checked state
      const checkbox = page.locator('input[type="checkbox"]').first();
      await expect(checkbox).toBeChecked();
    });

    test('should display citations', async ({ page }) => {
      await page.click('text=Chat');
      
      // Enable RAG
      await page.click('text=Use RAG');
      
      // Send message
      await page.fill('textarea[placeholder*="message"]', 'magnetic anomaly interpretation');
      await page.click('button:has-text("Send")');
      
      // Wait for response with citations
      await page.waitForSelector('.citation', { timeout: 15000 });
      
      const citations = await page.locator('.citation').count();
      expect(citations).toBeGreaterThan(0);
    });

    test('should create new conversation', async ({ page }) => {
      await page.click('text=Chat');
      
      // Click new conversation button
      await page.click('button:has-text("New")');
      
      // Check conversation list updated
      await expect(page.locator('.conversation-item').first()).toBeVisible();
    });
  });

  test.describe('Processing Interface', () => {
    test('should select function and show parameters', async ({ page }) => {
      await page.click('text=Processing');
      
      // Select RTP function
      await page.click('text=Reduction to Pole');
      
      // Check parameters visible
      await expect(page.locator('text=Inclination')).toBeVisible();
      await expect(page.locator('text=Declination')).toBeVisible();
    });

    test('should search functions', async ({ page }) => {
      await page.click('text=Processing');
      
      // Search for "bouguer"
      await page.fill('input[placeholder*="Search"]', 'bouguer');
      
      // Check filtered results
      await expect(page.locator('text=Bouguer Correction')).toBeVisible();
      await expect(page.locator('text=Euler Deconvolution')).not.toBeVisible();
    });

    test('should add job to queue', async ({ page }) => {
      await page.click('text=Processing');
      
      // Select function
      await page.click('text=Upward Continuation');
      
      // Set parameter
      await page.fill('input[name="altitude"]', '1000');
      
      // Execute
      await page.click('button:has-text("Execute")');
      
      // Check job in queue
      await expect(page.locator('.processing-job').first()).toBeVisible();
    });

    test('should expand/collapse categories', async ({ page }) => {
      await page.click('text=Processing');
      
      // Collapse Gravity
      await page.click('button:has-text("Gravity")');
      
      // Check functions hidden
      await expect(page.locator('text=Bouguer Correction')).not.toBeVisible();
      
      // Expand again
      await page.click('button:has-text("Gravity")');
      await expect(page.locator('text=Bouguer Correction')).toBeVisible();
    });
  });

  test.describe('Project Management', () => {
    test('should display projects list', async ({ page }) => {
      await page.click('text=Projects');
      
      // Check projects visible
      await expect(page.locator('.project-card').first()).toBeVisible();
    });

    test('should select project and show files', async ({ page }) => {
      await page.click('text=Projects');
      
      // Click first project
      await page.click('.project-card').first();
      
      // Check file tree visible
      await expect(page.locator('.file-tree')).toBeVisible();
    });

    test('should search projects', async ({ page }) => {
      await page.click('text=Projects');
      
      // Search
      await page.fill('input[placeholder*="Search"]', 'Serra');
      
      // Check filtered results
      const projects = await page.locator('.project-card').count();
      expect(projects).toBeGreaterThan(0);
    });

    test('should filter by tags', async ({ page }) => {
      await page.click('text=Projects');
      
      // Click tag
      await page.click('button:has-text("magnetic")');
      
      // Check filtered
      await expect(page.locator('.project-card')).toBeVisible();
    });

    test('should expand folder in file tree', async ({ page }) => {
      await page.click('text=Projects');
      
      // Select project
      await page.locator('.project-card').first().click();
      
      // Click folder
      await page.click('button:has-text("Raw Data")');
      
      // Check files visible
      await expect(page.locator('text=.xyz')).toBeVisible();
    });

    test('should select file and show details', async ({ page }) => {
      await page.click('text=Projects');
      
      // Select project
      await page.locator('.project-card').first().click();
      
      // Click file
      await page.click('text=magnetic_survey.xyz');
      
      // Check details panel
      await expect(page.locator('text=Metadata')).toBeVisible();
      await expect(page.locator('text=Tags')).toBeVisible();
    });
  });

  test.describe('Map Viewer', () => {
    test('should toggle plot settings', async ({ page }) => {
      // Assuming map viewer is on home or processing page
      await page.goto('http://localhost:3000');
      
      // Click settings button
      await page.click('button[title="Settings"]');
      
      // Check settings panel visible
      await expect(page.locator('text=Plot Type')).toBeVisible();
      await expect(page.locator('text=Colorscale')).toBeVisible();
    });

    test('should change plot type', async ({ page }) => {
      await page.goto('http://localhost:3000');
      
      // Open settings
      await page.click('button[title="Settings"]');
      
      // Click filled contour
      await page.click('button[title="Filled Contour"]');
      
      // Map should update (check for Plotly)
      await expect(page.locator('.plotly')).toBeVisible();
    });

    test('should change colorscale', async ({ page }) => {
      await page.goto('http://localhost:3000');
      
      // Open settings
      await page.click('button[title="Settings"]');
      
      // Change colorscale
      await page.selectOption('select[aria-label*="Colorscale"]', 'Plasma');
      
      // Map should update
      await page.waitForTimeout(500);
      await expect(page.locator('.plotly')).toBeVisible();
    });

    test('should toggle fullscreen', async ({ page }) => {
      await page.goto('http://localhost:3000');
      
      // Click fullscreen button
      await page.click('button[title="Fullscreen"]');
      
      // Check fullscreen class
      const mapContainer = page.locator('.map-viewer');
      await expect(mapContainer).toHaveClass(/fullscreen/);
    });

    test('should reset view', async ({ page }) => {
      await page.goto('http://localhost:3000');
      
      // Click reset button
      await page.click('button[title="Reset View"]');
      
      // Map should be visible and reset
      await expect(page.locator('.plotly')).toBeVisible();
    });
  });

  test.describe('Settings', () => {
    test('should navigate to settings', async ({ page }) => {
      await page.click('text=Settings');
      
      // Check settings page
      await expect(page.locator('text=API Keys')).toBeVisible();
    });

    test('should save API key', async ({ page }) => {
      await page.click('text=Settings');
      
      // Enter API key
      await page.fill('input[name="openai_key"]', 'test-key-123');
      
      // Save
      await page.click('button:has-text("Save")');
      
      // Check success message
      await expect(page.locator('text=Saved')).toBeVisible({ timeout: 3000 });
    });

    test('should toggle theme', async ({ page }) => {
      await page.click('text=Settings');
      
      // Toggle dark mode
      await page.click('button:has-text("Dark")');
      
      // Check body class
      const body = page.locator('body');
      await expect(body).toHaveClass(/dark/);
    });
  });

  test.describe('Navigation', () => {
    test('should navigate between pages', async ({ page }) => {
      // Start on home
      await expect(page).toHaveURL(/localhost:3000/);
      
      // Go to Chat
      await page.click('text=Chat');
      await expect(page).toHaveURL(/chat/);
      
      // Go to Processing
      await page.click('text=Processing');
      await expect(page).toHaveURL(/processing/);
      
      // Go to Projects
      await page.click('text=Projects');
      await expect(page).toHaveURL(/projects/);
      
      // Go to Settings
      await page.click('text=Settings');
      await expect(page).toHaveURL(/settings/);
    });

    test('should have working sidebar', async ({ page }) => {
      // Check sidebar visible
      await expect(page.locator('nav')).toBeVisible();
      
      // Check all nav links
      await expect(page.locator('text=Chat')).toBeVisible();
      await expect(page.locator('text=Processing')).toBeVisible();
      await expect(page.locator('text=Projects')).toBeVisible();
    });
  });

  test.describe('Error Handling', () => {
    test('should handle network error gracefully', async ({ page }) => {
      await page.route('**/api/**', route => route.abort());
      
      await page.click('text=Chat');
      await page.fill('textarea', 'test message');
      await page.click('button:has-text("Send")');
      
      // Check error message
      await expect(page.locator('text=Error')).toBeVisible({ timeout: 5000 });
    });

    test('should show validation errors', async ({ page }) => {
      await page.click('text=Processing');
      
      // Select function
      await page.click('text=Reduction to Pole');
      
      // Enter invalid value
      await page.fill('input[name="inclination"]', '999');
      
      // Try to execute
      await page.click('button:has-text("Execute")');
      
      // Check error
      await expect(page.locator('text=Invalid')).toBeVisible({ timeout: 3000 });
    });
  });
});
