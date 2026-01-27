# 🔧 Troubleshooting Guide - GeoBot

## Quick Diagnostic Checklist

Before diving into specific issues, run this checklist:

- [ ] GeoBot is updated to latest version
- [ ] Backend server is running (check terminal/logs)
- [ ] API keys are configured correctly
- [ ] Internet connection is active (for AI features)
- [ ] Sufficient disk space (>1GB free)
- [ ] Antivirus/Firewall not blocking GeoBot
- [ ] Port 8000 is not in use by another application

---

## Installation Issues

### Windows: "Windows protected your PC"

**Symptoms**: SmartScreen blocks installation

**Solution**:
1. Click "More info"
2. Click "Run anyway"
3. Or: Right-click installer → Properties → Unblock → Apply

**Why**: GeoBot is not yet signed with Microsoft certificate (planned for v1.1)

### Linux: "Permission denied"

**Symptoms**: Cannot execute AppImage

**Solution**:
```bash
chmod +x GeoBot-*.AppImage
```

### Linux: Missing dependencies

**Symptoms**: AppImage fails to start

**Solution** (Ubuntu/Debian):
```bash
sudo apt install libgtk-3-0 libnotify4 libnss3 libxss1 libxtst6 xdg-utils libatspi2.0-0 libdrm2 libgbm1
```

**Solution** (Fedora):
```bash
sudo dnf install gtk3 libnotify nss libXScrnSaver libXtst xdg-utils at-spi2-atk libdrm mesa-libgbm
```

---

## Startup Issues

### Backend Server Won't Start

**Symptoms**: "Failed to connect to server" error

**Diagnosis**:
```powershell
# Windows
netstat -ano | findstr :8000

# Linux
lsof -i :8000
```

**Solutions**:

1. **Port Already in Use**
   - Kill process using port 8000
   - Or configure different port in settings

2. **Python Dependencies Missing**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Check Logs**
   - Windows: `C:\Users\[user]\AppData\Local\GeoBot\logs\backend.log`
   - Linux: `~/.local/share/GeoBot/logs/backend.log`

### Frontend Won't Load

**Symptoms**: Blank white screen

**Solutions**:

1. **Clear Cache**
   - Press `Ctrl + Shift + Del`
   - Clear browsing data
   - Restart GeoBot

2. **Reset Settings**
   - Delete: `AppData/Local/GeoBot/config.json`
   - Restart

3. **Check Console**
   - Press `F12` (Developer Tools)
   - Look for errors in Console tab
   - Screenshot and report if unknown error

---

## API Key Issues

### "Invalid API Key" Error

**Diagnosis Steps**:

1. **Verify Key Format**
   - OpenAI: starts with `sk-`
   - Anthropic: starts with `sk-ant-`
   - Google: alphanumeric, no prefix

2. **Test Key Manually**
   ```bash
   # OpenAI
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer YOUR_KEY"
   
   # Anthropic
   curl https://api.anthropic.com/v1/messages \
     -H "x-api-key: YOUR_KEY"
   ```

3. **Common Mistakes**
   - Extra spaces before/after key
   - Key expired or revoked
   - Quota exceeded (check billing)
   - Wrong key for wrong provider

**Solutions**:
- Regenerate key in provider console
- Copy-paste carefully (no line breaks)
- Verify billing/quota limits

### API Rate Limiting

**Symptoms**: "Rate limit exceeded" error

**Solutions**:

1. **Reduce Request Frequency**
   - Wait 60 seconds between messages
   - Use lower-tier model (GPT-3.5 instead of GPT-4)

2. **Upgrade Plan**
   - OpenAI: https://platform.openai.com/account/billing
   - Increase rate limits

3. **Use Alternative Provider**
   - Switch to Groq (no rate limits)
   - Or Anthropic/Google

---

## Processing Issues

### "Out of Memory" During Processing

**Symptoms**: Processing crashes or freezes

**Solutions**:

1. **Reduce Grid Size**
   ```python
   # Downsample before processing
   from scipy.ndimage import zoom
   z_small = zoom(z, 0.5)  # 50% size
   ```

2. **Increase Memory Limit**
   - Settings → Advanced → Memory Limit → 16GB

3. **Process in Tiles**
   - Split grid into 4 quadrants
   - Process separately
   - Merge results

4. **Close Other Applications**
   - Chrome/Firefox consume lots of RAM
   - Free up memory

### Processing Stuck at 0%

**Symptoms**: Job stays at 0% forever

**Diagnosis**:
- Check `backend.log` for errors
- Look for Python tracebacks

**Solutions**:

1. **Restart Backend**
   - Close GeoBot
   - Wait 10 seconds
   - Reopen

2. **Check Data Format**
   - Ensure x, y, z are numeric arrays
   - No NaN or Inf values
   - Grid is regular (not scattered points)

3. **Validate Parameters**
   - Check parameter ranges
   - Consult function documentation

### Results Look Wrong

**Symptoms**: Strange values, NaN, or unexpected patterns

**Common Causes**:

1. **Wrong Parameters**
   - Example: Negative altitude for upward continuation
   - **Solution**: Review documentation

2. **Data Issues**
   - Spikes or outliers
   - **Solution**: Apply median filter first

3. **Grid Resolution**
   - Too coarse for method
   - **Solution**: Interpolate to finer grid

4. **Coordinate System**
   - Mixed units (meters vs degrees)
   - **Solution**: Ensure consistent units

---

## Visualization Issues

### Map Not Displaying

**Symptoms**: Blank map or "No data" message

**Solutions**:

1. **Check Data Loaded**
   - Open browser console (F12)
   - Look for data array

2. **Verify Format**
   ```javascript
   // Expected format
   {
     x: [0, 1, 2, ...],
     y: [0, 1, 2, ...],
     z: [[v11, v12, ...], [v21, v22, ...], ...],
     nx: 100,
     ny: 100
   }
   ```

3. **Clear Browser Cache**
   - `Ctrl + F5` (hard refresh)

### Plotly Performance Issues

**Symptoms**: Map lags or freezes with large datasets

**Solutions**:

1. **Reduce Data Points**
   - Downsample: 1000×1000 max for smooth UX
   - Use heatmap instead of contour (faster)

2. **Disable Animations**
   - Settings → Visualization → Animations: Off

3. **Use WebGL**
   - Automatically enabled for >100k points

### Colors Look Wrong

**Symptoms**: Unexpected colormap or range

**Solutions**:

1. **Check Z-Range**
   - Settings → Z-Range → Auto or Manual
   - Adjust min/max

2. **Verify Colorscale**
   - Some scales are inverted (RdBu vs RdBu_r)
   - Use "Reverse" toggle

3. **Outliers Affecting Scale**
   - Remove outliers first
   - Or set manual Z-range

---

## Chat and RAG Issues

### Chat Not Responding

**Symptoms**: Message sent but no response

**Diagnosis**:
1. Check network tab (F12 → Network)
2. Look for WebSocket connection
3. Check for HTTP errors (401, 500, etc.)

**Solutions**:

1. **API Key Issue**
   - Verify key is valid
   - Check quota/billing

2. **Backend Crash**
   - Check `backend.log`
   - Restart GeoBot

3. **Network Timeout**
   - Increase timeout in settings
   - Check firewall rules

### RAG Not Working (No Citations)

**Symptoms**: No citations in responses even with RAG enabled

**Solutions**:

1. **Supabase Not Configured**
   - Go to Settings → RAG
   - Enter Supabase URL and Key

2. **No Documents Ingested**
   ```bash
   cd scripts
   python ingest_pdfs.py
   ```
   - Wait for completion (may take hours for many PDFs)

3. **Verify Documents in Database**
   ```sql
   SELECT COUNT(*) FROM documents;
   SELECT COUNT(*) FROM embeddings;
   ```

4. **Check Embedding Model**
   - Ensure E5-Large downloaded
   - Or use alternative model

### Slow Chat Responses

**Symptoms**: 10-30 second delays

**Causes and Solutions**:

1. **Large Context**
   - Limit conversation history to 10 messages
   - Start new conversation

2. **RAG Overhead**
   - Disable RAG for simple questions
   - RAG adds ~2-5s

3. **Model Choice**
   - GPT-4: Slow but best quality
   - GPT-3.5-turbo: Faster
   - Groq Llama 3: Fastest

---

## Project and File Issues

### Cannot Import Files

**Symptoms**: "Invalid file format" error

**Solutions**:

1. **Check File Extension**
   - Supported: `.xyz`, `.csv`, `.grd`, `.json`
   - Rename if needed

2. **Verify File Contents**
   ```
   # Valid XYZ format
   X        Y        Z
   0.0      0.0      12.5
   1.0      0.0      13.2
   ...
   ```

3. **Encoding Issues**
   - Save as UTF-8 (not UTF-16 or Latin-1)
   - Use standard delimiters (space, comma, tab)

### Project Won't Save

**Symptoms**: "Save failed" error

**Solutions**:

1. **Disk Space**
   - Check available space (need >1GB)
   - Clear cache if needed

2. **Permissions**
   - Windows: Run as administrator
   - Linux: Check write permissions

3. **File Path Too Long**
   - Windows has 260-char limit
   - Move project to shorter path

### Corrupted Project File

**Symptoms**: Cannot open `.geobot` file

**Solutions**:

1. **Extract Manually**
   ```bash
   # .geobot is a ZIP file
   unzip project.geobot -d project_extracted
   ```

2. **Recover from Backup**
   - Check: `AppData/Local/GeoBot/backups/`
   - Automatic backups every 6 hours

3. **Contact Support**
   - Send corrupted file
   - We'll attempt recovery

---

## Performance Optimization

### General Slowness

**Solutions**:

1. **Hardware Upgrade**
   - RAM: 16GB+ recommended
   - CPU: 4+ cores
   - Storage: SSD > HDD

2. **Software Optimization**
   - Close unnecessary applications
   - Update GPU drivers
   - Windows: Disable visual effects

3. **GeoBot Settings**
   - Reduce threads if overheating
   - Lower cache size
   - Disable animations

### High CPU Usage

**Normal**: Processing geophysical data is CPU-intensive

**Abnormal**: >90% CPU when idle

**Solutions**:
- Check for infinite loops in logs
- Kill and restart backend
- Update to latest version

### High Memory Usage

**Normal**: Up to configured limit (default 8GB)

**Abnormal**: Memory leak (usage keeps growing)

**Solutions**:
- Restart GeoBot every 4 hours (temporary)
- Clear cache regularly
- Report bug with memory snapshot

---

## Network Issues

### Firewall Blocking Connections

**Symptoms**: "Connection refused" or "Timeout"

**Solutions**:

1. **Windows Firewall**
   ```
   Settings → Privacy & Security → Windows Security
   → Firewall & network protection
   → Allow an app → GeoBot → Private & Public
   ```

2. **Third-Party Antivirus**
   - Add GeoBot to whitelist
   - Kaspersky, Norton, Avast, etc.

3. **Corporate Firewall**
   - Request IT to whitelist localhost:8000
   - Or use VPN

### Proxy Configuration

**Symptoms**: API calls fail in corporate network

**Solutions**:

1. **Set Environment Variables**
   ```bash
   export HTTP_PROXY=http://proxy.company.com:8080
   export HTTPS_PROXY=http://proxy.company.com:8080
   ```

2. **Configure in .env**
   ```
   HTTP_PROXY=http://proxy:8080
   HTTPS_PROXY=http://proxy:8080
   NO_PROXY=localhost,127.0.0.1
   ```

---

## Data Recovery

### Lost Unsaved Work

**Symptoms**: Crash before saving

**Recovery**:

1. **Auto-Save**
   - Check: `AppData/Local/GeoBot/autosave/`
   - Files saved every 5 minutes

2. **Temp Files**
   - Windows: `C:\Users\[user]\AppData\Local\Temp\geobot-*`
   - Linux: `/tmp/geobot-*`

3. **Cache**
   - Processing results cached
   - Settings → Advanced → View Cache

### Deleted Project by Mistake

**Recovery**:

1. **Recycle Bin**
   - Check Windows Recycle Bin or Linux Trash

2. **Backup Folder**
   - Check: `AppData/Local/GeoBot/backups/`

3. **File Recovery Tools**
   - Windows: Recuva, TestDisk
   - Linux: extundelete, photorec

---

## When All Else Fails

### Collect Diagnostic Info

```bash
# Generate diagnostic report
python scripts/diagnostic.py > diagnostic.txt
```

Includes:
- OS version
- GeoBot version
- Python version
- Installed packages
- Recent logs
- Configuration

### Reset to Factory Settings

**WARNING**: Deletes all settings and projects

```bash
# Windows
rd /s /q "%LOCALAPPDATA%\GeoBot"

# Linux
rm -rf ~/.local/share/GeoBot
```

### Contact Support

**Before contacting**:
- [ ] Read FAQ
- [ ] Try troubleshooting steps
- [ ] Collect diagnostic info

**Email**: support@geobot.com

**Include**:
- Diagnostic report
- Screenshots of error
- Steps to reproduce
- What you expected vs what happened

**Response time**: 24-48 hours

---

## Known Issues

### v1.0.0 Known Bugs

1. **Memory leak in 3D visualization**
   - Workaround: Use 2D plots
   - Fixed in: v1.0.1

2. **Slow Plotly rendering on Windows**
   - Workaround: Reduce data points
   - Investigating

3. **RAG occasionally returns irrelevant citations**
   - Improving search algorithm
   - Fixed in: v1.1.0

### Reporting New Bugs

**GitHub Issues**: https://github.com/yourusername/geobot/issues

**Template**:
```
**Bug Description**: Clear description

**Steps to Reproduce**:
1. Go to ...
2. Click on ...
3. See error

**Expected Behavior**: What should happen

**Screenshots**: Attach if applicable

**Environment**:
- OS: Windows 11 / Ubuntu 22.04
- GeoBot version: 1.0.0
- Browser: Chrome 120

**Logs**: Attach relevant logs
```

---

**Last Updated**: January 2026  
**Version**: 1.0.0
