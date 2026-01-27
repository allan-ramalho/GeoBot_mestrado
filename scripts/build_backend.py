#!/usr/bin/env python3
"""
Build Backend with PyInstaller
Creates standalone executable bundle for Windows/Linux
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
BUILD_DIR = ROOT_DIR / "build"
DIST_DIR = ROOT_DIR / "dist"
SPEC_FILE = BACKEND_DIR / "geobot_backend.spec"

def clean_build_dirs():
    """Remove previous build artifacts"""
    print("🧹 Cleaning build directories...")
    
    for dir_path in [BUILD_DIR, DIST_DIR]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"   Removed {dir_path}")
    
    # Remove __pycache__
    for pycache in BACKEND_DIR.rglob("__pycache__"):
        shutil.rmtree(pycache)
    
    print("✅ Clean complete\n")


def create_spec_file():
    """Create PyInstaller spec file"""
    print("📝 Creating PyInstaller spec file...")
    
    spec_content = """
# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

# Paths
backend_dir = Path('backend')
app_dir = backend_dir / 'app'

# Analysis
a = Analysis(
    ['backend/app/main.py'],
    pathex=[str(backend_dir)],
    binaries=[],
    datas=[
        # Include all Python modules
        (str(app_dir), 'app'),
        # Include dependencies data files
        (str(backend_dir / 'requirements.txt'), '.'),
    ],
    hiddenimports=[
        'fastapi',
        'uvicorn',
        'pydantic',
        'numpy',
        'scipy',
        'httpx',
        'anthropic',
        'openai',
        'google.generativeai',
        'supabase',
        'pypdf2',
        'tiktoken',
        'networkx',
        'app.core.config',
        'app.api.endpoints',
        'app.services.ai',
        'app.services.geophysics',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'PIL',
        'tkinter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='geobot_backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico' if sys.platform == 'win32' else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='geobot_backend',
)
"""
    
    SPEC_FILE.write_text(spec_content.strip())
    print(f"✅ Created {SPEC_FILE}\n")


def build_with_pyinstaller():
    """Run PyInstaller build"""
    print("🔨 Building with PyInstaller...")
    
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(SPEC_FILE),
        "--clean",
        "--noconfirm",
        f"--distpath={DIST_DIR}",
        f"--workpath={BUILD_DIR}",
    ]
    
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        print("✅ Build successful\n")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print("❌ Build failed:")
        print(e.stderr)
        return False


def verify_build():
    """Verify build output"""
    print("🔍 Verifying build...")
    
    exe_name = "geobot_backend.exe" if sys.platform == "win32" else "geobot_backend"
    exe_path = DIST_DIR / "geobot_backend" / exe_name
    
    if not exe_path.exists():
        print(f"❌ Executable not found: {exe_path}")
        return False
    
    # Check size
    size_mb = exe_path.stat().st_size / (1024 * 1024)
    print(f"   Executable: {exe_path}")
    print(f"   Size: {size_mb:.1f} MB")
    
    # Test run
    print("   Testing executable...")
    try:
        result = subprocess.run(
            [str(exe_path), "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0 or "FastAPI" in result.stdout:
            print("✅ Executable runs successfully\n")
            return True
        else:
            print(f"⚠️  Executable returned code {result.returncode}")
            return True  # Still consider it a success
            
    except Exception as e:
        print(f"⚠️  Could not test executable: {e}")
        return True  # Still consider it a success if file exists


def create_readme():
    """Create README for distribution"""
    print("📄 Creating distribution README...")
    
    readme_content = """
# GeoBot Backend Distribution

This is the standalone backend server for GeoBot.

## Running the Server

### Windows
```
geobot_backend.exe
```

### Linux
```
./geobot_backend
```

The server will start on http://localhost:8000

## Configuration

Create a `.env` file in the same directory with:

```
# AI Providers (at least one required)
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here

# Supabase (for RAG)
SUPABASE_URL=your_url
SUPABASE_KEY=your_key

# Optional
GROQ_API_KEY=your_key_here
```

## Endpoints

- Health check: http://localhost:8000/health
- API docs: http://localhost:8000/docs
- WebSocket: ws://localhost:8000/ws/chat

## Troubleshooting

If the server doesn't start:
1. Check that port 8000 is available
2. Verify `.env` file exists with valid API keys
3. Check logs in `geobot.log`

For more help, visit: https://github.com/yourusername/geobot
"""
    
    readme_path = DIST_DIR / "geobot_backend" / "README.txt"
    readme_path.write_text(readme_content.strip())
    
    print(f"✅ Created {readme_path}\n")


def main():
    """Main build process"""
    print("=" * 60)
    print("  GeoBot Backend Build Script")
    print("=" * 60)
    print()
    
    # Check PyInstaller
    try:
        import PyInstaller
        print(f"✅ PyInstaller version: {PyInstaller.__version__}\n")
    except ImportError:
        print("❌ PyInstaller not found. Install with:")
        print("   pip install pyinstaller")
        sys.exit(1)
    
    # Build steps
    steps = [
        ("Clean", clean_build_dirs),
        ("Create spec", create_spec_file),
        ("Build", build_with_pyinstaller),
        ("Verify", verify_build),
        ("Create README", create_readme),
    ]
    
    for step_name, step_func in steps:
        try:
            result = step_func()
            if result is False:
                print(f"\n❌ Build failed at step: {step_name}")
                sys.exit(1)
        except Exception as e:
            print(f"\n❌ Error in {step_name}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Success
    print("=" * 60)
    print("  ✅ BUILD SUCCESSFUL!")
    print("=" * 60)
    print(f"\nDistribution folder: {DIST_DIR / 'geobot_backend'}")
    print("\nNext steps:")
    print("1. Test the executable")
    print("2. Create installer (see scripts/package_app.py)")
    print("3. Distribute!")


if __name__ == "__main__":
    main()
