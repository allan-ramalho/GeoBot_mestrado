#!/usr/bin/env python3
"""
Package Complete Application
Creates installers for Windows and Linux
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
BACKEND_DIST = ROOT_DIR / "dist" / "geobot_backend"
INSTALLERS_DIR = ROOT_DIR / "installers"

def check_backend_built():
    """Check if backend is already built"""
    print("🔍 Checking backend build...")
    
    exe_name = "geobot_backend.exe" if sys.platform == "win32" else "geobot_backend"
    exe_path = BACKEND_DIST / exe_name
    
    if not exe_path.exists():
        print("❌ Backend not built. Run scripts/build_backend.py first")
        sys.exit(1)
    
    print(f"✅ Backend found: {exe_path}\n")


def update_electron_builder_config():
    """Update electron-builder config with backend bundle"""
    print("📝 Updating Electron Builder config...")
    
    package_json = FRONTEND_DIR / "package.json"
    
    with open(package_json, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Add electron-builder config if not exists
    if "build" not in config:
        config["build"] = {}
    
    build_config = config["build"]
    
    # Update build settings
    build_config.update({
        "appId": "com.geobot.app",
        "productName": "GeoBot",
        "artifactName": "${productName}-${version}-${os}-${arch}.${ext}",
        "directories": {
            "output": "dist"
        },
        "files": [
            "build/**/*",
            "node_modules/**/*",
            "package.json"
        ],
        "extraResources": [
            {
                "from": str(BACKEND_DIST),
                "to": "backend",
                "filter": ["**/*"]
            }
        ],
        "win": {
            "target": [
                {
                    "target": "nsis",
                    "arch": ["x64"]
                },
                {
                    "target": "portable",
                    "arch": ["x64"]
                }
            ],
            "icon": "public/icon.ico"
        },
        "nsis": {
            "oneClick": False,
            "allowToChangeInstallationDirectory": True,
            "createDesktopShortcut": True,
            "createStartMenuShortcut": True,
            "shortcutName": "GeoBot"
        },
        "linux": {
            "target": [
                "AppImage",
                "deb",
                "rpm"
            ],
            "category": "Science",
            "icon": "public/icon.png"
        },
        "deb": {
            "depends": ["libgtk-3-0", "libnotify4", "libnss3"]
        }
    })
    
    # Save updated package.json
    with open(package_json, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Config updated\n")


def build_electron_app():
    """Build Electron app with electron-builder"""
    print("🔨 Building Electron app...")
    
    # Install electron-builder if needed
    print("   Installing electron-builder...")
    subprocess.run(
        ["npm", "install", "--save-dev", "electron-builder"],
        cwd=FRONTEND_DIR,
        check=True
    )
    
    # Build
    print("   Running electron-builder...")
    cmd = ["npx", "electron-builder", "build", "--publish", "never"]
    
    if sys.platform == "win32":
        cmd.extend(["--win", "--x64"])
    else:
        cmd.extend(["--linux"])
    
    try:
        subprocess.run(
            cmd,
            cwd=FRONTEND_DIR,
            check=True
        )
        
        print("✅ Electron build successful\n")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Electron build failed: {e}")
        return False


def organize_installers():
    """Organize installers in dedicated folder"""
    print("📦 Organizing installers...")
    
    INSTALLERS_DIR.mkdir(exist_ok=True)
    
    dist_dir = FRONTEND_DIR / "dist"
    
    if not dist_dir.exists():
        print("❌ No installers found")
        return False
    
    # Copy installers
    installer_files = []
    for ext in ["*.exe", "*.msi", "*.AppImage", "*.deb", "*.rpm"]:
        installer_files.extend(dist_dir.glob(ext))
    
    if not installer_files:
        print("❌ No installer files found")
        return False
    
    for installer in installer_files:
        dest = INSTALLERS_DIR / installer.name
        shutil.copy2(installer, dest)
        print(f"   Copied: {installer.name}")
    
    print(f"✅ Installers in: {INSTALLERS_DIR}\n")
    return True


def create_checksums():
    """Create SHA256 checksums for installers"""
    print("🔐 Creating checksums...")
    
    import hashlib
    
    checksums_file = INSTALLERS_DIR / "checksums.txt"
    
    with open(checksums_file, 'w') as f:
        for installer in INSTALLERS_DIR.glob("*"):
            if installer.suffix in [".exe", ".msi", ".AppImage", ".deb", ".rpm"]:
                # Calculate SHA256
                sha256 = hashlib.sha256()
                with open(installer, 'rb') as file:
                    for chunk in iter(lambda: file.read(4096), b""):
                        sha256.update(chunk)
                
                checksum = sha256.hexdigest()
                f.write(f"{checksum}  {installer.name}\n")
                print(f"   {installer.name}: {checksum[:16]}...")
    
    print(f"✅ Checksums saved to {checksums_file}\n")


def create_readme():
    """Create installer README"""
    print("📄 Creating installer README...")
    
    readme_content = """
# GeoBot Installers

## Windows

### Option 1: NSIS Installer (Recommended)
- `GeoBot-x.x.x-win-x64.exe` - Full installer with uninstaller
- Creates desktop and start menu shortcuts
- Allows custom installation directory

### Option 2: Portable
- `GeoBot-x.x.x-win-x64-portable.exe` - No installation required
- Run directly from any location
- Includes all dependencies

## Linux

### Option 1: AppImage (Recommended)
- `GeoBot-x.x.x-linux-x86_64.AppImage` - Universal Linux binary
- No installation required
- Make executable: `chmod +x GeoBot-*.AppImage`
- Run: `./GeoBot-*.AppImage`

### Option 2: Debian/Ubuntu (.deb)
- `GeoBot_x.x.x_amd64.deb` - For Debian-based distributions
- Install: `sudo dpkg -i GeoBot_x.x.x_amd64.deb`
- Or: `sudo apt install ./GeoBot_x.x.x_amd64.deb`

### Option 3: Fedora/RHEL (.rpm)
- `GeoBot-x.x.x.x86_64.rpm` - For RPM-based distributions
- Install: `sudo rpm -i GeoBot-x.x.x.x86_64.rpm`
- Or: `sudo dnf install ./GeoBot-x.x.x.x86_64.rpm`

## System Requirements

- Windows 10/11 (64-bit) or Linux
- 4 GB RAM minimum (8 GB recommended)
- 1 GB free disk space
- Internet connection for AI features

## First Run

1. Launch GeoBot
2. Go to Settings
3. Enter API keys:
   - OpenAI API key (required for GPT)
   - Anthropic API key (optional for Claude)
   - Google API key (optional for Gemini)
   - Supabase credentials (optional for RAG)
4. Start using GeoBot!

## Verification

See `checksums.txt` for SHA256 checksums to verify installer integrity.

## Support

- Documentation: docs/USER_MANUAL.md
- Issues: https://github.com/yourusername/geobot/issues
- Email: support@geobot.com
"""
    
    readme_path = INSTALLERS_DIR / "README.txt"
    readme_path.write_text(readme_content.strip())
    
    print(f"✅ README created: {readme_path}\n")


def main():
    """Main packaging process"""
    print("=" * 60)
    print("  GeoBot Packaging Script")
    print("=" * 60)
    print()
    
    steps = [
        ("Check backend", check_backend_built),
        ("Update config", update_electron_builder_config),
        ("Build Electron", build_electron_app),
        ("Organize installers", organize_installers),
        ("Create checksums", create_checksums),
        ("Create README", create_readme),
    ]
    
    for step_name, step_func in steps:
        try:
            result = step_func()
            if result is False:
                print(f"\n❌ Packaging failed at step: {step_name}")
                sys.exit(1)
        except Exception as e:
            print(f"\n❌ Error in {step_name}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Success
    print("=" * 60)
    print("  ✅ PACKAGING SUCCESSFUL!")
    print("=" * 60)
    print(f"\nInstallers: {INSTALLERS_DIR}")
    print("\nDistribution files:")
    for file in INSTALLERS_DIR.glob("*"):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  - {file.name} ({size_mb:.1f} MB)")
    
    print("\nNext steps:")
    print("1. Test installers on target platforms")
    print("2. Upload to release page")
    print("3. Update documentation")
    print("4. Announce release!")


if __name__ == "__main__":
    main()
