#!/usr/bin/env python
"""
Script de instalação para desenvolvimento
Configura ambiente backend
"""

import subprocess
import sys
from pathlib import Path


def main():
    print("🔧 Configurando ambiente de desenvolvimento GeoBot...")
    
    backend_dir = Path(__file__).parent.parent / "backend"
    
    # Verificar Python version
    version = sys.version_info
    if version.major != 3 or version.minor != 11:
        print(f"⚠️  Aviso: Python 3.11.9 recomendado, você tem {version.major}.{version.minor}.{version.micro}")
        response = input("Continuar mesmo assim? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Criar venv
    print("\n📦 Criando ambiente virtual...")
    venv_path = backend_dir / "venv"
    if not venv_path.exists():
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
        print("✅ Ambiente virtual criado")
    else:
        print("✅ Ambiente virtual já existe")
    
    # Ativar venv e instalar dependências
    print("\n📥 Instalando dependências...")
    
    if sys.platform == "win32":
        pip_path = venv_path / "Scripts" / "pip.exe"
    else:
        pip_path = venv_path / "bin" / "pip"
    
    subprocess.run([
        str(pip_path),
        "install",
        "-r",
        str(backend_dir / "requirements.txt")
    ], check=True)
    
    print("✅ Dependências instaladas")
    
    # Criar .env se não existir
    env_file = backend_dir / ".env"
    env_example = backend_dir / ".env.example"
    
    if not env_file.exists() and env_example.exists():
        print("\n📝 Criando arquivo .env...")
        env_file.write_text(env_example.read_text())
        print("✅ Arquivo .env criado")
        print("⚠️  Lembre-se de configurar as variáveis no arquivo .env")
    
    print("\n" + "="*60)
    print("✨ Setup concluído!")
    print("="*60)
    print("\nPróximos passos:")
    print("1. Configure o arquivo backend/.env com suas credenciais")
    print("2. Ative o ambiente virtual:")
    if sys.platform == "win32":
        print("   backend\\venv\\Scripts\\activate")
    else:
        print("   source backend/venv/bin/activate")
    print("3. Inicie o backend:")
    print("   cd backend")
    print("   uvicorn app.main:app --reload")
    print("\nPara o frontend:")
    print("   cd frontend")
    print("   npm install")
    print("   npm run dev")
    print()


if __name__ == "__main__":
    main()
