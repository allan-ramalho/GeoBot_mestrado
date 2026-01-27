# 🚀 Quick Start Guide - GeoBot

## Instalação Rápida

### Windows

1. **Requisitos**:
   - Python 3.11.9: https://www.python.org/downloads/
   - Node.js 18+: https://nodejs.org/

2. **Clone o repositório** (ou extraia o ZIP)

3. **Execute o script de setup**:
   ```powershell
   python scripts\setup_dev.py
   ```

4. **Configure o .env**:
   - Abra `backend\.env`
   - Configure suas credenciais (Supabase, etc.)

5. **Inicie o backend**:
   ```powershell
   cd backend
   venv\Scripts\activate
   uvicorn app.main:app --reload
   ```

6. **Em outro terminal, inicie o frontend**:
   ```powershell
   cd frontend
   npm install
   npm run dev
   ```

7. **Abra o navegador**: http://localhost:5173

### Linux/macOS

1. **Requisitos**: Python 3.11.9, Node.js 18+

2. **Clone e setup**:
   ```bash
   python scripts/setup_dev.py
   ```

3. **Configure .env**: `backend/.env`

4. **Inicie backend**:
   ```bash
   cd backend
   source venv/bin/activate
   uvicorn app.main:app --reload
   ```

5. **Inicie frontend** (novo terminal):
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

6. **Abra**: http://localhost:5173

## Primeira Execução

1. **Tela de Configuração**:
   - Escolha um AI Provider (Groq recomendado para começar)
   - Insira sua API Key
   - Selecione um modelo

2. **Configurar Supabase** (opcional para RAG):
   - Crie projeto em https://supabase.com
   - Execute `scripts/supabase_setup.sql` no SQL Editor
   - Configure URL e KEY no .env

3. **Upload de dados**:
   - Crie um projeto
   - Faça upload de arquivos XYZ/CSV

4. **Teste o chat**:
   - "Olá, explique o que é redução ao polo"
   - "Liste as funções disponíveis"

## Desenvolvimento

Ver [DEVELOPMENT.md](docs/DEVELOPMENT.md) para guia completo.

## Problemas?

Ver [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

## Documentação

- [Arquitetura](docs/ARCHITECTURE.md)
- [Roadmap](docs/ROADMAP.md)
- [Desenvolvimento](docs/DEVELOPMENT.md)
