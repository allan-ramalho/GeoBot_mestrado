# ✅ CHECKLIST DE VERIFICAÇÃO - GEOBOT

Use este checklist para verificar se tudo está funcionando corretamente.

## 🔧 Setup Inicial

### 1. Ambiente de Desenvolvimento

- [ ] Python 3.11.9 instalado
  ```powershell
  python --version
  # Deve mostrar: Python 3.11.9
  ```

- [ ] Node.js 18+ instalado
  ```powershell
  node --version
  # Deve mostrar: v18.x.x ou superior
  ```

- [ ] Git instalado (opcional)
  ```powershell
  git --version
  ```

### 2. Backend Setup

- [ ] Ambiente virtual criado
  ```powershell
  cd backend
  python -m venv venv
  ```

- [ ] Ambiente virtual ativado
  ```powershell
  venv\Scripts\activate  # Windows
  # Prompt deve mostrar (venv)
  ```

- [ ] Dependências instaladas
  ```powershell
  pip install -r requirements.txt
  # Verificar sem erros
  ```

- [ ] Arquivo .env criado
  ```powershell
  # Copiar de .env.example
  # Configurar minimamente HOST e PORT
  ```

- [ ] Backend inicia sem erros
  ```powershell
  uvicorn app.main:app --reload
  # Ver: "Application startup complete"
  ```

- [ ] Health check funciona
  ```
  Abrir: http://localhost:8000/health
  Deve retornar: {"status": "healthy", ...}
  ```

- [ ] Docs API acessível
  ```
  Abrir: http://localhost:8000/api/docs
  Deve mostrar interface Swagger
  ```

### 3. Frontend Setup

- [ ] Dependências instaladas
  ```powershell
  cd frontend
  npm install
  # Aguardar sem erros
  ```

- [ ] Development server inicia
  ```powershell
  npm run dev
  # Ver: "Local: http://localhost:5173"
  ```

- [ ] Frontend acessível
  ```
  Abrir: http://localhost:5173
  Deve mostrar tela de loading ou setup
  ```

## 🧪 Testes Funcionais

### Backend Tests

- [ ] **Teste 1: API Root**
  ```
  GET http://localhost:8000/
  Espera: {"service": "GeoBot API", ...}
  ```

- [ ] **Teste 2: Health Check**
  ```
  GET http://localhost:8000/health
  Espera: {"status": "healthy"}
  ```

- [ ] **Teste 3: List Providers**
  ```
  GET http://localhost:8000/api/v1/ai/providers
  Espera: ["groq", "openai", "claude", "gemini"]
  ```

- [ ] **Teste 4: System Config**
  ```
  GET http://localhost:8000/api/v1/config/system
  Espera: {"theme": "dark", ...}
  ```

- [ ] **Teste 5: List Functions**
  ```
  GET http://localhost:8000/api/v1/processing/functions
  Espera: {"functions": [...]} com 5+ funções
  ```

### Frontend Tests

- [ ] **Teste 1: Setup Page**
  - Acessar http://localhost:5173
  - Ver tela de configuração com branding
  - Botões de providers clicáveis

- [ ] **Teste 2: Navigation**
  - Após configurar, sidebar deve aparecer
  - Links devem mudar de cor ao clicar
  - Páginas devem carregar

- [ ] **Teste 3: Theme**
  - Verificar dark theme por padrão
  - Cores consistentes

## 🤖 Testes com AI (Requer API Key)

### Configuração AI

- [ ] **Groq Setup**
  1. Obter API key em: https://console.groq.com
  2. Na UI, selecionar Groq
  3. Inserir API key
  4. Deve listar modelos disponíveis
  5. Selecionar modelo
  6. Configuração deve salvar

- [ ] **OpenAI Setup** (opcional)
  1. API key de https://platform.openai.com
  2. Testar listagem de modelos
  3. Configurar e salvar

### Chat Tests (após configurar)

- [ ] **Teste 1: Mensagem Simples**
  ```
  Input: "Olá"
  Espera: Resposta do AI
  ```

- [ ] **Teste 2: Pergunta Geofísica**
  ```
  Input: "O que é redução ao polo?"
  Espera: Explicação técnica
  ```

- [ ] **Teste 3: Listar Funções**
  ```
  Input: "Quais funções de processamento estão disponíveis?"
  Espera: Lista com 5 funções magnéticas
  ```

## 📊 Testes de Processing

- [ ] **Function Registry**
  ```powershell
  # No Python
  from app.services.geophysics.function_registry import get_registry
  registry = get_registry()
  functions = registry.list_functions()
  # Deve ter 5+ funções
  ```

- [ ] **Semantic Search**
  ```powershell
  # Testar busca
  results = await registry.search_functions("redução ao polo")
  # Deve encontrar reduction_to_pole
  ```

## 🗄️ Testes de Storage

- [ ] **Project Creation**
  ```
  POST http://localhost:8000/api/v1/projects/create
  Body: {"name": "Test Project", "project_type": "magnetic"}
  Espera: Project metadata com ID
  ```

- [ ] **List Projects**
  ```
  GET http://localhost:8000/api/v1/projects/list
  Espera: {"projects": [...]}
  ```

- [ ] **File Upload** (opcional)
  ```
  POST http://localhost:8000/api/v1/data/upload
  Form-data: file + project_id
  Espera: File metadata
  ```

## 🔌 Testes Electron (Production-like)

- [ ] **Electron Dev Mode**
  ```powershell
  cd frontend
  npm run electron:dev
  # Deve abrir janela do Electron
  # Backend deve iniciar automaticamente
  ```

- [ ] **Backend Auto-start**
  - Verificar console Electron
  - Ver mensagem "Backend server is ready"

- [ ] **Window Features**
  - Redimensionar janela
  - Minimizar/Maximizar
  - Fechar (backend deve parar)

## 🐛 Troubleshooting Common Issues

### Backend não inicia

```powershell
# 1. Verificar Python version
python --version

# 2. Verificar venv ativado
# Prompt deve mostrar (venv)

# 3. Reinstalar dependências
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall

# 4. Verificar porta 8000 livre
netstat -ano | findstr :8000
# Se ocupada, mudar PORT no .env
```

### Frontend não conecta

```powershell
# 1. Backend rodando?
# Abrir http://localhost:8000/health

# 2. Verificar CORS no backend
# Deve ter localhost:5173 em ALLOWED_ORIGINS

# 3. Limpar cache
npm run dev -- --force

# 4. Verificar console do navegador
# F12 → Console → Ver erros
```

### AI não funciona

```powershell
# 1. API Key válida?
# Testar no site do provider

# 2. Backend recebe requests?
# Ver logs do uvicorn

# 3. Provider acessível?
# Verificar firewall/proxy

# 4. Tentar outro provider
# Groq é mais confiável para começar
```

### Electron não abre

```powershell
# 1. Frontend buildado?
cd frontend
npm run build

# 2. Backend empacotado?
# Verificar backend/ está completo

# 3. Ver logs Electron
# Abrir DevTools no Electron
```

## ✅ Checklist Final de Produção

Antes de considerar completo:

- [ ] Backend inicia sem warnings
- [ ] Frontend compila sem erros
- [ ] Todos os endpoints respondem
- [ ] UI responsiva (resize window)
- [ ] Sem erros no console
- [ ] Logs aparecem corretamente
- [ ] Configuração AI funciona
- [ ] Pelo menos 1 provider configurado
- [ ] Chat responde
- [ ] Projects podem ser criados
- [ ] Documentação revisada

## 📋 Próximos Testes (Futuro)

Quando implementar:

- [ ] RAG: Testar busca de documentos
- [ ] Processing: Executar funções com dados reais
- [ ] Workflow: Testar encadeamento
- [ ] Map viewer: Visualização Plotly
- [ ] File upload: Upload e parsing
- [ ] Export: Exportar resultados
- [ ] Batch processing: Múltiplos jobs
- [ ] WebSocket: Chat em tempo real

## 🎯 Critérios de Sucesso

### MVP (Minimum Viable Product)
- ✅ Backend responde
- ✅ Frontend carrega
- ✅ AI configurável
- ✅ Chat básico funciona
- ✅ Funções registradas

### Beta
- [ ] RAG implementado
- [ ] 10+ funções de processamento
- [ ] UI completa
- [ ] Map viewer funcional
- [ ] Processing via chat

### Production
- [ ] Empacotado como .exe
- [ ] Instalador funcional
- [ ] Zero dependências externas
- [ ] Documentação completa
- [ ] Testes automatizados

---

**Use este checklist progressivamente ao desenvolver cada fase!**
