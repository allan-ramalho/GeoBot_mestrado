# 🌍 GeoBot - AI-Powered Geophysical Data Processing Platform

## Visão Geral

GeoBot é um software desktop profissional para processamento e interpretação de dados geofísicos potenciais (gravimetria e magnetometria) com integração profunda de Inteligência Artificial.

**Status**: ✅ **Fase 2 Completa** - AI Core totalmente funcional

## 🚀 Características Principais

### ✅ Implementado (Fase 1 + 2)

- **Interface Desktop Profissional**: Electron + React + TypeScript com UI moderna
- **Backend Científico Robusto**: FastAPI + Python 3.11.9 com async/await
- **AI Assistant com RAG**: 
  - ✅ Consulta literatura científica (PDFs)
  - ✅ Busca semântica com embeddings E5-Large (1024 dim)
  - ✅ Citações formatadas academicamente
  - ✅ Multi-idioma (PT-BR, EN, ES)
- **Multi-provider AI**: 
  - ✅ Groq (llama-3.3-70b, mixtral-8x7b)
  - ✅ OpenAI (GPT-4, GPT-3.5-turbo)
  - ✅ Claude 3 (Opus, Sonnet, Haiku)
  - ✅ Gemini Pro
  - ✅ Fallback automático entre modelos
- **Chat UI Moderna**:
  - ✅ Markdown rendering completo
  - ✅ Code highlighting (syntax highlighter)
  - ✅ Exibição de citações científicas
  - ✅ WebSocket streaming (respostas em tempo real)
- **Processamento Geofísico**: 
  - ✅ 5 funções magnéticas implementadas
  - ✅ Function calling via linguagem natural
  - ✅ Registro automático de funções
- **RAG System**:
  - ✅ PDF parser com chunking inteligente
  - ✅ Ingestão automatizada de documentos
  - ✅ Vector search com pgvector (Supabase)
- **Visualizações Interativas**: Framework Plotly configurado
- **Standalone**: Configuração Electron Builder para Windows/Linux

### 🚧 Em Desenvolvimento (Próximas Fases)

- Mais 25+ funções de processamento geofísico
- Map viewer interativo com Plotly
- Processing workflows encadeados
- Batch processing
- Análise estatística avançada

## 📁 Estrutura do Projeto

```
GeoBot_Mestrado/
├── backend/                    # FastAPI Backend
│   ├── app/
│   │   ├── api/               # Endpoints REST + WebSocket
│   │   │   └── endpoints/     # Chat, AI, Processing, Projects, Data
│   │   ├── core/              # Configurações e logging
│   │   ├── models/            # Modelos Pydantic
│   │   ├── services/          # Lógica de negócio
│   │   │   ├── ai/           # ✅ RAG Engine, PDF Parser, Chat Service
│   │   │   ├── geophysics/   # ✅ Function Registry, Processing Engine
│   │   │   └── storage/      # ✅ Project & Data Manager
│   │   └── main.py           # ✅ Entry point com lifespan
│   ├── requirements.txt       # ✅ 30+ dependências
│   └── .env.example
├── frontend/                   # React + TypeScript + Electron
│   ├── src/
│   │   ├── components/       # ✅ Sidebar, LoadingScreen
│   │   ├── pages/            # ✅ Setup, Chat, MainLayout
│   │   ├── stores/           # ✅ Zustand stores (app, config)
│   │   ├── services/         # ✅ Axios API client
│   │   └── App.tsx           # ✅ Router com guards
│   ├── electron/             # ✅ Main + Preload (IPC seguro)
│   ├── package.json          # ✅ Electron Builder config
│   └── tsconfig.json
├── docs/                       # ✅ 25,000+ palavras documentação
│   ├── ARCHITECTURE.md        # ✅ Arquitetura detalhada
│   ├── DEVELOPMENT.md         # ✅ Guia de desenvolvimento
│   └── ROADMAP.md             # ✅ Plano 6 fases
├── scripts/                    # Scripts utilitários
│   ├── setup_dev.py           # ✅ Setup automático ambiente
│   ├── ingest_pdfs.py         # ✅ Ingestão RAG completa
│   └── supabase_setup.sql     # ✅ Setup pgvector
├── FASE_2_COMPLETA.md         # ✅ Resumo Fase 2
├── GUIA_TESTES_FASE2.md       # ✅ Guia de testes
└── README.md                   # Este arquivo
```

**Estatísticas**:
- **80+ arquivos** criados
- **~9,000 linhas** de código
- **~30,000 palavras** de documentação
- **7 módulos** principais

## 🛠️ Tecnologias

### Backend
- **Python 3.11.9** (versão específica requerida)
- **FastAPI 0.109.0** - Framework async web
- **Supabase** (PostgreSQL + pgvector) - Vector database
- **sentence-transformers 2.3.1** - E5-Large embeddings
- **PyPDF2 3.0.1** - PDF parsing
- **NumPy, SciPy, Pandas** - Computação científica
- **AI SDKs**: groq, openai, anthropic, google-generativeai
- **httpx** - Cliente HTTP async
- **langdetect** - Detecção de idioma

### Frontend
- **TypeScript 5.3.3** - Type safety
- **React 18** - UI framework
- **Electron 28.1.3** - Desktop container
- **Vite 5.0.11** - Build tool ultra-rápido
- **Zustand 4.4.7** - State management leve
- **Tailwind CSS 3.4.1** - Utility-first CSS
- **Radix UI** - Componentes acessíveis
- **Plotly.js 2.28.0** - Visualizações interativas
- **react-markdown 9.0.1** - Markdown rendering
- **react-syntax-highlighter 15.5.0** - Code highlighting
- **Axios 1.6.5** - HTTP client

## 📦 Instalação

### Método Rápido (Recomendado)

```bash
# Clone o repositório
cd GeoBot_Mestrado

# Setup automático (cria venv, instala deps)
python scripts/setup_dev.py

# Frontend
cd ../frontend
npm install

# Executar em desenvolvimento
# Terminal 1 - Backend
cd backend
uvicorn app.main:app --reload

# Terminal 2 - Frontend
cd frontend
npm run dev
```

### Build Production

```bash
# Build completo
npm run build:all

# Gera executável standalone para Windows/Linux
```

## 🔧 Configuração Inicial

1. **Primeira Execução**: Tela de configuração obrigatória do AI Provider
2. **Selecione o Provider**: Groq, OpenAI, Claude, ou Gemini
3. **Insira API Key**: Suas credenciais
4. **Escolha o Modelo**: Lista automática de modelos disponíveis
5. **Configure Supabase**: Para RAG e armazenamento de dados

## 🤖 AI Assistant

O GeoBot Assistant combina:

- **RAG (Retrieval Augmented Generation)**: Consulta literatura científica em geofísica
- **Function Calling**: Executa processamentos via linguagem natural

### Exemplos de Uso

```
"Aplique redução ao polo e depois calcule o gradiente horizontal total"
"Quais são as melhores práticas para correção do terreno em gravimetria?"
"Mostre-me as anomalias magnéticas com amplitude superior a 100 nT"
"Crie um perfil gravimétrico na direção N-S"
```

## 📚 Documentação

- [Arquitetura Completa](docs/ARCHITECTURE.md)
- [Guia de Desenvolvimento](docs/DEVELOPMENT.md)
- [API Reference](docs/API.md)
- [Funções de Processamento](docs/PROCESSING_FUNCTIONS.md)
- [Guia de Extensão](docs/EXTENDING.md)

## 🎯 Roadmap

### Fase 1: Fundação ✅
- [x] Estrutura base do projeto
- [x] Backend FastAPI
- [x] Frontend React + Electron

### Fase 2: AI Core
- [ ] Implementação multi-provider
- [ ] RAG com Supabase + pgvector
- [ ] Function registry e execution engine

### Fase 3: Geophysics Engine
- [ ] Funções de processamento magnético
- [ ] Funções de processamento gravimétrico
- [ ] Pipeline de workflows

### Fase 4: UI/UX
- [ ] Interface profissional completa
- [ ] Visualizações interativas
- [ ] Project tree e data management

### Fase 5: Production
- [ ] Testes completos
- [ ] Empacotamento standalone
- [ ] Documentação final

## 👥 Contribuindo

Este é um projeto acadêmico de mestrado. Contribuições são bem-vindas seguindo as diretrizes em [CONTRIBUTING.md](docs/CONTRIBUTING.md).

## 📄 Licença

[Definir licença apropriada]

## 📧 Contato

Allan Ramalho - [Informações de contato]

---

**GeoBot** - Transformando processamento geofísico com Inteligência Artificial
