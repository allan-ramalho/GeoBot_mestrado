# 🏗️ Arquitetura do GeoBot

## Visão Geral

O GeoBot é composto por três camadas principais:

1. **Frontend** - Electron + React + TypeScript
2. **Backend** - FastAPI + Python
3. **AI/RAG Layer** - Sistema de IA com RAG

```
┌─────────────────────────────────────────────────────────────┐
│                    ELECTRON CONTAINER                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           React Frontend (TypeScript)                │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │   │
│  │  │   UI     │  │  Stores  │  │ Services │          │   │
│  │  │Components│  │ (Zustand)│  │  (API)   │          │   │
│  │  └──────────┘  └──────────┘  └──────────┘          │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                   │
│                    HTTP/WebSocket                            │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │          FastAPI Backend (Python)                    │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │   │
│  │  │   API    │  │ Services │  │  Models  │          │   │
│  │  │ Endpoints│  │   AI     │  │   Data   │          │   │
│  │  └──────────┘  │ Geophys  │  └──────────┘          │   │
│  │                │ Storage  │                          │   │
│  │                └──────────┘                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                   │
│                    External Services                         │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Supabase (Postgres + Storage) | AI Providers       │   │
│  │  pgvector, PDF Storage         | Groq/OpenAI/etc    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Camada Frontend

### Estrutura

```
frontend/
├── electron/           # Electron main process
│   ├── main.js        # Entry point, backend manager
│   └── preload.js     # Secure IPC bridge
├── src/
│   ├── components/    # React components
│   ├── pages/         # Page components
│   ├── stores/        # Zustand stores
│   ├── services/      # API clients
│   ├── types/         # TypeScript types
│   └── utils/         # Utilities
└── package.json
```

### Tecnologias

- **Electron**: Container desktop
- **React 18**: UI framework
- **TypeScript**: Type safety
- **Zustand**: State management (lightweight)
- **Tailwind + Shadcn**: Styling
- **Plotly.js**: Visualizações científicas
- **Axios**: HTTP client

### Fluxo de Dados

1. Usuário interage com UI
2. Componente dispara action no Store
3. Store chama Service (API)
4. Service faz requisição HTTP ao Backend
5. Resposta atualiza Store
6. UI re-renderiza automaticamente

## Camada Backend

### Estrutura

```
backend/
├── app/
│   ├── api/
│   │   └── endpoints/      # REST endpoints
│   ├── core/               # Config, logging
│   ├── models/             # Data models
│   ├── services/
│   │   ├── ai/            # AI providers, RAG
│   │   ├── geophysics/    # Processing engine
│   │   └── storage/       # Data management
│   └── main.py            # FastAPI app
├── requirements.txt
└── pyproject.toml
```

### Tecnologias

- **FastAPI**: Web framework moderno
- **Uvicorn**: ASGI server
- **Pydantic**: Validação de dados
- **NumPy/SciPy**: Computação científica
- **Sentence-Transformers**: Embeddings
- **Supabase SDK**: Database e storage
- **httpx**: Async HTTP client

### Endpoints Principais

#### AI Configuration
- `POST /ai/providers/configure` - Configurar provider
- `GET /ai/providers/{provider}/models` - Listar modelos
- `GET /ai/config/current` - Config atual

#### Chat
- `POST /chat/message` - Enviar mensagem
- `WS /chat/ws` - WebSocket streaming
- `GET /chat/conversations/{id}` - Histórico

#### Processing
- `GET /processing/functions` - Listar funções
- `POST /processing/execute` - Executar processamento
- `POST /processing/workflow` - Executar workflow
- `GET /processing/jobs/{id}` - Status do job

#### Projects
- `POST /projects/create` - Criar projeto
- `GET /projects/list` - Listar projetos
- `GET /projects/{id}/tree` - Árvore de arquivos

#### Data
- `POST /data/upload` - Upload de arquivo
- `POST /data/import` - Importar dados
- `GET /data/files/{project_id}` - Listar arquivos

## Sistema AI + RAG

### Arquitetura RAG

```
┌─────────────────────────────────────────────────────┐
│                   RAG Pipeline                       │
│                                                      │
│  1. PDF Ingestion                                   │
│     ↓                                               │
│  2. Chunking (1000 tokens, 200 overlap)            │
│     ↓                                               │
│  3. Embeddings (E5-Large, 1024 dim)                │
│     ↓                                               │
│  4. Storage (Supabase + pgvector)                  │
│                                                      │
│  Query Flow:                                        │
│  User Query → Embedding → Vector Search →          │
│  → Top-K Results → Context for LLM                 │
└─────────────────────────────────────────────────────┘
```

### Multi-Provider AI

Suporta múltiplos providers com fallback automático (Groq):

```python
providers = {
    "groq": ["llama-3.3-70b", "llama-3.1-70b", "mixtral-8x7b"],
    "openai": ["gpt-4", "gpt-3.5-turbo"],
    "claude": ["claude-3-opus", "claude-3-sonnet"],
    "gemini": ["gemini-pro"]
}
```

### Function Calling System

```
User Command (NL)
    ↓
Semantic Search (Function Registry)
    ↓
Find Best Match(es)
    ↓
Extract Parameters (LLM)
    ↓
Execute Function(s)
    ↓
Return Results + Update UI
```

#### Function Registry

Auto-discovery via decorator:

```python
@register(
    name="reduction_to_pole",
    description="Apply RTP transformation...",
    keywords=["RTP", "reduction", "pole", "magnetic"],
    parameters={...}
)
def reduction_to_pole(data, inclination, declination):
    # Implementation
    pass
```

Registro automático com embeddings para busca semântica.

## Processing Engine

### Workflow System

Suporta execução encadeada:

```json
{
  "data_id": "mag_001",
  "steps": [
    {"function": "reduction_to_pole", "parameters": {"inc": -30, "dec": -20}},
    {"function": "upward_continuation", "parameters": {"height": 500}},
    {"function": "horizontal_gradient", "parameters": {}}
  ]
}
```

### Job Management

- Execução síncrona ou assíncrona
- Sistema de fila
- Tracking de progresso
- Cancelamento de jobs
- Logs estruturados

## Storage Architecture

### Project Structure

```
~/GeoBot/
├── data/
│   ├── pdfs/              # RAG documents
│   ├── logs/              # Application logs
│   └── config/            # Configuration files
└── projects/
    └── {project_id}/
        ├── raw_data/
        ├── processed_data/
        ├── interpretations/
        ├── maps/
        ├── exports/
        ├── features/
        └── project.json
```

### Supabase Integration

```
Supabase
├── Storage
│   └── pdfs/             # Scientific literature PDFs
└── Postgres + pgvector
    ├── documents         # Chunks + embeddings
    └── RPC: match_documents()  # Vector search
```

## Comunicação Frontend ↔ Backend

### HTTP REST

```typescript
// Frontend service
export const processingService = {
  async execute(functionName, dataId, params) {
    return apiClient.post('/processing/execute', {
      function_name: functionName,
      data_id: dataId,
      parameters: params
    });
  }
};
```

### WebSocket (Chat)

```typescript
const ws = new WebSocket('ws://localhost:8000/api/v1/chat/ws');

ws.send(JSON.stringify({
  message: "Apply RTP with I=-30, D=-20",
  use_rag: true
}));

ws.onmessage = (event) => {
  const chunk = JSON.parse(event.data);
  // Handle streaming response
};
```

## Empacotamento

### Electron Builder

```json
{
  "build": {
    "files": ["dist/**/*", "electron/**/*", "backend/**/*"],
    "extraResources": [
      {"from": "../backend", "to": "backend"}
    ]
  }
}
```

### Estratégia

1. **Frontend**: Build React → `dist/`
2. **Backend**: PyInstaller → executável Python
3. **Bundle**: Electron Builder empacota tudo
4. **Resultado**: Instalador standalone (.exe, .AppImage, .deb)

## Segurança

- API Keys armazenadas localmente (não no código)
- Comunicação backend localhost apenas
- Validação de entrada (Pydantic)
- CORS configurado para localhost apenas
- Electron context isolation

## Performance

- **Backend**: Async/await, ThreadPoolExecutor
- **Frontend**: React.memo, lazy loading
- **Processing**: Workers paralelos, cache
- **RAG**: Embeddings pre-computados, índice vetorial

## Extensibilidade

### Adicionar Nova Função de Processamento

1. Criar arquivo em `backend/app/services/geophysics/functions/`
2. Usar decorator `@register`
3. Documentar detalhadamente
4. Auto-descoberta pelo sistema

### Adicionar Novo AI Provider

1. Adicionar em `AIProvider` enum
2. Implementar `_validate_{provider}` e `_list_{provider}_models`
3. Implementar `_call_{provider}` no ChatService

## Monitoramento

- Logs estruturados (por dia)
- Health check endpoints
- Job status tracking
- Error handling centralizado

## Próximos Passos

1. Implementar UI completa (mapas, processamento, chat)
2. Adicionar mais funções geofísicas
3. Implementar picking interativo
4. Sistema de exportação (CSV, JSON, imagens)
5. Documentação interna navegável
6. Testes automatizados
7. CI/CD pipeline
