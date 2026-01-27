# 📖 Guia de Desenvolvimento - GeoBot

## Setup do Ambiente de Desenvolvimento

### Pré-requisitos

- **Python 3.11.9** (exato)
- **Node.js 18+**
- **Git**
- **VS Code** (recomendado)

### 1. Clonar e Configurar

```bash
cd GeoBot_Mestrado
```

### 2. Setup Backend

```bash
cd backend

# Criar ambiente virtual
python -m venv venv

# Ativar
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Instalar dependências
pip install -r requirements.txt

# Configurar variáveis de ambiente
copy .env.example .env
# Editar .env com suas credenciais

# Testar
python -m pytest
```

### 3. Setup Frontend

```bash
cd ../frontend

# Instalar dependências
npm install

# Iniciar desenvolvimento
npm run dev
```

### 4. Executar Completo

#### Opção 1: Separado (Desenvolvimento)

Terminal 1:
```bash
cd backend
venv\Scripts\activate
uvicorn app.main:app --reload
```

Terminal 2:
```bash
cd frontend
npm run dev
```

Abrir: http://localhost:5173

#### Opção 2: Electron (Produção-like)

```bash
cd frontend
npm run electron:dev
```

## Estrutura de Código

### Backend

#### Criar Novo Endpoint

```python
# backend/app/api/endpoints/my_endpoint.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


class MyRequest(BaseModel):
    field: str


@router.post("/my-route")
async def my_handler(request: MyRequest):
    """
    Descrição do endpoint
    """
    try:
        # Lógica
        return {"result": "success"}
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

Registrar no router:

```python
# backend/app/api/__init__.py

from app.api.endpoints import my_endpoint

api_router.include_router(
    my_endpoint.router,
    prefix="/my",
    tags=["My Feature"]
)
```

#### Criar Nova Função de Processamento

```python
# backend/app/services/geophysics/functions/my_functions.py

from app.services.geophysics.function_registry import register
import numpy as np

@register(
    name="my_transformation",
    description="""
    Descrição detalhada da transformação.
    
    Inclua:
    - O que faz
    - Quando usar
    - Referências científicas
    - Boas práticas
    
    Keywords: keyword1, keyword2, ...
    """,
    keywords=["keyword1", "keyword2", "transformation"],
    parameters={
        "param1": {
            "type": "number",
            "description": "Descrição do parâmetro",
            "required": True
        }
    },
    examples=[
        "Apply my transformation with param1=10",
        "Transform using param1 of 10"
    ]
)
def my_transformation(data: dict, param1: float):
    """
    Implementação da transformação
    
    Args:
        data: Dictionary com 'x', 'y', 'z'
        param1: Parâmetro da transformação
    
    Returns:
        Dados transformados
    """
    z = data['z']
    
    # Processamento
    z_transformed = z * param1  # Exemplo
    
    result = data.copy()
    result['z'] = z_transformed
    result['processing_history'] = data.get('processing_history', []) + [
        f"My Transformation: param1={param1}"
    ]
    
    return result
```

Importar no módulo:

```python
# backend/app/services/geophysics/__init__.py
from app.services.geophysics.functions import my_functions
```

#### Criar Novo Service

```python
# backend/app/services/my_service.py

import logging

logger = logging.getLogger(__name__)


class MyService:
    """
    Descrição do serviço
    """
    
    def __init__(self):
        self.initialized = False
    
    async def initialize(self):
        """Initialize service"""
        if not self.initialized:
            # Setup
            self.initialized = True
            logger.info("✅ MyService initialized")
    
    async def do_something(self, param: str):
        """
        Faz algo
        """
        await self.initialize()
        
        # Lógica
        return {"result": param}
```

### Frontend

#### Criar Novo Componente

```typescript
// frontend/src/components/MyComponent.tsx

interface MyComponentProps {
  title: string;
  onAction: () => void;
}

export default function MyComponent({ title, onAction }: MyComponentProps) {
  return (
    <div className="p-4 bg-card rounded-lg">
      <h2 className="text-xl font-bold mb-4">{title}</h2>
      <button
        onClick={onAction}
        className="px-4 py-2 bg-primary text-primary-foreground rounded"
      >
        Action
      </button>
    </div>
  );
}
```

#### Criar Nova Página

```typescript
// frontend/src/pages/MyPage.tsx

import { useState, useEffect } from 'react';
import { useMyStore } from '@/stores/myStore';

export default function MyPage() {
  const { data, loadData } = useMyStore();
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadData();
  }, []);

  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold mb-4">My Page</h1>
      {loading ? (
        <p>Loading...</p>
      ) : (
        <div>{/* Conteúdo */}</div>
      )}
    </div>
  );
}
```

Adicionar rota:

```typescript
// frontend/src/App.tsx

<Route path="/my-page" element={<MyPage />} />
```

#### Criar Nova Store (Zustand)

```typescript
// frontend/src/stores/myStore.ts

import { create } from 'zustand';
import { apiClient } from '@/services/api';

interface MyState {
  data: any[];
  loading: boolean;
  
  loadData: () => Promise<void>;
  addItem: (item: any) => void;
}

export const useMyStore = create<MyState>((set, get) => ({
  data: [],
  loading: false,

  loadData: async () => {
    set({ loading: true });
    try {
      const response = await apiClient.get('/my-endpoint');
      set({ data: response.data });
    } catch (error) {
      console.error('Failed to load data:', error);
    } finally {
      set({ loading: false });
    }
  },

  addItem: (item) => {
    set({ data: [...get().data, item] });
  },
}));
```

#### Criar Novo Service

```typescript
// frontend/src/services/myService.ts

import { apiClient } from './api';

export const myService = {
  async getData() {
    const response = await apiClient.get('/my-endpoint');
    return response.data;
  },

  async postData(data: any) {
    const response = await apiClient.post('/my-endpoint', data);
    return response.data;
  },
};
```

## Testes

### Backend

```python
# backend/tests/test_my_feature.py

import pytest
from app.services.my_service import MyService


@pytest.mark.asyncio
async def test_my_service():
    service = MyService()
    await service.initialize()
    
    result = await service.do_something("test")
    assert result["result"] == "test"
```

Executar:

```bash
cd backend
pytest
pytest --cov  # Com coverage
```

### Frontend

```typescript
// frontend/src/components/__tests__/MyComponent.test.tsx

import { render, screen } from '@testing-library/react';
import MyComponent from '../MyComponent';

describe('MyComponent', () => {
  it('renders title', () => {
    render(<MyComponent title="Test" onAction={() => {}} />);
    expect(screen.getByText('Test')).toBeInTheDocument();
  });
});
```

## Debugging

### Backend

VS Code launch.json:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: FastAPI",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": ["app.main:app", "--reload"],
      "cwd": "${workspaceFolder}/backend",
      "env": {
        "PYTHONPATH": "${workspaceFolder}/backend"
      }
    }
  ]
}
```

### Frontend

React DevTools + Redux DevTools (Zustand middleware)

## Boas Práticas

### Python

- Type hints sempre que possível
- Docstrings em funções públicas
- Logging estruturado
- Async/await para I/O
- Pydantic para validação

### TypeScript

- Interfaces para props
- Hooks customizados quando reutilizável
- Componentes pequenos e focados
- Estado local vs global apropriado
- Error boundaries

### Git

```bash
# Branches
main          # Produção estável
develop       # Desenvolvimento
feature/xyz   # Features
fix/xyz       # Bug fixes

# Commits
feat: Add new processing function
fix: Resolve RTP calculation bug
docs: Update architecture documentation
refactor: Improve function registry
```

## Performance

### Backend

- Use `async def` para I/O
- ThreadPoolExecutor para CPU-bound
- Cache com `functools.lru_cache`
- Batch operations quando possível

### Frontend

- React.memo para componentes caros
- useMemo/useCallback apropriadamente
- Lazy loading de rotas
- Virtualização para listas longas

## Empacotamento

### Build Desenvolvimento

```bash
# Frontend apenas
cd frontend
npm run build

# Backend teste
cd backend
python app/main.py
```

### Build Produção

```bash
cd frontend
npm run electron:build:win    # Windows
npm run electron:build:linux  # Linux
```

Resultado em `frontend/dist-electron/`

## Troubleshooting

### Backend não inicia

- Verificar Python 3.11.9
- Verificar venv ativado
- Verificar requirements instalados
- Checar logs em `~/GeoBot/data/logs/`

### Frontend não conecta

- Backend rodando?
- Porta 8000 disponível?
- CORS configurado?
- Verificar console do navegador

### Electron não inicia

- Backend bundled corretamente?
- Python disponível no bundle?
- Verificar electron/main.js logs

## Recursos

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [React Docs](https://react.dev/)
- [Electron Docs](https://www.electronjs.org/)
- [Zustand Docs](https://zustand-demo.pmnd.rs/)
- [Tailwind Docs](https://tailwindcss.com/)
