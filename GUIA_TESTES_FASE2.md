# 🧪 Guia de Testes - Fase 2 (AI Core)

Guia prático para testar todas as funcionalidades implementadas na Fase 2.

---

## ⚙️ Pré-requisitos

### 1. Instalar dependências backend
```powershell
cd backend
venv\Scripts\activate
pip install PyPDF2==3.0.1
```

### 2. Instalar dependências frontend
```powershell
cd frontend
npm install
# Isso instalará react-markdown e react-syntax-highlighter
```

### 3. Configurar .env
```bash
# backend/.env

# Escolha pelo menos 1 provider
GROQ_API_KEY=gsk_...              # Recomendado (gratuito)
OPENAI_API_KEY=sk-...             # Opcional
ANTHROPIC_API_KEY=sk-ant-...      # Opcional  
GOOGLE_API_KEY=AI...              # Opcional

# Para RAG (opcional nesta fase)
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJ...
SUPABASE_PDF_BUCKET=pdfs
```

---

## 🧪 Teste 1: Backend Inicialização

### Iniciar backend
```powershell
cd backend
venv\Scripts\activate
uvicorn app.main:app --reload
```

### Verificar health
```
Abrir navegador: http://localhost:8000/health
Esperado: {"status": "healthy", ...}
```

### Verificar docs
```
Abrir: http://localhost:8000/api/docs
Deve mostrar Swagger UI com todos os endpoints
```

**✅ Sucesso**: Backend inicia sem erros

---

## 🧪 Teste 2: AI Providers

### Testar via UI

1. **Iniciar frontend**:
```powershell
cd frontend
npm run dev
```

2. **Abrir**: http://localhost:5173

3. **Configurar provider**:
   - Selecionar provider (Groq, OpenAI, Claude ou Gemini)
   - Inserir API key
   - Deve listar modelos disponíveis
   - Selecionar modelo
   - Salvar configuração

4. **Navegar para Chat**

5. **Testar mensagem simples**:
   ```
   Input: "Olá, quem é você?"
   Esperado: Resposta do AI apresentando-se como GeoBot
   ```

### Testar providers diferentes

**Groq** (Recomendado primeiro):
- API key: https://console.groq.com
- Modelo: `llama-3.3-70b-versatile`
- Teste: Enviar "Explique o que é geofísica"

**OpenAI**:
- API key: https://platform.openai.com
- Modelo: `gpt-3.5-turbo` ou `gpt-4`
- Teste: "O que é redução ao polo?"

**Claude**:
- API key: https://console.anthropic.com
- Modelo: `claude-3-sonnet-20240229`
- Teste: "Descreva continuação para cima"

**Gemini**:
- API key: https://makersuite.google.com/app/apikey
- Modelo: `gemini-pro`
- Teste: "Quais métodos geofísicos você conhece?"

**✅ Sucesso**: Todos os providers respondem corretamente

---

## 🧪 Teste 3: Chat UI Features

### 3.1 Markdown Rendering

**Teste**:
```
Input: "Explique redução ao polo com **negrito**, *itálico*, 
e código: `reduction_to_pole(data, inc=45, dec=-30)`"
```

**Esperado**:
- Texto em **negrito** e *itálico* renderizado
- Código inline com background destacado

### 3.2 Code Blocks

**Teste**:
```
Input: "Mostre exemplo de código Python para processar dados magnéticos"
```

**Esperado**:
- Bloco de código com syntax highlighting
- Linguagem Python detectada
- Tema VS Code Dark aplicado
- Botão copiar código (se implementado)

### 3.3 Listas

**Teste**:
```
Input: "Liste 5 métodos geofísicos em bullet points"
```

**Esperado**:
- Lista com bullets renderizada
- Formatação correta

### 3.4 Múltiplas Mensagens

**Teste**:
1. Enviar 5 mensagens consecutivas
2. Verificar que todas aparecem
3. Verificar auto-scroll para última mensagem
4. Verificar timestamps diferentes

### 3.5 Empty State

**Teste**:
1. Abrir Chat pela primeira vez
2. Verificar mensagem de boas-vindas
3. Ver exemplos de perguntas
4. Clicar em exemplo
5. Verificar que preenche input

### 3.6 Nova Conversa

**Teste**:
1. Enviar algumas mensagens
2. Clicar em "Nova Conversa"
3. Verificar que mensagens são limpas
4. Enviar nova mensagem
5. Verificar novo conversation_id

**✅ Sucesso**: Todas as features visuais funcionam

---

## 🧪 Teste 4: RAG System (Se configurado Supabase)

### 4.1 Preparar PDFs

1. **Criar bucket no Supabase**:
   - Dashboard → Storage → Create bucket
   - Nome: `pdfs`
   - Public: No

2. **Upload PDFs**:
   - Fazer upload de 2-3 PDFs científicos de geofísica
   - Aceita qualquer PDF (artigos, manuais, livros)

### 4.2 Executar Ingestão

```powershell
cd scripts
python ingest_pdfs.py
```

**Output esperado**:
```
📚 GeoBot PDF Ingestion System
============================================================
🔧 Configuration:
  Supabase URL: https://xxx.supabase.co
  Bucket: pdfs
  Embedding Model: intfloat/e5-large-v2
  Chunk Size: 1000

🚀 Initializing services...
✅ RAG engine initialized
✅ PDF parser initialized

📥 Downloading PDFs from Supabase...
✅ Found 3 PDF files

🔄 Processing PDFs...
------------------------------------------------------------
📥 Downloading: Smith_2020_Magnetic.pdf
✅ Downloaded: Smith_2020_Magnetic.pdf
🔄 Processing: Smith_2020_Magnetic.pdf
📄 Parsed 15 pages, 52 chunks
  Embedding chunk 1/52
  Embedding chunk 11/52
  Embedding chunk 21/52
  ...
💾 Storing 52 chunks in database...
✅ Stored 52/52 chunks
------------------------------------------------------------

📊 Ingestion Summary:
============================================================
✅ Successful: 3
❌ Failed: 0

📈 Statistics:
  Total pages processed: 45
  Total chunks created: 156
  Total chunks stored: 156

✅ Ingestion complete!
============================================================
```

### 4.3 Testar RAG no Chat

1. **Com RAG ativado** (toggle ON):
```
Input: "O que dizem os artigos sobre anomalias magnéticas?"
```

**Esperado**:
- Resposta baseada nos PDFs ingeridos
- Citações aparecem abaixo da resposta
- Formato: "Autor (Ano). Título, p. X"

2. **Sem RAG** (toggle OFF):
```
Input: mesma pergunta
```

**Esperado**:
- Resposta genérica sem citações
- Sem referências específicas aos PDFs

**✅ Sucesso**: RAG busca e cita documentos corretamente

---

## 🧪 Teste 5: WebSocket Streaming

### Testar manualmente via browser

1. **Abrir Console do navegador** (F12)

2. **Executar código**:
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/chat/ws');

ws.onopen = () => {
    console.log('✅ WebSocket connected');
    ws.send(JSON.stringify({
        message: "Explique o método de Euler deconvolution em detalhes",
        use_rag: true
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('📨 Received:', data);
    
    if (data.type === 'content') {
        console.log('💬 Content:', data.content);
    }
    if (data.type === 'end') {
        console.log('✅ Complete!');
        ws.close();
    }
};

ws.onerror = (error) => {
    console.error('❌ Error:', error);
};

ws.onclose = () => {
    console.log('🔌 WebSocket closed');
};
```

**Esperado no console**:
```
✅ WebSocket connected
📨 Received: {type: 'start', conversation_id: '...'}
💬 Content: (chunks de resposta gradualmente)
📨 Received: {type: 'citation', citation: {...}}
📨 Received: {type: 'end', message_id: '...'}
✅ Complete!
🔌 WebSocket closed
```

### Testar via Python

```python
import asyncio
import websockets
import json

async def test_ws():
    uri = "ws://localhost:8000/api/v1/chat/ws"
    
    async with websockets.connect(uri) as ws:
        print("✅ Connected")
        
        # Send message
        await ws.send(json.dumps({
            "message": "O que é transformada de Fourier em geofísica?",
            "use_rag": False
        }))
        
        # Receive responses
        async for message in ws:
            data = json.loads(message)
            
            if data['type'] == 'start':
                print(f"🚀 Started: {data.get('conversation_id')}")
            
            elif data['type'] == 'content':
                print(f"💬 {data['content']}", end='', flush=True)
            
            elif data['type'] == 'citation':
                print(f"\n📚 Citation: {data['citation']}")
            
            elif data['type'] == 'end':
                print(f"\n✅ Done: {data['message_id']}")
                break
            
            elif data['type'] == 'error':
                print(f"\n❌ Error: {data['error']}")
                break

asyncio.run(test_ws())
```

**✅ Sucesso**: WebSocket recebe chunks em tempo real

---

## 🧪 Teste 6: PDF Parser Detalhado

### Teste unitário

```python
# backend/test_pdf_parser.py
from app.services.ai.pdf_parser import PDFParser
import json

# Caminho para PDF de teste
pdf_path = "path/to/test.pdf"

# Inicializar parser
parser = PDFParser(
    chunk_size=1000,
    chunk_overlap=200,
    min_chunk_size=100
)

# Parse PDF
result = parser.parse_pdf(pdf_path)

# Verificar resultado
print(f"📄 Parsed: {result['metadata']['filename']}")
print(f"📊 Pages: {result['pages']}")
print(f"📝 Total chars: {len(result['text'])}")
print(f"🔪 Chunks: {len(result['chunks'])}")

# Ver primeiro chunk
print("\n--- First Chunk ---")
print(result['chunks'][0]['text'][:200])
print(f"Metadata: {json.dumps(result['chunks'][0]['metadata'], indent=2)}")

# Ver estatísticas de chunks
chunk_sizes = [len(c['text']) for c in result['chunks']]
print(f"\n📊 Chunk size stats:")
print(f"  Min: {min(chunk_sizes)}")
print(f"  Max: {max(chunk_sizes)}")
print(f"  Avg: {sum(chunk_sizes) / len(chunk_sizes):.0f}")
```

**✅ Sucesso**: Parser extrai texto e cria chunks consistentes

---

## 🧪 Teste 7: Function Calling

### Testar comando de processamento

**Via Chat UI**:
```
Input: "Processe meus dados magnéticos com redução ao polo, 
        usando inclinação -30 e declinação -45"
```

**Esperado**:
- AI interpreta como function call
- Executa `reduction_to_pole`
- Retorna resultado ou pede data_id se faltando

### Testar busca de funções

```
Input: "Quais funções de processamento estão disponíveis?"
```

**Esperado**:
- Lista das 5 funções magnéticas
- Descrição de cada uma
- Parâmetros necessários

### Testar semantic search

```
Input: "Como remover ruído regional dos dados?"
```

**Esperado**:
- AI sugere `upward_continuation`
- Explica que remove componentes de alta frequência

**✅ Sucesso**: Function calling interpreta comandos corretamente

---

## 📊 Checklist de Validação Final

### Backend
- [ ] Backend inicia sem erros
- [ ] `/health` retorna 200
- [ ] `/api/docs` acessível
- [ ] Logs aparecem corretamente

### AI Providers
- [ ] Pelo menos 1 provider configurado e funcionando
- [ ] Chat responde perguntas
- [ ] Respostas em português (ou idioma detectado)
- [ ] Sem erros de API key

### Chat UI
- [ ] Interface carrega sem erros de console
- [ ] Markdown renderiza corretamente
- [ ] Code highlighting funciona
- [ ] Auto-scroll para nova mensagem
- [ ] Timestamps aparecem
- [ ] Botão "Nova Conversa" limpa histórico

### RAG (Opcional)
- [ ] PDFs fazem upload para Supabase
- [ ] Script ingest_pdfs.py executa sem erros
- [ ] Chunks armazenados no banco
- [ ] Busca RAG retorna documentos relevantes
- [ ] Citações aparecem na resposta

### WebSocket
- [ ] Conexão estabelece com sucesso
- [ ] Mensagens enviam/recebem
- [ ] Eventos tipados corretos (start, content, end)
- [ ] Conexão fecha gracefully

### Error Handling
- [ ] API key inválida mostra erro apropriado
- [ ] Timeout de API retorna erro amigável
- [ ] WebSocket error não trava aplicação
- [ ] Frontend mostra erros ao usuário

---

## 🐛 Troubleshooting Comum

### Erro: "PyPDF2 not found"
```powershell
pip install PyPDF2==3.0.1
```

### Erro: "react-markdown not found"
```powershell
cd frontend
npm install react-markdown react-syntax-highlighter
```

### Erro: WebSocket connection failed
- Verificar backend rodando
- Verificar porta 8000 livre
- Verificar CORS configurado para localhost

### Erro: Supabase authentication failed
- Verificar SUPABASE_URL correto
- Verificar SUPABASE_KEY válido
- Verificar bucket existe e tem permissões

### PDFs não aparecem
- Verificar bucket name correto no .env
- Verificar arquivos têm extensão .pdf
- Verificar RLS (Row Level Security) não bloqueia

---

## ✅ Critérios de Sucesso

### Mínimo (MVP)
- ✅ Backend responde
- ✅ 1 provider AI configurado
- ✅ Chat básico funciona
- ✅ Markdown renderiza

### Completo
- ✅ 2+ providers configurados
- ✅ RAG funciona com PDFs
- ✅ WebSocket streaming
- ✅ Citações aparecem
- ✅ Code highlighting

### Excelente
- ✅ Todos 4 providers testados
- ✅ 10+ PDFs ingeridos
- ✅ RAG retorna resultados relevantes
- ✅ UI responsiva e polida
- ✅ Zero erros no console

---

**Última atualização**: Fase 2 - Janeiro 2026
