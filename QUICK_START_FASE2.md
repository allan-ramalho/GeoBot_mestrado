# 🚀 Quick Start - GeoBot Fase 2

Guia rápido para começar a usar o GeoBot imediatamente.

---

## ⚡ Início Rápido (5 minutos)

### 1. Instalar Dependências

```powershell
# Backend
cd backend
pip install PyPDF2==3.0.1

# Frontend
cd frontend
npm install
```

### 2. Configurar AI Provider

Escolha **1 provider** (Groq é recomendado - grátis):

**Opção A - Groq (Recomendado)**:
```bash
# backend/.env
GROQ_API_KEY=gsk_...
```
→ Obter key: https://console.groq.com/keys

**Opção B - OpenAI**:
```bash
OPENAI_API_KEY=sk-...
```
→ Obter key: https://platform.openai.com/api-keys

**Opção C - Claude**:
```bash
ANTHROPIC_API_KEY=sk-ant-...
```
→ Obter key: https://console.anthropic.com/

**Opção D - Gemini**:
```bash
GOOGLE_API_KEY=AI...
```
→ Obter key: https://makersuite.google.com/app/apikey

### 3. Iniciar Aplicação

**Terminal 1 - Backend**:
```powershell
cd backend
venv\Scripts\activate
uvicorn app.main:app --reload
```

**Terminal 2 - Frontend**:
```powershell
cd frontend
npm run dev
```

### 4. Usar

1. Abrir: http://localhost:5173
2. Configurar provider na tela inicial
3. Navegar para **Chat**
4. Começar a conversar!

---

## 💬 Exemplos de Uso

### Perguntas Gerais

```
"O que é redução ao polo?"
"Explique continuação para cima"
"Qual a diferença entre gravimetria e magnetometria?"
```

### Listar Funções

```
"Quais funções de processamento estão disponíveis?"
"Liste as funções magnéticas"
"Mostre funções para remoção de ruído"
```

### Executar Processamento

```
"Processe com redução ao polo usando inclinação -30 e declinação -45"
"Aplique continuação para cima com altura de 500 metros"
"Calcule o gradiente horizontal dos dados"
```

### Com RAG (se configurado)

```
"O que dizem os artigos sobre anomalias magnéticas?"
"Cite referências sobre interpretação de dados gravimétricos"
"Pesquise papers sobre transformada de Fourier em geofísica"
```

---

## 🧪 Testar Features

### 1. Markdown
```
"Explique RTP com **negrito** e código: `reduction_to_pole(data)`"
```

### 2. Code Highlighting
```
"Mostre exemplo de código Python para FFT"
```

### 3. Listas
```
"Liste 5 métodos geofísicos em bullets"
```

### 4. Múltiplas mensagens
```
Envie 3-4 perguntas seguidas e veja histórico
```

---

## 🎨 Customizações Rápidas

### Alterar Modelo

1. Botão "Nova Conversa"
2. Vai para setup
3. Escolher novo modelo
4. Voltar ao chat

### Desligar RAG

Toggle "Usar RAG" no header do chat

### Tema Dark/Light

Configuração será adicionada futuramente (atualmente dark por padrão)

---

## 🔧 Configuração Avançada (Opcional)

### RAG com PDFs

**1. Criar Supabase Project**:
- Ir para https://supabase.com
- Criar projeto gratuito
- Copiar URL e Key

**2. Configurar .env**:
```bash
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJ...
SUPABASE_PDF_BUCKET=pdfs
```

**3. Executar SQL Setup**:
```sql
-- No Supabase SQL Editor, executar:
-- scripts/supabase_setup.sql
```

**4. Upload PDFs**:
- Dashboard → Storage → pdfs
- Upload arquivos PDF

**5. Ingerir**:
```powershell
cd scripts
python ingest_pdfs.py
```

**6. Testar no Chat**:
```
"O que dizem os PDFs sobre [tópico]?"
```

---

## 🐛 Problemas Comuns

### "ModuleNotFoundError: PyPDF2"
```powershell
pip install PyPDF2==3.0.1
```

### "react-markdown not found"
```powershell
cd frontend
npm install
```

### Backend não conecta
- Verificar se porta 8000 está livre
- Verificar CORS no backend/.env

### API Key inválida
- Verificar key copiada corretamente (sem espaços)
- Verificar key ativa no dashboard do provider
- Testar key com curl ou Postman

### WebSocket error
- Verificar backend rodando
- Limpar cache do navegador (Ctrl+Shift+R)

---

## 📚 Recursos

### Documentação
- [FASE_2_COMPLETA.md](FASE_2_COMPLETA.md) - Detalhes de implementação
- [GUIA_TESTES_FASE2.md](GUIA_TESTES_FASE2.md) - Guia completo de testes
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Arquitetura
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) - Desenvolvimento

### Links Úteis
- Groq API: https://console.groq.com
- OpenAI API: https://platform.openai.com
- Claude API: https://console.anthropic.com
- Gemini API: https://makersuite.google.com
- Supabase: https://supabase.com

### Comandos Úteis

```powershell
# Ver logs backend
cd backend
tail -f logs/app.log

# Rebuild frontend
cd frontend
npm run build

# Limpar cache
npm run dev -- --force

# Verificar saúde do backend
curl http://localhost:8000/health

# Testar WebSocket (PowerShell)
# Ver GUIA_TESTES_FASE2.md
```

---

## ✅ Checklist Primeira Execução

- [ ] Backend instalado (pip install)
- [ ] Frontend instalado (npm install)
- [ ] .env configurado com pelo menos 1 API key
- [ ] Backend iniciado (porta 8000)
- [ ] Frontend iniciado (porta 5173)
- [ ] Provider configurado na UI
- [ ] Primeira mensagem enviada
- [ ] Resposta recebida
- [ ] Markdown renderiza
- [ ] Sem erros no console

---

## 🎯 Próximos Passos

Após configurar básico:

1. **Explorar Chat**:
   - Testar diferentes tipos de perguntas
   - Ver markdown e code highlighting
   - Testar múltiplas conversas

2. **Configurar RAG** (opcional):
   - Setup Supabase
   - Upload PDFs
   - Ingerir documentos
   - Testar busca semântica

3. **Testar Providers**:
   - Configurar 2+ providers
   - Comparar respostas
   - Testar fallback

4. **Aguardar Fase 3**:
   - Mais funções de processamento
   - Map viewer
   - Workflows

---

**Tempo estimado para setup completo**: 5-10 minutos

**Dúvidas?** Ver [GUIA_TESTES_FASE2.md](GUIA_TESTES_FASE2.md)

**Pronto!** 🚀 Comece a usar o GeoBot!
