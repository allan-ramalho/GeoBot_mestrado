# 🌍 GeoBot - Agente de IA para Processamento Geofísico

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11.9-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31.1-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1+cu124-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Groq](https://img.shields.io/badge/LLM-Groq_API-7C3AED)](https://groq.com/)

**AI Assistant com aceleração GPU para análise e processamento de dados geofísicos de gravimetria e magnetometria**

[🚀 Instalação](#-instalação-rápida) • [📖 Documentação](#-documentação) • [🎯 Recursos](#-recursos) • [⚡ GPU](#-aceleração-gpu)

</div>

---

## ✨ O que é o GeoBot?

GeoBot é um agente de IA que combina **processamento geofísico** com **inteligência artificial generativa** para tornar a análise de dados de métodos potenciais mais acessível e eficiente, possibilitando ao usuário otimização de tempo e foco maior em atividados de maior relevância.

### 🎯 Principais Funcionalidades

| Funcionalidade | Descrição |
|----------------|-----------|
| 💬 **Conversação Natural** | Processe dados simplesmente conversando: *"Aplique correção de Bouguer com densidade 2.67"* |
| 📚 **Citações Automáticas** | Sistema RAG (Retrieval-Augmented Generation) busca e cita papers científicos automaticamente em formato ABNT |
| 🔬 **Processamento Geofísico** | Biblioteca completa: Bouguer, RTP, derivadas, continuação, filtros, sinal analítico, tilt angle |
| 📊 **Visualizações Interativas** | Mapas 2D/3D com Plotly, comparações antes/depois, histogramas, estatísticas |
| 🚀 **Aceleração GPU** | Suporte automático para NVIDIA CUDA e Apple Silicon (M1/M2) |
| 🔌 **Extensível** | Sistema de registro de funções permite adicionar novos processamentos |

---

## 🚀 Instalação Rápida

### Windows (Recomendado)

```powershell
# 1. Clone o repositório
git clone https://github.com/allan-ramalho/GeoBot_mestrado.git
cd GeoBot_mestrado

# 2. Execute o instalador automático
.\INSTALAR.bat

# 3. Configure suas chaves de API
# Copie o arquivo .env.example para .env e preencha suas chaves
copy .env.example .env
notepad .env

# 4. Inicie o GeoBot
.\INICIAR_GEOBOT.bat
```

A aplicação abrirá automaticamente no navegador em `http://localhost:8501` 

## 🚀 Instalação manual
### 1. Crie ambiente virtual Python
```powershell
python -m venv venv
source venv/bin/activate
```

### 2. Instale dependências
```powershell
pip install -r requirements.txt
```

### 3. Instale PyTorch com suporte GPU
```powershell
# Para NVIDIA CUDA 12.4 (Recomendado, porém verifique sua versão CUDA):
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

### 4. Configure suas chaves de API
```powershell
cp .env.example .env
```

### 5. Execute o GeoBot
```powershell
streamlit run geobot.py
```
---

## 🎓 Primeiros Passos

### 1️⃣ Configure sua API Key da Groq

O GeoBot usa a **Groq API** (gratuita!) para conversação com IA:

1. Acesse [console.groq.com/keys](https://console.groq.com/keys)
2. Crie uma conta gratuita
3. Gere uma nova API Key
4. Cole a chave na interface do GeoBot

### 2️⃣ Carregue seus Dados

O GeoBot aceita diversos formatos:

- **CSV/TXT:** Colunas com X, Y (coordenadas) e valor (gravidade/magnetometria)
- **Excel:** Arquivos `.xlsx` ou `.xls`

**Exemplo de CSV:**
```csv
longitude,latitude,gravity
-43.2,25.8,982.5
-43.1,25.9,983.2
...
```

### 3️⃣ Converse com o GeoBot!

Experimente comandos como:

```
"Mostre as estatísticas dos dados"
"Aplique correção de Bouguer"
"Calcule a derivada vertical"
"Faça um histograma da gravidade"
"Qual a diferença entre redução ao polo e sinal analítico?"
```
---

## 📖 Documentação

- Guia de uso para não programadores
- Configuração e manutenção
- Como adicionar novas funções de processamento
- Como atualizar e escalar o RAG
- Boas práticas de interpretação geofísica

---

## 🎯 Exemplos de Uso

### Exemplo 1: Correção de Bouguer

```
Usuário : "Carregue os dados de gravidade"
[Arraste arquivo CSV na sidebar]

Usuário: "Aplique correção de Bouguer com densidade 2.67 g/cm³"

GeoBot: "✅ Correção de Bouguer aplicada!
- Densidade: 2670 kg/m³
- Range original: 980.2 a 985.7 mGal
- Range corrigido: -15.3 a +8.9 mGal
- Tempo de processamento: 0.34s

📖 Referência:
BLAKELY, R. J. Potential Theory in Gravity and Magnetic Applications..."
```

### Exemplo 2: Pipeline de Processamento Magnético

```
Usuário: "Faça redução ao polo seguida de derivada vertical"

GeoBot: "✅ Pipeline executado com sucesso!

1. Redução ao Polo (RTP)
   - Inclinação: -25°, Declinação: -20°
   
2. Derivada Vertical (1ª ordem)
   - Realça bordas rasas
   
[Exibe mapas comparativos]"
```

---

## 🧠 Arquitetura do Sistema

```
┌─────────────────────────────────────────────┐
│           Interface Streamlit               │
│  (Sidebar + Painel de Dados + Chat)         │
└────────────────┬────────────────────────────┘
                 │
┌────────────────┴────────────────────────────┐
│          LLM Manager (Groq API)             │
│  • 5 modelos com fallback automático        │
│  • llama-3.3-70b, llama-3.1-70b, etc.       │
└────────────────┬────────────────────────────┘
                 │
┌────────────────┴────────────────────────────┐
│      RAG Engine (ChromaDB/Supabase)         │
│  • Embeddings: all-MiniLM-L6-v2             │
│  • Vetor local ou remoto (pgvector)         │
│  • Citações científicas automáticas         │
└────────────────┬────────────────────────────┘
                 │
┌────────────────┴────────────────────────────┐
│   Processing Pipeline (NumPy/SciPy)        │
│  • Registro modular de funções              │
│  • Aceleração GPU (PyTorch)                 │
│  • 10+ métodos de processamento             │
└─────────────────────────────────────────────┘
```

---

## 🔬 Processamentos Disponíveis

### Gravimetria
- ✅ Correção de Bouguer (simples e completa)
- ✅ Anomalia ar-livre
- ✅ Remoção de tendência regional

### Magnetometria
- ✅ Redução ao Polo (RTP)
- ✅ Sinal Analítico
- ✅ Ângulo de Tilt

### Geral (Gravimetria + Magnetometria)
- ✅ Continuação ascendente/descendente
- ✅ Derivadas verticais (1ª e 2ª ordem)
- ✅ Derivada horizontal total (THD)
- ✅ Filtros Gaussianos (passa-alta/passa-baixa)
- ✅ Interpolação (linear, cubic, RBF)

---

## 🚀 Aceleração por GPU

O GeoBot detecta automaticamente GPUs disponíveis:

| GPU | Suporte | Ganho de Performance |
|-----|---------|----------------------|
| **NVIDIA** (CUDA) | ✅ Automático | ~10-50x mais rápido |
| **CPU** (Fallback) | ✅ Sempre funciona | Performance padrão |

Para verificar se sua GPU está sendo usada, veja o log de inicialização:

```
🚀 GPU NVIDIA detectada: NVIDIA GeForce RTX 3050 Ti
✅ Módulo de otimizações GPU ativado
```

### 🚀 Performance Comparativa

| Operação | CPU (numpy) | GPU (CUDA) | Speedup |
|----------|-------------|------------|---------|
| **FFT 2D (150×150)** | 120ms | 8ms | **15x** ⚡ |
| **Derivada Vertical** | 250ms | 12ms | **21x** ⚡ |
| **Redução ao Polo** | 450ms | 28ms | **16x** ⚡ |
| **Sinal Analítico** | 380ms | 24ms | **16x** ⚡ |
| **Embeddings (RAG)** | 850ms | 85ms | **10x** ⚡ |
| **Grid Cache** | 2000ms | 2ms | **1000x** 💾 |

---

## 📊 Formatos de Dados Suportados

| Formato | Extensões | Notas |
|---------|-----------|-------|
| **CSV** | `.csv` | Delimitador: `,` ou `;` |
| **TXT** | `.txt` | Espaços ou tabs |
| **Excel** | `.xlsx`, `.xls` | Múltiplas planilhas |
| **Grid** | `.grd`, `.nc` | NetCDF, Surfer |

**Colunas esperadas:**
- Coordenadas: `longitude`, `latitude`, `x`, `y`
- Valores: `gravity`, `bouguer`, `magnetic`, `tmi`, `rtp`

---

### Adicionando Novas Funções

Veja a seção **"Criando novas funções de processamento"** neste README para um passo a passo completo.

**Exemplo rápido:**

```python
@register_processing(
    category="Magnetometria",
    description="Meu novo filtro customizado",
    input_type="grid"
)
def meu_filtro(data: GeophysicalData, param: float) -> ProcessingResult:
    """
    Implementa um filtro X.
    
    Parameters:
    -----------
    data : GeophysicalData
        Dados de entrada
    param : float
        Parâmetro do filtro
    
    Returns:
    --------
    ProcessingResult
        Dados processados
    """
    # Seu código aqui!
    result = ... 
    
    return ProcessingResult(
        processed_data=result,
        original_data=data,
        method_name="meu_filtro"
    )
```
---

# 📚 Documentação de Desenvolvimento

Esta seção consolida todo o conteúdo necessário para **usar**, **manter** e **expandir** o GeoBot.

## 1) Início rápido

1. **Instale** executando INSTALAR.bat
2. **Abra** INICIAR_GEOBOT.bat
3. **Cole a API Key** da Groq quando solicitado
4. **Carregue seus dados** na barra lateral
5. **Converse** com o GeoBot (ex.: “Aplique correção de Bouguer”)


## 2) Configuração do ambiente (.env)

Crie o arquivo .env com base em .env.example e preencha:

- GROQ_API_KEY: obrigatório para o chat com IA
- RAG_BACKEND: chroma (local), supabase (nuvem) ou none (desliga)
- SUPABASE_URL / SUPABASE_KEY / SUPABASE_SERVICE_KEY: se usar Supabase

### Recomendações

- Para uso local simples: RAG_BACKEND=chroma
- Para uso compartilhado/escala: RAG_BACKEND=supabase
- Para desligar o RAG: RAG_BACKEND=none

## 3) Como os dados devem estar organizados

O GeoBot precisa de pelo menos **X**, **Y** e **Valor**.

### Exemplos de colunas aceitas

- X: x, lon, longitude, easting
- Y: y, lat, latitude, northing
- Valor (gravidade): gravity, bouguer, free_air
- Valor (magnetismo): magnetic, tmi, igrf

### Formatos aceitos

- CSV / TXT / Excel
- Grid regular (quando o arquivo já estiver em formato de grid)

Se o nome das colunas estiver diferente, o GeoBot tenta inferir automaticamente.

## 4) Como conversar com o GeoBot

Usuário pode digitar comandos naturais como:

- “Mostre estatísticas dos dados”
- “Faça histograma”
- “Aplique redução ao polo”
- “Continuação ascendente de 1000 m”

O GeoBot detecta a intenção, executa a função, mostra gráficos e adiciona referências científicas.

## 5) RAG (Base de conhecimento científica)

O RAG é o sistema que permite **citações automáticas**. Ele pode rodar:

### 5.1) Modo local (ChromaDB)

1. Coloque PDFs em rag_database/
2. Rode o script de atualização (veja seção 6)
3. Abra o GeoBot normalmente

### 5.2) Modo Supabase (nuvem)

Ideal para equipe ou produção. O GeoBot lê a base que está no Supabase. Quando Usuário roda o script de atualização, a base remota é atualizada e **o GeoBot passa a usar os novos documentos imediatamente nas próximas perguntas**.

## 6) Atualização do RAG (script separado)

O script de atualização é  rag_update.py, exatamente para o usuário rodar quando quiser atualizar a base.

### 6.1) Atualizar Chroma (local)

Exemplo:
```
python rag_update.py --backend chroma --force-reindex
```

### 6.2) Atualizar Supabase (nuvem)

Exemplo:
```
python rag_update.py --backend supabase --force-reindex --clear-existing
```

### 6.3) Parâmetros úteis

- --chunk-size: tamanho dos trechos de texto
- --overlap: sobreposição de palavras

## 7) Configurando o Supabase para RAG

### 7.1) Crie a extensão pgvector

```sql
create extension if not exists vector;
```

### 7.2) Crie a tabela

```sql
create table if not exists rag_documents (
    id text primary key,
    content text not null,
    metadata jsonb,
    embedding vector(384)
);
```

### 7.3) Crie o índice vetorial

```sql
create index if not exists rag_documents_embedding_idx
on rag_documents
using ivfflat (embedding vector_cosine_ops) with (lists = 100);
```

### 7.4) Crie a função de busca (RPC)

```sql
create or replace function match_rag_documents(
    query_embedding vector(384),
    match_count int default 5
)
returns table (
    id text,
    content text,
    metadata jsonb,
    distance float
)
language plpgsql
as $$
begin
    return query
    select
        r.id,
        r.content,
        r.metadata,
        1 - (r.embedding <=> query_embedding) as distance
    from rag_documents r
    order by r.embedding <=> query_embedding
    limit match_count;
end;
$$;
```

> Observação: o modelo all-MiniLM-L6-v2 gera vetores de dimensão 384. Se trocar o modelo, ajuste esse número.

## 8) Criando novas funções de processamento (passo a passo)

### 8.1) Crie a função no arquivo geobot.py

Use o decorador @register_processing e retorne ProcessingResult.

```python
@register_processing(
        category="Gravimetria",
        description="Correção de terreno (exemplo)",
        input_type="grid",
        requires_params=["density"]
)
def terrain_correction(data: GeophysicalData, density: float = 2.67) -> ProcessingResult:
        # 1) Valide entrada
        # 2) Transforme dados
        # 3) Gere figuras
        # 4) Retorne ProcessingResult
        ...
```

### 8.2) Atualize o mapeamento de comandos

No método detect_processing_command, inclua palavras-chave para a nova função.

### 8.3) Boas práticas obrigatórias

- **Validação:** verifique se as colunas necessárias existem
- **Units:** use unidades coerentes (mGal, nT, m)
- **Metadados:** preencha execution_time, parameters e references
- **Erros claros:** use InvalidDataError ou ProcessingError

### 8.4) Checklist rápido

- [ ] Função registrada
- [ ] Função aparece no chat via comando
- [ ] Retorno é ProcessingResult
- [ ] Inclui referências científicas

## 9) Catálogo de possibilidades (para implementar)

No momento, todos os itens listados anteriormente já estão implementados.

## 10) Boas práticas de interpretação

- **Sempre aplique QC** antes de derivadas
- **Derivadas amplificam ruído** → use passa-baixa
- **RTP é instável em baixas latitudes** → prefira ASA/Tilt
- **Compare mapas antes/depois** para evitar artefatos

## 11) Manutenção e escala

### 11.1) Performance
- Use GPU quando disponível
- Prefira grids regulares para FFT
- Evite reprocessar dados sem necessidade

### 11.2) RAG escalável
- Supabase com pgvector
- Use índices ivfflat/hnsw
- Atualize a base via rag_update.py

### 11.3) Crescimento do projeto
- Separe módulos conforme funções crescem
- Crie testes unitários para cada processamento
- Versione os dados e mantenha changelog

## 12) Solução de problemas

- **RAG não retorna citações:** verifique RAG_BACKEND e PDFs
- **Supabase não conecta:** confira SUPABASE_URL/KEY
- **Processamento falha:** verifique se há colunas X/Y e valor
- **Resultados estranhos:** revise unidades e CRS

---

## 📄 Licença

Este projeto está licenciado sob a **MIT License** - veja [LICENSE](LICENSE) para detalhes.

Você pode usar, modificar e distribuir livremente, desde que mantenha os créditos originais.

---

## 👥 Autores

Desenvolvido por:
- **Allan Ramalho** - Geofísico, Cientista de Dados e Mestrando em Geofísica
- **Dr. Rodrigo Bijani** - Professor Orientador

**Instituição:** Programa de Pós-Graduação em Dinâmica dos Oceanos e da Terra (PPG DOT) - Universidade Federal Fluminense (UFF)

---

## 📞 Contato

- 📧 Email: [allansoares@id.uff.br](mailto:allansoares@id.uff.br)
- 📧 Email: [rodrigobijani@id.uff.br](mailto:rodrigobijani@id.uff.br)
- 💬 Issues: [GitHub Issues](https://github.com/allan-ramalho/GeoBot_mestrado/issues)
- 🐙 GitHub: [@allan-ramalho](https://github.com/allan-ramalho)

---

<div align="center">

**[⬆ Voltar ao topo](#-geobot---agente-de-ia-para-processamento-geofísico)**

</div>
