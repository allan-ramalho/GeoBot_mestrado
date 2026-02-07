# 📚 GeoBot - Documentação Completa de Uso e Manutenção

> **Versão:** 1.0.0  
> **Data:** Fevereiro 2025  
> **Autor:** Allan Ramalho  
> **Python:** 3.11.9  
> **Framework:** Streamlit 1.31.1

---

## 📑 Índice

1. [Visão Geral](#1-visão-geral)
2. [Instalação e Configuração](#2-instalação-e-configuração)
3. [Guia de Uso](#3-guia-de-uso)
4. [Arquitetura do Sistema](#4-arquitetura-do-sistema)
5. [Adicionar Novas Funções de Processamento](#5-adicionar-novas-funções-de-processamento)
6. [Manutenção e Troubleshooting](#6-manutenção-e-troubleshooting)
7. [Sistema RAG](#7-sistema-rag)
8. [Referências Científicas](#8-referências-científicas)

---

## 1. Visão Geral

### 1.1 O que é o GeoBot?

O **GeoBot** é um agente conversacional de IA desenvolvido para processar e analisar dados geofísicos de métodos potenciais (gravimetria e magnetometria). Combina:

- ✅ **LLM (Groq API)** - Conversação natural e interpretação de dados
- ✅ **RAG (Retrieval-Augmented Generation)** - Citações científicas automáticas em formato ABNT
- ✅ **Processamento Geofísico** - Correções, filtros e transformações clássicas
- ✅ **Visualização Interativa** - Plots 2D/3D com Plotly
- ✅ **Pipeline Modular** - Sistema de registro de funções extensível

### 1.2 Características Principais

| Característica | Descrição |
|----------------|-----------|
| **Arquitetura** | Monolítica (arquivo único `geobot.py`) |
| **Interface** | Streamlit com tema profissional |
| **LLM Provider** | Groq API com fallback automático entre 5 modelos |
| **Embeddings** | SentenceTransformers (all-MiniLM-L6-v2) |
| **Vector Store** | ChromaDB persistente |
| **Processamento** | NumPy, SciPy, Harmonica (Fatiando a Terra) |
| **Dados** | Polars (alta performance) + Pandas (compatibilidade) |
| **Formatos Aceitos** | CSV, TXT, Excel, grids geofísicos |
| **Performance** | Otimizado para até 1M de pontos |

### 1.3 Estrutura de Arquivos

```
GeoBot_Mestrado/
│
├── geobot.py                    # 🎯 APLICAÇÃO PRINCIPAL
├── requirements.txt             # Dependências Python
├── generate_example_data.py     # Gerador de dados sintéticos
│
├── INSTALAR.bat                 # Script de instalação Windows
├── INICIAR_GEOBOT.bat          # Script de execução Windows
├── LICENSE                      # Licença MIT
├── DOCUMENTACAO.md              
│
├── .streamlit/
│   └── config.toml              # Tema da interface
│
├── example_data/
│   ├── README.md                # Documentação dos datasets
│   ├── gravity_basin_example.csv       # 10,000 pontos
│   ├── magnetic_dike_example.csv       # 6,400 pontos
│   └── gravity_profile_sphere.csv      # 100 pontos
│
├── rag_database/
│   ├── README.md                # Instruções para adicionar papers
│   └── chromadb/                # Banco vetorial (criado automaticamente)
│
└── venv/                        # Ambiente virtual (criado na instalação)
```

---

## 2. Instalação e Configuração

### 2.1 Requisitos de Sistema

- **Sistema Operacional:** Windows 10/11, Linux, macOS
- **Python:** 3.11.9 (obrigatório)
- **Memória RAM:** Mínimo 4GB, recomendado 8GB+
- **Espaço em Disco:** 2GB (instalação completa)
- **Conexão Internet:** Necessária para API Groq e download de modelos

### 2.2 Instalação Rápida (Windows)

#### Passo 1: Verificar Python

```powershell
python --version
```

Deve retornar `Python 3.11.9`. Se não tiver, baixe em [python.org](https://www.python.org/downloads/release/python-3119/).

#### Passo 2: Executar Instalação Automática

Clique duas vezes em `INSTALAR.bat` ou execute no terminal:

```powershell
.\INSTALAR.bat
```

**O que o script faz:**
1. Valida versão do Python
2. Cria ambiente virtual em `venv/`
3. Instala todas as 80+ dependências
4. Configura ChromaDB
5. Gera dados de exemplo

#### Passo 3: Obter API Key do Groq

1. Acesse: https://console.groq.com/keys
2. Faça login (gratuito)
3. Clique em "Create API Key"
4. Copie a chave (começa com `gsk_...`)

**⚠️ IMPORTANTE:** Nunca compartilhe sua API key publicamente!

#### Passo 4: Iniciar GeoBot

Clique duas vezes em `INICIAR_GEOBOT.bat` ou:

```powershell
.\INICIAR_GEOBOT.bat
```

O navegador abrirá automaticamente em `http://localhost:8501`

### 2.3 Instalação Manual (Linux/macOS)

```bash
# 1. Clone ou extraia o projeto
cd GeoBot_Mestrado

# 2. Crie ambiente virtual
python3.11 -m venv venv

# 3. Ative o ambiente
source venv/bin/activate  # Linux/macOS
# ou
.\venv\Scripts\activate   # Windows PowerShell

# 4. Instale dependências
pip install --upgrade pip
pip install -r requirements.txt

# 5. Gere dados de exemplo
python generate_example_data.py

# 6. Inicie aplicação
streamlit run geobot.py
```

### 2.4 Configuração do Tema

O arquivo `.streamlit/config.toml` controla a aparência:

```toml
[theme]
primaryColor = "#1E88E5"           # Azul principal
backgroundColor = "#FFFFFF"         # Fundo branco
secondaryBackgroundColor = "#F5F5F5"  # Cinza claro
textColor = "#212121"              # Texto escuro
font = "sans-serif"
```

Para customizar, edite o arquivo e reinicie o GeoBot.

---

## 3. Guia de Uso

### 3.1 Primeira Execução

#### Tela 1: Landing Page

1. **Inserir API Key Groq:**
   - Cole sua chave no campo
   - Clique em "Confirmar e Continuar"
   - A chave é validada automaticamente

2. **Explorar Documentação Inline:**
   - Funcionalidades disponíveis
   - Tipos de dados aceitos
   - Processamentos registrados

#### Tela 2: Seleção de Modelo

1. **Escolher Modelo LLM:**
   - `llama-3.3-70b-versatile` (recomendado) - Melhor qualidade
   - `llama-3.1-70b-versatile` - Alternativa robusta
   - `mixtral-8x7b-32768` - Contexto longo
   - `llama-3.1-8b-instant` - Mais rápido
   - `gemma2-9b-it` - Fallback

2. **Sistema de Fallback:**
   - Se o modelo atingir rate limit, troca automaticamente
   - Contexto da conversa é preservado
   - Você será notificado da mudança

#### Tela 3: Interface Principal

Dividida em **Sidebar** (esquerda) e **Chat** (direita).

### 3.2 Carregar Dados

#### Sidebar: Upload de Arquivo

1. **Clique em "Carregar dados geofísicos"**
2. **Selecione arquivo:** CSV, TXT ou Excel
3. **Detecção Automática:**
   - Tipo de dado (gravimetria/magnetometria)
   - Colunas de coordenadas (X, Y, Z)
   - Coluna de valores
   - Dimensionalidade (1D, 2D, 3D)

#### Formato Esperado (CSV exemplo):

```csv
longitude,latitude,elevation,gravity
-45.5231,-23.1234,150.5,-35.2
-45.5232,-23.1235,148.3,-34.8
-45.5233,-23.1236,152.1,-35.6
```

**Regras:**
- Primeira linha: nomes de colunas
- Coordenadas: `x`, `y`, `z` ou `lon`, `lat`, `elevation`
- Valores: `gravity`, `magnetic`, `anomaly`, `value`
- Delimitadores aceitos: `,` `;` `\t` (tab) ou espaços

#### Estatísticas Exibidas

Após carregar, a sidebar mostra:
- **Tipo de Dado:** Gravimetria/Magnetometria
- **Dimensão:** 1D/2D/3D
- **Número de Pontos:** Total de observações
- **Estatísticas:** Média, desvio padrão, min, max
- **Mapa Interativo:** Scatter plot das coordenadas

### 3.3 Conversar com o GeoBot

#### Exemplos de Perguntas:

**Análise Exploratória:**
```
- Analise os dados carregados
- Qual a distribuição de valores?
- Existem outliers significativos?
```

**Processamento:**
```
- Aplique correção de Bouguer com densidade 2670 kg/m³
- Faça continuação ascendente para 500 metros
- Calcule a derivada vertical de segunda ordem
```

**Interpretação:**
```
- O que estas anomalias indicam geologicamente?
- Qual a profundidade estimada das fontes?
- Compare com modelos de bacia sedimentar
```

**Pipeline Complexo:**
```
- Aplique os seguintes processamentos em sequência:
  1. Correção de Bouguer (densidade 2500)
  2. Continuação ascendente (1000m)
  3. Gere plots comparativos
```

### 3.4 Trabalhar com Resultados

#### Visualizações Geradas

Cada processamento retorna:

1. **Plots Comparativos:**
   - Dado original
   - Dado processado
   - Diferença (residual)

2. **Histogramas:**
   - Distribuição antes/depois
   - Análise estatística

3. **Mapas Interativos (se 2D/3D):**
   - Heatmaps com colorbar
   - Zoom e pan
   - Exportável como HTML

#### Exportar Dados

No chat, peça:
```
Exporte os dados processados como CSV
```

O bot gerará um link de download.

---

## 4. Arquitetura do Sistema

### 4.1 Estrutura do Código `geobot.py`

O arquivo é organizado em **seções lógicas** de ~150 linhas cada:

| Linhas | Seção | Descrição |
|--------|-------|-----------|
| 1-91 | **Imports e Configurações** | Bibliotecas, constantes globais, logging |
| 92-180 | **Exceções e Registry** | `@register_processing` decorator |
| 181-450 | **Classes de Domínio** | `GeophysicalData`, `ProcessingResult`, `ProcessingPipeline` |
| 451-650 | **RAG e LLM** | `RAGEngine`, `LLMManager` com fallback |
| 651-1100 | **Funções de Processamento** | `bouguer_correction`, `upward_continuation` |
| 1101-1500 | **Utilitários** | `detect_data_type`, `parse_uploaded_file` |
| 1501-1928 | **Interface Streamlit** | Landing, model selection, chat |

### 4.2 Classes Principais

#### 4.2.1 `GeophysicalData`

**Propósito:** Encapsula dados geofísicos com metadados.

**Atributos:**
```python
@dataclass
class GeophysicalData:
    data: pl.DataFrame           # Dados em Polars
    data_type: str               # 'gravity', 'magnetic', 'topography'
    dimension: str               # '1D', '2D', '3D'
    coords: Dict[str, str]       # {'x': 'longitude', 'y': 'latitude'}
    value_column: str            # Nome da coluna de valores
    units: str                   # 'mGal', 'nT', 'SI'
    crs: str                     # Sistema de coordenadas (EPSG)
    metadata: Dict[str, Any]     # Estatísticas e info adicional
```

**Métodos:**
- `to_pandas()` - Converte para Pandas DataFrame
- `to_grid(method='linear')` - Interpola para grid regular
- `_compute_stats()` - Calcula estatísticas automáticas

**Exemplo de Uso:**
```python
geo_data = GeophysicalData(
    data=df,
    data_type='gravity',
    dimension='2D',
    coords={'x': 'lon', 'y': 'lat', 'z': 'elev'},
    value_column='bouguer',
    units='mGal'
)

# Acessar estatísticas
print(geo_data.metadata['mean'])  # Média
print(geo_data.metadata['bbox'])  # Bounding box
```

#### 4.2.2 `ProcessingResult`

**Propósito:** Retorna resultados completos de processamento.

**Atributos:**
```python
@dataclass
class ProcessingResult:
    processed_data: GeophysicalData    # Dados processados
    original_data: GeophysicalData     # Dados originais
    method_name: str                   # Nome do método
    parameters: Dict[str, Any]         # Parâmetros utilizados
    figures: List[go.Figure]           # Figuras Plotly
    explanation: str                   # Explicação técnica
    execution_time: float              # Tempo em segundos
    references: List[str]              # Citações ABNT
```

**Métodos:**
- `summary()` - Retorna dict com sumário

#### 4.2.3 `ProcessingPipeline`

**Propósito:** Gerencia sequência de processamentos.

**Exemplo:**
```python
pipeline = ProcessingPipeline(initial_data)
pipeline.add_step('bouguer_correction', density=2670)
pipeline.add_step('upward_continuation', height=1000)
results = pipeline.execute()

# Sumário completo
print(pipeline.get_summary())
```

#### 4.2.4 `RAGEngine`

**Propósito:** Sistema RAG para citações científicas.

**Workflow:**
1. PDFs na pasta `rag_database/` são parseados
2. Textos divididos em chunks de 500 palavras
3. Embeddings gerados com SentenceTransformer
4. Armazenados no ChromaDB
5. Busca semântica retorna trechos relevantes

**Métodos:**
```python
rag = RAGEngine()
rag.initialize()
rag.index_documents()  # Indexa PDFs

# Buscar contexto
results = rag.search("correção de Bouguer", top_k=3)
for r in results:
    print(r['document'])
    print(r['metadata'])
```

#### 4.2.5 `LLMManager`

**Propósito:** Comunicação com Groq API + fallback.

**Recursos:**
- Fila de modelos alternativos
- Detecção automática de rate limit
- Preservação de contexto na troca
- Histórico de fallbacks

**Exemplo:**
```python
llm = LLMManager(api_key="gsk_...")
response = llm.chat_completion(
    messages=[
        {"role": "system", "content": "Você é um geofísico."},
        {"role": "user", "content": "Explique Bouguer"}
    ],
    temperature=0.7
)
```

### 4.3 Sistema de Registro de Funções

#### Decorator `@register_processing`

**Propósito:** Auto-registro de funções para descoberta pelo LLM.

**Parâmetros:**
- `category` - Categoria ('Gravimetria', 'Magnetometria', 'Geral')
- `description` - Descrição curta
- `input_type` - Tipo esperado ('grid', 'profile', 'points')
- `requires_params` - Lista de parâmetros obrigatórios

**Funcionamento:**
```python
PROCESSING_REGISTRY = {}  # Dict global

@register_processing(
    category="Gravimetria",
    description="Correção de Bouguer",
    requires_params=['density']
)
def bouguer_correction(data: GeophysicalData, density: float) -> ProcessingResult:
    # Implementação
    pass
```

Após registro, a função fica em `PROCESSING_REGISTRY`:
```python
{
    'bouguer_correction': {
        'function': <function>,
        'category': 'Gravimetria',
        'description': 'Correção de Bouguer',
        'requires_params': ['density'],
        'signature': inspect.signature(...),
        'docstring': '...'
    }
}
```

**Vantagens:**
- LLM pode listar funções disponíveis dinamicamente
- Adicionar nova função não requer alterar UI
- Autodocumentação via docstrings
- Type hints garantem contratos

---

## 5. Adicionar Novas Funções de Processamento

### 5.1 Template Completo

**Exemplo: Derivada Vertical de 1ª Ordem**

```python
@register_processing(
    category="Geral",
    description="Derivada vertical de 1ª ordem (dU/dz)",
    input_type="grid",
    requires_params=[]
)
def vertical_derivative(data: GeophysicalData) -> ProcessingResult:
    """
    Derivada Vertical de Primeira Ordem
    
    Calcula a taxa de variação do campo potencial na direção vertical.
    É equivalente à continuação ascendente negativa e realça anomalias rasas.
    
    No domínio da frequência:
        F{dU/dz} = F{U} × |k|
    
    Onde:
        F{} = transformada de Fourier 2D
        k = número de onda = sqrt(kx² + ky²)
    
    Aplicações:
    -----------
    - Realce de bordas de corpos
    - Estimativa de profundidade (regra de Peters)
    - Delineamento de contatos geológicos
    
    Limitações:
    -----------
    - Amplifica ruído de alta frequência
    - Requer grid regular
    - Sensível a qualidade do gridding
    
    Referências:
    ------------
    BLAKELY, R. J. **Potential Theory in Gravity and Magnetic Applications**. 
    Cambridge University Press, 1995. p. 320-325. ISBN: 978-0521575478
    
    NABIGHIAN, M. N. et al. **The historical development of the magnetic method 
    in exploration**. Geophysics, v. 70, n. 6, p. 33ND-61ND, 2005. 
    DOI: 10.1190/1.2133784
    
    Parameters:
    -----------
    data : GeophysicalData
        Dados em grid regular
    
    Returns:
    --------
    ProcessingResult
        Derivada vertical com figuras
    
    Examples:
    ---------
    >>> result = vertical_derivative(magnetic_data)
    >>> result.processed_data.to_pandas()
    """
    start_time = datetime.now()
    
    try:
        # 1. Interpola para grid regular
        Xi, Yi, Zi = data.to_grid(method='linear')
        ny, nx = Zi.shape
        
        # 2. Remove NaN (se houver)
        mask = np.isnan(Zi)
        if mask.any():
            logger.warning(f"{mask.sum()} NaN encontrados, interpolando...")
            from scipy.ndimage import distance_transform_edt
            indices = distance_transform_edt(
                mask, 
                return_distances=False, 
                return_indices=True
            )
            Zi[mask] = Zi[tuple(indices[:, mask])]
        
        # 3. Calcula espaçamento
        dx = (Xi.max() - Xi.min()) / (nx - 1)
        dy = (Yi.max() - Yi.min()) / (ny - 1)
        
        # 4. Números de onda
        kx = 2 * np.pi * fftfreq(nx, d=dx)
        ky = 2 * np.pi * fftfreq(ny, d=dy)
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX**2 + KY**2)
        
        # 5. Transformada de Fourier
        F = fft2(Zi)
        
        # 6. Aplica operador de derivada
        # dU/dz = F^-1{ F{U} × |k| }
        F_deriv = F * K
        
        # 7. Transformada inversa
        Zi_deriv = np.real(ifft2(F_deriv))
        
        # 8. Converte de volta para pontos
        x_flat = Xi.flatten()
        y_flat = Yi.flatten()
        z_deriv_flat = Zi_deriv.flatten()
        
        x_col = data.coords['x']
        y_col = data.coords['y']
        
        deriv_df = pl.DataFrame({
            x_col: x_flat,
            y_col: y_flat,
            f"{data.value_column}_dz": z_deriv_flat
        })
        
        # 9. Cria GeophysicalData de saída
        processed_data = GeophysicalData(
            data=deriv_df,
            data_type=data.data_type,
            dimension=data.dimension,
            coords=data.coords,
            value_column=f"{data.value_column}_dz",
            units=f"{data.units}/m",  # Unidade muda
            crs=data.crs,
            metadata={
                **data.metadata,
                'processing': 'vertical_derivative',
                'grid_spacing': f"{dx:.2f} x {dy:.2f}"
            }
        )
        
        # 10. Gera figuras
        figures = create_comparison_plots(
            data, 
            processed_data, 
            "Derivada Vertical (dU/dz)"
        )
        
        # 11. Explicação técnica
        explanation = f"""
### 📊 Derivada Vertical Aplicada

**Parâmetros:**
- Dimensão do grid: {ny} × {nx}
- Espaçamento: {dx:.2f} × {dy:.2f} m

**Domínio da Frequência:**
- Número de onda máximo: {K.max():.6f} rad/m
- Comprimento de onda mínimo: {2*np.pi/K.max():.1f} m

**Resultado:**
- Campo original: {Zi.min():.2f} a {Zi.max():.2f} {data.units}
- Derivada vertical: {Zi_deriv.min():.3f} a {Zi_deriv.max():.3f} {data.units}/m
- Realce de bordas: {(Zi_deriv.std()/Zi.std()):.2f}x

A derivada vertical realça anomalias rasas e bordas de corpos, sendo útil
para delineamento estrutural e interpretação qualitativa.

**⚠️ Atenção:** Amplifica ruído. Considere suavizar antes de aplicar.
"""
        
        # 12. Referências
        references = [
            "BLAKELY, R. J. **Potential Theory in Gravity and Magnetic Applications**. Cambridge University Press, 1995. p. 320-325. ISBN: 978-0521575478",
            "NABIGHIAN, M. N. et al. **The historical development of the magnetic method in exploration**. Geophysics, v. 70, n. 6, p. 33ND-61ND, 2005. DOI: 10.1190/1.2133784"
        ]
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return ProcessingResult(
            processed_data=processed_data,
            original_data=data,
            method_name="vertical_derivative",
            parameters={},
            figures=figures,
            explanation=explanation,
            execution_time=execution_time,
            references=references
        )
        
    except Exception as e:
        logger.error(f"Erro na derivada vertical: {str(e)}")
        raise ProcessingError(f"Falha na derivada vertical: {str(e)}")
```

### 5.2 Checklist de Implementação

Ao adicionar uma nova função, siga:

- [ ] **1. Decorador:**
  - Usar `@register_processing` com categoria, descrição, parâmetros
  
- [ ] **2. Type Hints:**
  - `data: GeophysicalData` obrigatório
  - Outros parâmetros com tipos explícitos
  - Retorno: `ProcessingResult`

- [ ] **3. Docstring Completa:**
  - Descrição do método
  - Fundamento teórico (fórmulas)
  - Aplicações práticas
  - Limitações
  - Referências científicas em ABNT
  - Parâmetros documentados
  - Exemplos de uso

- [ ] **4. Logging:**
  - `logger.info` para início
  - `logger.warning` para casos especiais
  - `logger.error` para falhas

- [ ] **5. Tratamento de Erros:**
  - Try-except ao redor do código
  - Levantar `ProcessingError` com mensagens claras

- [ ] **6. Validações:**
  - Verificar tipo de dado compatível
  - Validar parâmetros (ranges, tipos)
  - Checar dimensionalidade

- [ ] **7. Processamento:**
  - Implementar algoritmo
  - Calcular tempo de execução
  - Gerar dados de saída

- [ ] **8. Visualizações:**
  - Usar `create_comparison_plots()`
  - Adicionar plots específicos se necessário

- [ ] **9. Explicação:**
  - String Markdown formatada
  - Estatísticas antes/depois
  - Interpretação dos resultados

- [ ] **10. Metadados:**
  - Adicionar info no `metadata` do resultado
  - Preservar rastreabilidade

### 5.3 Localização no Código

Adicione novas funções na seção de processamentos:

```python
# geobot.py, após linha 1100
# ============================================================================
# FUNÇÕES DE PROCESSAMENTO GEOFÍSICO
# ============================================================================

# Funções existentes...
def bouguer_correction(...):
    pass

def upward_continuation(...):
    pass

# ↓ ADICIONAR NOVAS FUNÇÕES AQUI ↓
def vertical_derivative(...):
    pass

def horizontal_derivative(...):
    pass

def matched_filter(...):
    pass
```

### 5.4 Testar Nova Função

#### 5.4.1 Teste Isolado

Crie script `test_new_function.py`:

```python
import polars as pl
from geobot import GeophysicalData, vertical_derivative

# Dados sintéticos
df = pl.DataFrame({
    'x': range(100),
    'y': range(100),
    'gravity': [i*0.1 for i in range(100)]
})

data = GeophysicalData(
    data=df,
    data_type='gravity',
    dimension='1D',
    coords={'x': 'x', 'y': 'y'},
    value_column='gravity',
    units='mGal'
)

# Executar
result = vertical_derivative(data)

# Verificar
print(f"Executado em {result.execution_time:.3f}s")
print(f"Figuras geradas: {len(result.figures)}")
print(result.summary())
```

#### 5.4.2 Teste no GeoBot

1. Execute `streamlit run geobot.py`
2. Carregue dados de exemplo
3. No chat, digite: `Aplique derivada vertical`
4. Verifique:
   - Plots gerados
   - Explicação técnica
   - Referências citadas
   - Tempo de execução

### 5.5 Funções Avançadas: Parâmetros Múltiplos

**Exemplo: Matched Filter**

```python
@register_processing(
    category="Geral",
    description="Matched filter para separação regional-residual",
    input_type="grid",
    requires_params=['wavelength_min', 'wavelength_max']
)
def matched_filter(
    data: GeophysicalData,
    wavelength_min: float,
    wavelength_max: float,
    order: int = 1
) -> ProcessingResult:
    """
    Matched Filter (Filtro Passa-Banda)
    
    Isola componentes do campo em faixa de comprimentos de onda específica.
    Útil para separar anomalias de diferentes profundidades.
    
    Parameters:
    -----------
    data : GeophysicalData
        Dados em grid
    wavelength_min : float
        Comprimento de onda mínimo (metros)
    wavelength_max : float
        Comprimento de onda máximo (metros)
    order : int
        Ordem do filtro Butterworth (padrão: 1)
    
    Returns:
    --------
    ProcessingResult
    """
    # Implementação...
    pass
```

**Uso no chat:**
```
Aplique matched filter com:
- Comprimento de onda mínimo: 500m
- Comprimento de onda máximo: 5000m
- Ordem: 2
```

---

## 6. Manutenção e Troubleshooting

### 6.1 Logs do Sistema

#### Localização

- **Console:** Saída padrão durante execução
- **Arquivo:** `geobot.log` na raiz do projeto

#### Níveis de Log

```python
logger.debug("Detalhes técnicos")     # Desenvolvimento
logger.info("Operação normal")        # Informações
logger.warning("Atenção necessária")  # Avisos
logger.error("Erro crítico")          # Erros
logger.success("Operação bem-sucedida")  # Confirmação
```

#### Configuração de Logs

No início de `geobot.py`:

```python
# Logs no console
logger.add(
    sys.stderr,
    format="<green>{time}</green> | <level>{level}</level> | {message}",
    level="INFO"  # ← Altere para "DEBUG" para mais detalhes
)

# Logs em arquivo
logger.add(
    "geobot.log",
    rotation="10 MB",    # Rotação ao atingir 10MB
    retention="7 days",  # Mantém últimos 7 dias
    level="DEBUG"
)
```

### 6.2 Problemas Comuns

#### 6.2.1 Erro: "API Key inválida"

**Sintoma:**
```
❌ Erro ao validar API Key: Invalid API Key
```

**Soluções:**
1. Verificar se key começa com `gsk_`
2. Regenerar key em console.groq.com
3. Verificar cota de requisições

#### 6.2.2 Erro: "ModuleNotFoundError"

**Sintoma:**
```
ModuleNotFoundError: No module named 'polars'
```

**Soluções:**
1. Ativar ambiente virtual:
   ```powershell
   .\venv\Scripts\activate
   ```
2. Reinstalar dependências:
   ```powershell
   pip install -r requirements.txt
   ```

#### 6.2.3 Erro: "Port 8501 already in use"

**Sintoma:**
```
OSError: [Errno 98] Address already in use
```

**Soluções:**

Windows:
```powershell
# Encontrar processo
netstat -ano | findstr :8501

# Matar processo (substitua PID)
taskkill /PID <PID> /F
```

Linux/macOS:
```bash
# Encontrar e matar
lsof -ti:8501 | xargs kill -9
```

#### 6.2.4 Erro: "ChromaDB initialization failed"

**Sintoma:**
```
RAGError: Falha na inicialização: ...
```

**Soluções:**
1. Deletar banco corrompido:
   ```powershell
   Remove-Item -Recurse rag_database/chromadb
   ```
2. Reiniciar aplicação (recria automaticamente)

#### 6.2.5 Performance Lenta

**Sintomas:**
- Upload lento
- Processamento demorado
- Interface travando

**Soluções:**

1. **Reduzir tamanho do dataset:**
   ```python
   # Downsample antes de carregar
   df = df.sample(n=10000)  # Limita a 10k pontos
   ```

2. **Otimizar gridding:**
   ```python
   # Em to_grid(), reduzir resolução
   xi = np.linspace(x.min(), x.max(), 50)  # Ao invés de 100
   ```

3. **Desabilitar logs verbose:**
   ```python
   logger.remove()
   logger.add(sys.stderr, level="WARNING")
   ```

4. **Usar Polars ao máximo:**
   ```python
   # EVITAR conversão desnecessária para Pandas
   df_polars.to_pandas()  # ❌ Lento
   
   # PREFERIR operações Polars nativas
   df_polars.select([...])  # ✅ Rápido
   ```

### 6.3 Atualizar Dependências

#### Verificar Versões

```powershell
pip list --outdated
```

#### Atualizar Pacote Específico

```powershell
pip install --upgrade streamlit
```

#### Atualizar Tudo (Cuidado!)

```powershell
pip install --upgrade -r requirements.txt
```

**⚠️ Atenção:** Pode quebrar compatibilidade. Teste antes!

### 6.4 Backup e Restauração

#### Backup Completo

```powershell
# Windows
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
Compress-Archive -Path . -DestinationPath "..\GeoBot_backup_$timestamp.zip"
```

```bash
# Linux/macOS
tar -czf ../GeoBot_backup_$(date +%Y%m%d_%H%M%S).tar.gz .
```

#### Backup Apenas Dados Importantes

```powershell
# Salvar apenas:
# - Dados de exemplo customizados
# - PDFs científicos
# - Logs relevantes
# - Configurações

$items = "example_data", "rag_database", "geobot.log", ".streamlit"
Compress-Archive -Path $items -DestinationPath "GeoBot_data_backup.zip"
```

### 6.5 Limpeza de Cache

#### ChromaDB

```powershell
Remove-Item -Recurse -Force rag_database/chromadb
```

#### Streamlit

```powershell
Remove-Item -Recurse -Force $env:USERPROFILE\.streamlit
```

#### Python Cache

```powershell
Get-ChildItem -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force
```

---

## 7. Sistema RAG

### 7.1 Adicionar Papers Científicos

#### Passo 1: Obter PDFs

Fontes confiáveis:
- **Google Scholar** - scholar.google.com
- **ScienceDirect** - sciencedirect.com
- **IEEE Xplore** - ieeexplore.ieee.org
- **ResearchGate** - researchgate.net

#### Passo 2: Organizar na Pasta

```
rag_database/
├── README.md
├── geophysics/
│   ├── Blakely_1995_Potential_Theory.pdf
│   ├── Telford_1990_Applied_Geophysics.pdf
│   └── Nabighian_2005_Magnetic_Method.pdf
├── signal_processing/
│   ├── Oppenheim_DSP.pdf
│   └── Smith_1997_Signal_Processing.pdf
└── chromadb/
    └── (gerado automaticamente)
```

**Dica:** Organize por subpastas (geofísica, processamento, interpretação).

#### Passo 3: Indexar Documentos

**Automático na primeira execução:**
```python
# Dentro de geobot.py
rag = RAGEngine()
rag.initialize()
rag.index_documents()  # Varre rag_database/*.pdf
```

**Manual via Python:**
```python
from geobot import RAGEngine

rag = RAGEngine()
rag.initialize()
rag.index_documents(force_reindex=True)  # Reindexar tudo
```

#### Passo 4: Verificar Indexação

No chat do GeoBot:
```
Quantos documentos estão na base RAG?
```

Ou via código:
```python
rag = RAGEngine()
rag.initialize()
print(f"Documentos: {rag.collection.count()}")
```

### 7.2 Melhorar Qualidade das Citações

#### 7.2.1 Metadados nos PDFs

Ao adicionar PDF, nomeie de forma descritiva:
```
✅ Blakely_1995_Potential_Theory_Gravity_Magnetic.pdf
❌ paper1.pdf
```

#### 7.2.2 Ajustar Tamanho dos Chunks

Em `RAGEngine._split_text()`:

```python
def _split_text(self, text: str, chunk_size: int = 500, overlap: int = 50):
    # chunk_size: palavras por chunk
    # overlap: palavras de sobreposição
    
    # Para papers densos, reduza chunk_size
    chunk_size = 300  # Chunks menores, mais precisos
    
    # Para livros, aumente overlap
    overlap = 100  # Mais contexto entre chunks
```

#### 7.2.3 Top-k Resultados

Em `RAGEngine.search()`:

```python
results = rag.search("correção de Bouguer", top_k=5)  # Retorna 5 resultados
```

Aumentar `top_k` = mais contexto, mas pode diluir relevância.

### 7.3 Formato ABNT das Citações

#### Template Atual

```python
def format_citation_abnt(self, metadata: Dict, text_snippet: str = "") -> str:
    source = metadata.get('source', 'Documento desconhecido')
    
    citation = f"""
> 📚 **Referência:**
> **{source}**
"""
    
    if text_snippet:
        citation += f"""
> *Trecho relevante:*
> "{text_snippet[:200]}..."
"""
    
    return citation
```

#### Customizar para ABNT Completo

```python
def format_citation_abnt(self, metadata: Dict, text_snippet: str = "") -> str:
    """
    Formata citação em ABNT completo.
    
    Requer metadados:
    - author: Autor (SOBRENOME, Nome)
    - title: Título do trabalho
    - year: Ano
    - publisher: Editora
    - doi: DOI (opcional)
    """
    author = metadata.get('author', 'AUTOR DESCONHECIDO')
    title = metadata.get('title', metadata.get('source', 'Título desconhecido'))
    year = metadata.get('year', 's.d.')
    publisher = metadata.get('publisher', '')
    doi = metadata.get('doi', '')
    
    # ABNT: AUTOR. Título. Editora, ano. DOI (se houver)
    citation = f"{author}. **{title}**. "
    if publisher:
        citation += f"{publisher}, "
    citation += f"{year}."
    if doi:
        citation += f" DOI: {doi}"
    
    return f"> 📚 {citation}\n"
```

**Para usar:** Adicione metadados ao indexar PDFs manualmente.

---

## 8. Referências Científicas

### 8.1 Livros Fundamentais

**BLAKELY, R. J.** *Potential Theory in Gravity and Magnetic Applications*.  
Cambridge University Press, 1995. 441p. ISBN: 978-0521575478

**TELFORD, W. M.; GELDART, L. P.; SHERIFF, R. E.** *Applied Geophysics*.  
2nd ed. Cambridge University Press, 1990. 770p. ISBN: 978-0521339384

**HINZE, W. J.; VON FRESE, R. R. B.; SAAD, A. H.** *Gravity and Magnetic Exploration*.  
Cambridge University Press, 2013. 512p. ISBN: 978-0521871013

**SHERIFF, R. E.; GELDART, L. P.** *Exploration Seismology*.  
2nd ed. Cambridge University Press, 1995. 592p. ISBN: 978-0521468268

### 8.2 Artigos Seminais

**NABIGHIAN, M. N. et al.** The historical development of the magnetic method in exploration.  
*Geophysics*, v. 70, n. 6, p. 33ND-61ND, 2005. DOI: 10.1190/1.2133784

**JACOBSEN, B. H.** A case for upward continuation as the standard separation filter for potential-field maps.  
*Geophysics*, v. 52, n. 8, p. 1138-1148, 1987. DOI: 10.1190/1.1442378

**CORDELL, L.; GRAUCH, V. J. S.** Mapping basement magnetization zones from aeromagnetic data in the San Juan Basin, New Mexico.  
*SEG Technical Program Expanded Abstracts*, p. 181-183, 1985. DOI: 10.1190/1.1892795

### 8.3 Software e Bibliotecas

**Fatiando a Terra** - Python library for geophysical modeling and inversion.  
https://www.fatiando.org/  
Uieda et al. (2013). DOI: 10.5281/zenodo.157746

**PyGMT** - Python interface for the Generic Mapping Tools.  
https://www.pygmt.org/  
Uieda et al. (2021). DOI: 10.5281/zenodo.4592991

**SimPEG** - Simulation and Parameter Estimation in Geophysics.  
https://simpeg.xyz/  
Cockett et al. (2015). DOI: 10.1016/j.cageo.2015.09.015

### 8.4 Documentação Técnica

**Groq API Documentation**  
https://console.groq.com/docs

**Streamlit Documentation**  
https://docs.streamlit.io/

**Polars User Guide**  
https://pola-rs.github.io/polars-book/

**ChromaDB Documentation**  
https://docs.trychroma.com/

**Sentence-Transformers**  
https://www.sbert.net/

---

## 9. Apêndices

### 9.1 Constantes Geofísicas

```python
# Constante gravitacional
G = 6.67430e-11  # m³/kg·s²

# Aceleração da gravidade (padrão)
g0 = 9.80665  # m/s²

# Densidade típica da crosta continental
rho_crosta = 2670  # kg/m³ (2.67 g/cm³)

# Densidade da água
rho_agua = 1000  # kg/m³

# Fator de Bouguer (mGal)
bouguer_factor = 0.04191  # (g/cm³)⁻¹·m⁻¹
```

### 9.2 Conversões de Unidades

#### Gravimetria

```python
# mGal ↔ μGal
1 mGal = 1000 μGal
1 μGal = 0.001 mGal

# mGal ↔ m/s²
1 mGal = 1e-5 m/s²
1 m/s² = 1e5 mGal

# Gravity Unit (g.u.)
1 g.u. = 0.1 mGal
```

#### Magnetometria

```python
# nT ↔ γ (gamma)
1 nT = 1 γ

# nT ↔ Tesla
1 nT = 1e-9 T
1 T = 1e9 nT
```

### 9.3 Atalhos de Teclado (Streamlit)

| Atalho | Ação |
|--------|------|
| `Ctrl+R` | Recarregar aplicação |
| `Ctrl+Shift+R` | Limpar cache e recarregar |
| `Ctrl+K` | Focar no campo de busca |
| `Ctrl+Shift+M` | Abrir menu de configurações |

### 9.4 Variáveis de Ambiente

Crie arquivo `.env` na raiz:

```bash
# API Keys
GROQ_API_KEY=gsk_...

# Configurações
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=false

# ChromaDB
CHROMA_DB_PATH=./rag_database/chromadb

# Logs
LOG_LEVEL=INFO
LOG_FILE=geobot.log
```

Carregar no código:
```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
```

### 9.5 Comandos Úteis

#### Verificar Instalação

```powershell
# Python
python --version

# Pip
pip --version

# Streamlit
streamlit --version

# Listar pacotes instalados
pip list

# Verificar dependências
pip check
```

#### Testes Rápidos

```python
# Testar importações
python -c "import polars; import streamlit; import groq; print('OK')"

# Testar Groq API
python -c "from groq import Groq; c = Groq(api_key='gsk_...'); print(c.models.list())"

# Testar ChromaDB
python -c "import chromadb; c = chromadb.Client(); print('OK')"
```

---

## 10. Suporte e Contribuições

### 10.1 Relatar Bugs

**GitHub Issues (se aplicável):**
1. Descreva o problema
2. Inclua logs relevantes
3. Passos para reproduzir
4. Versão do Python e OS

**Formato:**
```markdown
**Descrição:**
Erro ao carregar arquivo CSV com dados magnéticos

**Passos para Reproduzir:**
1. Carregar magnetic_data.csv
2. Erro: "InvalidDataError: Colunas faltando: {'value'}"

**Logs:**
```
2025-02-06 10:30:15 | ERROR | Erro ao parsear arquivo: ...
```

**Ambiente:**
- OS: Windows 11
- Python: 3.11.9
- GeoBot: 1.0.0
```

### 10.2 Solicitar Funcionalidades

Abra discussão com:
- Descrição da funcionalidade
- Caso de uso
- Benefícios esperados
- Referências científicas (se aplicável)

### 10.3 Contribuir com Código

1. **Fork** do repositório (se open source)
2. Criar branch para feature: `git checkout -b feature/nova-funcao`
3. Implementar seguindo padrões deste documento
4. Adicionar testes
5. Documentar no docstring
6. Commit: `git commit -m "feat: adiciona derivada horizontal"`
7. Push: `git push origin feature/nova-funcao`
8. Abrir Pull Request

### 10.4 Licença

**MIT License**

```
Copyright (c) 2025 Allan Ramalho

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 11. Roadmap e Melhorias Futuras

### 11.1 Funcionalidades Planejadas

#### Versão 1.1 (Próxima Release)

- [ ] **Processamentos Adicionais:**
  - Redução ao polo (RTP)
  - Derivadas horizontais
  - Signal analítico
  - Matched filter
  - Separação regional-residual

- [ ] **Visualizações:**
  - Mapas 3D interativos
  - Seções verticais
  - Perfis customizáveis
  - Exportar figuras em alta resolução

- [ ] **RAG Aprimorado:**
  - Suporte a Word/LaTeX
  - Metadados ABNT automáticos
  - Sugestões de leitura

#### Versão 1.2 (Médio Prazo)

- [ ] **Modelagem Direta:**
  - Corpos geométricos (esfera, cilindro, prisma)
  - Cálculo de anomalias sintéticas
  - Comparação modelo vs observado

- [ ] **Inversão:**
  - Inversão de gravidade 2D
  - Estimativa de profundidade
  - Interfaces densidade

- [ ] **Banco de Dados:**
  - SQLite para projetos
  - Histórico de processamentos
  - Exportar relatórios PDF

#### Versão 2.0 (Longo Prazo)

- [ ] **Multi-modal:**
  - Análise de imagens de satélite
  - Integração com dados sísmicos
  - Cross-plot geofísico-geológico

- [ ] **Cloud:**
  - Deploy em AWS/Azure
  - Colaboração multi-usuário
  - API REST pública

- [ ] **IA Avançada:**
  - Fine-tuning de LLM em geofísica
  - Classificação automática de anomalias
  - Sugestão de próximos processamentos

### 11.2 Como Solicitar Features

Envie proposta detalhada incluindo:
1. Motivação científica
2. Casos de uso reais
3. Referências bibliográficas
4. Prioridade sugerida

---

## 12. FAQ - Perguntas Frequentes

### Q1: Posso usar offline?

**R:** Parcialmente. O processamento geofísico e visualizações funcionam offline. O LLM (conversação) requer internet para acessar Groq API.

### Q2: Quantos dados posso processar?

**R:** Testado até 1M de pontos. Para datasets maiores:
- Use Polars para carregamento
- Considere downsampling
- Processe em lotes

### Q3: O GeoBot substitui software comercial?

**R:** Não. É complementar. Use para:
- Prototipagem rápida
- Análises exploratórias
- Ensino e pesquisa

Para produção, valide com software comercial (Geosoft, Intrepid, etc.).

### Q4: Como adicionar suporte a outros idiomas?

**R:** O LLM já suporta PT/EN/ES. Para UI:
1. Use `langdetect` para auto-detecção
2. Crie dicts de tradução
3. Parametrize strings na interface

### Q5: Posso vender análises feitas com GeoBot?

**R:** Sim, licença MIT permite uso comercial. Mantenha créditos aos autores.

### Q6: Como citar o GeoBot em publicações?

**R:**
```
RAMALHO, A. GeoBot: Agente de IA para Processamento de Dados Geofísicos. 
Versão 1.0. 2025. Disponível em: <URL do repositório>. 
Acesso em: DD MMM. YYYY.
```

### Q7: Funciona com Python 3.12?

**R:** Não testado. Python 3.11.9 é obrigatório devido a dependências específicas. Futura compatibilidade será avaliada.

---

## 13. Glossário

**API Key:** Chave de autenticação para acessar serviços externos (Groq).

**Bouguer Correction:** Correção gravimétrica que remove efeito de placa infinita equivalente à topografia.

**ChromaDB:** Banco de dados vetorial open-source para embeddings.

**Embedding:** Representação vetorial de texto em espaço de alta dimensionalidade.

**Fallback:** Mecanismo de troca automática para alternativa quando principal falha.

**FFT (Fast Fourier Transform):** Algoritmo eficiente para transformada de Fourier.

**Gridding:** Interpolação de dados irregulares para grade regular.

**LLM (Large Language Model):** Modelo de linguagem de grande escala (ex: LLaMA).

**mGal (miligal):** Unidade de aceleração gravitacional (10⁻⁵ m/s²).

**nT (nanotesla):** Unidade de campo magnético (10⁻⁹ Tesla).

**Polars:** Biblioteca Python de DataFrames com alta performance.

**RAG (Retrieval-Augmented Generation):** Técnica que enriquece LLM com documentos recuperados.

**Rate Limit:** Limite de requisições por período imposto por API.

**Streamlit:** Framework Python para criar aplicações web interativas.

**Upward Continuation:** Continuação de campo potencial para plano acima do observado.

**Vector Store:** Banco especializado em busca por similaridade de vetores.

---

## 14. Contato e Créditos

### Autor Principal

**Allan Ramalho**  
Mestrando em Dinâmica dos Oceanos e da Terra  
Universidade Federal Fluminense (UFF)  
📧 Email: [contato@exemplo.com]  
🔗 LinkedIn: [linkedin.com/in/allanramalho]  
🐙 GitHub: [github.com/allanramalho]

### Orientador

**Prof. Dr. Rodrigo Bijani**  
Departamento de Geofísica  
Universidade Federal Fluminense (UFF)

### Agradecimentos

- Fatiando a Terra (Harmonica library)
- Groq AI (API gratuita)
- Streamlit Team
- Comunidade Python Geofísica

---

## 15. Changelog

### v1.0.0 (Fevereiro 2025) - Release Inicial

**Features:**
- ✅ Aplicação monolítica completa (geobot.py)
- ✅ Interface Streamlit com 3 páginas
- ✅ LLM via Groq API com fallback
- ✅ Sistema RAG com ChromaDB
- ✅ 2 processamentos geofísicos (Bouguer, upward continuation)
- ✅ Registro automático de funções
- ✅ Suporte a CSV, TXT, Excel
- ✅ Visualizações Plotly 2D
- ✅ Dados de exemplo sintéticos (3 datasets)
- ✅ Documentação completa

**Dependências:**
- Python 3.11.9
- 80+ pacotes Python
- Groq API (externa)

**Limitações Conhecidas:**
- Apenas 2 processamentos implementados
- RAG vazio por padrão (usuário adiciona PDFs)
- Sem testes unitários automatizados
- Sem deploy em cloud

---

**📌 Versão deste documento:** 1.0.0  
**📅 Última atualização:** Fevereiro 2025  
**✍️ Mantenedor:** Allan Ramalho

---

*Fim da documentação. Para dúvidas, consulte os logs, FAQs ou entre em contato.*
