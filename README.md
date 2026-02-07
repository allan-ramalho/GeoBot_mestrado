# 🌍 GeoBot - Agente de IA para Processamento Geofísico

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11.9-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31.1-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1+cu124-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Groq](https://img.shields.io/badge/LLM-Groq_API-7C3AED)](https://groq.com/)

**Assistente conversacional inteligente com aceleração GPU para processar e analisar dados geofísicos de gravimetria e magnetometria**

[🚀 Instalação](#-instalação-rápida) • [📖 Documentação](#-documentação) • [🎯 Recursos](#-recursos) • [⚡ GPU](#-aceleração-gpu) • [🤝 Contribuir](#-como-contribuir)

</div>

---

## ✨ O que é o GeoBot?

GeoBot é um agente de IA que combina **processamento geofísico clássico** com **inteligência artificial generativa** para tornar a análise de dados de métodos potenciais mais acessível e eficiente.

### 🎯 Principais Funcionalidades

| Funcionalidade | Descrição |
|----------------|-----------|
| 💬 **Conversação Natural** | Processe dados simplesmente conversando: *"Aplique correção de Bouguer com densidade 2.67"* |
| 📚 **Citações Automáticas** | Sistema RAG (Retrieval-Augmented Generation) busca e cita papers científicos automaticamente em formato ABNT |
| 🔬 **Processamento Geofísico** | Biblioteca completa: Bouguer, RTP, derivadas, continuação, filtros, sinal analítico, tilt angle |
| 📊 **Visualizações Interativas** | Mapas 2D/3D com Plotly, comparações antes/depois, histogramas, estatísticas |
| 🚀 **Aceleração GPU** | Suporte automático para NVIDIA CUDA e Apple Silicon (M1/M2) |
| 🔌 **Extensível** | Sistema de registro de funções permite adicionar novos processamentos facilmente |

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

A aplicação abrirá automaticamente no navegador em `http://localhost:8501` 🎉
allan-ramalho/GeoBot_mestrado.git
cd GeoBot_mestrado

# 2. Crie ambiente virtual Python 3.11+
python3.11 -m venv venv
source venv/bin/activate

# 3. Instale dependências
pip install -r requirements.txt

# 4. Instale PyTorch com suporte GPU
# Para NVIDIA CUDA 12.4 (Recomendado - 10-50x mais rápido!):
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# Para Apple Silicon (M1/M2):
pip install torch torchvision

# 5. Configure suas chaves de API
cp .env.example .env
nano .env  # ou use seu editor preferido

# 6 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Para Apple Silicon (M1/M2):
pip install torch torchvision

# 5. Execute o GeoBot
streamlit run geobot.py
```

---

## 🎓 Primeiros Passos (Para Iniciantes)

### 1️⃣ Configure sua API Key da Groq

O GeoBot usa a **Groq API** (gratuita!) para conversação com IA:

1. Acesse [console.groq.com/keys](https://console.groq.com/keys)
2. Crie uma conta gratuita
3. Gere uma nova API Key
4. Cole a chave na interface do GeoBot

> 💡 **Dica:** A Groq oferece modelos LLM de última geração gratuitamente!

### 2️⃣ Carregue seus Dados

O GeoBot aceita diversos formatos:

- **CSV/TXT:** Colunas com X, Y (coordenadas) e valor (gravidade/magnetometria)
- **Excel:** Arquivos `.xlsx` ou `.xls`
- **Formatos geofísicos:** Grids regulares

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

| Documento | Descrição |
|-----------|-----------|
| **[USER_GUIDE.md](USER_GUIDE.md)** | 📘 Manual do usuário com exemplos práticos |
| **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)** | 🛠️ Guia para desenvolvedores - como adicionar novas funções |
| **[CONTRIBUTING.md](CONTRIBUTING.md)** | 🤝 Como contribuir para o projeto |
| **[DOCUMENTACAO.md](DOCUMENTACAO.md)** | 📚 Documentação técnica completa (original) |

---

## 🎯 Exemplos de Uso

### Exemplo 1: Correção de Bouguer

```
Você: "Carregue os dados de gravidade"
[Arraste arquivo CSV na sidebar]

Você: "Aplique correção de Bouguer com densidade 2.67 g/cm³"

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
Você: "Faça redução ao polo seguida de derivada vertical"

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
│        RAG Engine (ChromaDB)                │
│  • Embeddings: all-MiniLM-L6-v2             │
│  • Vector store persistente                 │
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
| **Apple Silicon** (M1/M2) | ✅ Automático | ~5-20x mais rápido |
| **CPU** (Fallback) | ✅ Sempre funciona | Performance padrão |

Para verificar se sua GPU está sendo usada, veja o log de inicialização:

```
🚀 GPU NVIDIA detectada: NVIDIA GeForce RTX 3080
```

---

## ⚡ Aceleração GPU

O GeoBot possui suporte **automático** para aceleração GPU via NVIDIA CUDA e Apple Silicon (MPS), proporcionando **10-50x de speedup** em operações FFT!

### 🚀 Performance Comparativa

| Operação | CPU (numpy) | GPU (CUDA) | Speedup |
|----------|-------------|------------|---------|
| **FFT 2D (150×150)** | 120ms | 8ms | **15x** ⚡ |
| **Derivada Vertical** | 250ms | 12ms | **21x** ⚡ |
| **Redução ao Polo** | 450ms | 28ms | **16x** ⚡ |
| **Sinal Analítico** | 380ms | 24ms | **16x** ⚡ |
| **Embeddings (RAG)** | 850ms | 85ms | **10x** ⚡ |
| **Grid Cache** | 2000ms | 2ms | **1000x** 💾 |

### 📦 Instalação GPU

**NVIDIA (Windows/Linux):**
```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

**Apple Silicon (M1/M2/M3):**
```bash
pip install torch torchvision  # MPS é automático no PyTorch 2.x
```

### ✅ Verificação

O GeoBot detecta automaticamente sua GPU ao iniciar:
```
🚀 GPU NVIDIA detectada: NVIDIA GeForce RTX 3050 Ti
✅ Módulo de otimizações GPU ativado
```

Para mais detalhes, veja [OTIMIZACOES_GPU.md](OTIMIZACOES_GPU.md).

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

## 🤝 Como Contribuir

Adoramos contribuições! Veja como você pode ajudar:

1. **🐛 Reportar Bugs:** Abra uma [issue](https://github.com/allan-ramalho/GeoBot_mestrado/issues) detalhando o problema
2. **💡 Sugerir Funcionalidades:** Compartilhe suas ideias nas issues
3. **🔧 Enviar Pull Requests:** Consulte [CONTRIBUTING.md](CONTRIBUTING.md) para o processo
4. **📚 Melhorar Documentação:** Correções e melhorias são sempre bem-vindas
5. **⭐ Dar uma Estrela:** Se o projeto te ajudou, deixe uma estrela no GitHub!

### Adicionando Novos Processamentos

É muito fácil! Veja o guia completo em [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md).

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
