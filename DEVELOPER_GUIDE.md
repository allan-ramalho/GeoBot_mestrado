# 🛠️ Guia do Desenvolvedor - GeoBot

> **"Como expandir o GeoBot de forma fácil e divertida!"** 🚀

---

## 📚 Índice

1. [Visão Geral](#-visão-geral)
2. [Estrutura do Projeto](#-estrutura-do-projeto)
3. [Adicionando Novas Funções de Processamento](#-adicionando-novas-funções-de-processamento)
4. [Sistema de Registro](#-sistema-de-registro)
5. [Exemplos Práticos](#-exemplos-práticos)
6. [Boas Práticas](#-boas-práticas)
7. [Debugging e Testes](#-debugging-e-testes)

---

## 🎯 Visão Geral

O GeoBot foi projetado para ser **facilmente extensível**. Você pode adicionar novas funções de processamento geofísico em apenas **3 passos**:

1. ✍️ Escreva a função
2. 🎨 Decore com `@register_processing`
3. ✅ Pronto! A função já aparece no sistema

**Não precisa mexer em:**
- ❌ Interface (Streamlit)
- ❌ Sistema de chat
- ❌ LLM Manager
- ❌ RAG Engine

Tudo é **automático**! 🎉

---

## 📁 Estrutura do Projeto

```
GeoBot/
│
├── geobot.py               ⭐ ARQUIVO PRINCIPAL (4000+ linhas)
│   ├── [Linhas 1-100]     📦 Imports e configurações
│   ├── [Linhas 101-250]   🚀 Configuração GPU
│   ├── [Linhas 251-500]   🎯 Sistema de registro
│   ├── [Linhas 501-900]   📊 Classes de dados
│   ├── [Linhas 901-2500]  🔬 FUNÇÕES DE PROCESSAMENTO ⬅️ AQUI!
│   ├── [Linhas 2501-3000] 📈 Visualizações
│   └── [Linhas 3001-4000] 🎨 Interface Streamlit
│
├── requirements.txt        📋 Dependências Python
├── INSTALAR.bat           🪟 Instalador Windows
├── INICIAR_GEOBOT.bat     ▶️ Launcher Windows
│
├── assets/                🖼️ Logos e imagens
├── example_data/          📂 Dados de exemplo
├── rag_database/          🧠 Base de conhecimento (ChromaDB)
│
├── README.md              📖 Documentação principal
├── DEVELOPER_GUIDE.md     🛠️ Este arquivo!
├── USER_GUIDE.md          👤 Manual do usuário


```

---

## 🚀 Adicionando Novas Funções de Processamento

### 📝 Template Básico

Copie e cole este template no arquivo `geobot.py` **na seção de processamentos** (por volta da linha 1500-2500):

```python
@register_processing(
    category="Minha Categoria",
    description="Descrição curta do que a função faz",
    input_type="grid",  # 'grid', 'profile' ou 'points'
    requires_params=['param1', 'param2']  # Parâmetros obrigatórios
)
def minha_funcao(
    data: GeophysicalData,
    param1: float,
    param2: str = "valor_padrao"
) -> ProcessingResult:
    """
    Descrição detalhada da função.
    
    Esta função implementa [CONCEITO GEOFÍSICO] usando [MÉTODO].
    
    Fundamento Teórico:
    -------------------
    [Explique a teoria por trás do processamento]
    
    Aplicações:
    -----------
    - Aplicação 1
    - Aplicação 2
    
    Referências:
    ------------
    AUTOR, A. **Título do Paper**. Journal, v. XX, p. YY-ZZ, 2020.
    
    Parameters:
    -----------
    data : GeophysicalData
        Dados de entrada (gravimetria ou magnetometria)
    param1 : float
        Descrição do primeiro parâmetro
    param2 : str, optional
        Descrição do segundo parâmetro (padrão: "valor_padrao")
    
    Returns:
    --------
    ProcessingResult
        Objeto contendo:
        - processed_data: Dados processados
        - original_data: Dados originais
        - figures: Lista de gráficos
        - explanation: Explicação em Markdown
        - execution_time: Tempo de execução
    
    Raises:
    -------
    ProcessingError
        Se algo der errado no processamento
    
    Examples:
    ---------
    >>> result = minha_funcao(data, param1=10.5)
    >>> print(result.execution_time)
    0.342
    """
    start_time = datetime.now()
    
    try:
        # ============================================
        # PASSO 1: Validações
        # ============================================
        if data.dimension not in ['2D', '3D']:
            raise ProcessingError("Função requer dados 2D ou 3D")
        
        if param1 <= 0:
            raise ProcessingError("param1 deve ser positivo")
        
        # ============================================
        # PASSO 2: Conversão para grid (se necessário)
        # ============================================
        Xi, Yi, Zi = data.to_grid(method='linear')
        
        # ============================================
        # PASSO 3: Processamento (SEU CÓDIGO AQUI!)
        # ============================================
        
        # Exemplo: Multiplica valores por param1
        Zi_processed = Zi * param1
        
        # Se precisar de FFT:
        # F = fft2(Zi)
        # ... operações no domínio da frequência ...
        # Zi_processed = np.real(ifft2(F_modified))
        
        # Se precisar de GPU:
        # if GPU_INFO['available']:
        #     import torch
        #     Zi_tensor = torch.from_numpy(Zi).to(GPU_INFO['device'])
        #     ... operações em GPU ...
        #     Zi_processed = Zi_tensor.cpu().numpy()
        
        # ============================================
        # PASSO 4: Criar novo objeto GeophysicalData
        # ============================================
        x_flat = Xi.flatten()
        y_flat = Yi.flatten()
        z_flat = Zi_processed.flatten()
        
        processed_df = pl.DataFrame({
            data.coords['x']: x_flat,
            data.coords['y']: y_flat,
            f"{data.value_column}_processed": z_flat
        })
        
        processed_data = GeophysicalData(
            data=processed_df,
            data_type=data.data_type,
            dimension=data.dimension,
            coords=data.coords,
            value_column=f"{data.value_column}_processed",
            units=data.units,
            crs=data.crs,
            metadata={
                **data.metadata,
                'processing': 'minha_funcao',
                'param1': param1,
                'param2': param2
            }
        )
        
        # ============================================
        # PASSO 5: Criar visualizações
        # ============================================
        figures = create_comparison_plots(
            data, 
            processed_data, 
            f"Minha Função (param1={param1})"
        )
        
        # ============================================
        # PASSO 6: Explicação em Markdown
        # ============================================
        explanation = f"""
### 📊 Minha Função Aplicada!

**Parâmetros:**
- param1: {param1}
- param2: {param2}

**Resultado:**
- Original: {Zi.min():.2f} a {Zi.max():.2f} {data.units}
- Processado: {Zi_processed.min():.2f} a {Zi_processed.max():.2f} {data.units}
- Mudança: {((Zi_processed.mean() - Zi.mean()) / Zi.mean() * 100):.1f}%

**Interpretação:**
[Explique o que o resultado significa geologicamente/geofisicamente]
"""
        
        # ============================================
        # PASSO 7: Referências bibliográficas
        # ============================================
        references = [
            "AUTOR, A. **Título do Paper**. Journal, v. XX, p. YY-ZZ, 2020.",
            "AUTOR, B. **Outro Paper Relevante**. Journal, v. XX, 2019."
        ]
        
        # ============================================
        # PASSO 8: Retornar ProcessingResult
        # ============================================
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return ProcessingResult(
            processed_data=processed_data,
            original_data=data,
            method_name="minha_funcao",
            parameters={'param1': param1, 'param2': param2},
            figures=figures,
            explanation=explanation,
            execution_time=execution_time,
            references=references
        )
        
    except Exception as e:
        logger.error(f"Erro em minha_funcao: {str(e)}")
        raise ProcessingError(f"Falha no processamento: {str(e)}")
```

---

## 🎨 Sistema de Registro

### O Decorator `@register_processing`

Este decorator mágico faz 3 coisas automaticamente:

1. **Registra** a função no dicionário `PROCESSING_REGISTRY`
2. **Valida** os parâmetros de entrada
3. **Torna visível** na interface (sidebar, chat, help)

#### Parâmetros do Decorator

```python
@register_processing(
    category="Categoria",      # Agrupa funções similares
    description="Descrição",   # Aparece na UI
    input_type="grid",         # Valida tipo de entrada
    requires_params=[...]      # Lista de parâmetros obrigatórios
)
```

**Categorias disponíveis:**
- `"Gravimetria"` - Processamentos específicos de gravidade
- `"Magnetometria"` - Processamentos de magnetometria
- `"Geral"` - Aplicável a ambos os métodos

**Tipos de entrada:**
- `"grid"` - Requer dados em malha regular (2D/3D)
- `"profile"` - Aceita perfis 1D
- `"points"` - Aceita pontos irregulares

---

## 💡 Exemplos Práticos

### Exemplo 1: Filtro Gaussiano Customizado

```python
@register_processing(
    category="Geral",
    description="Filtro Gaussiano com sigma ajustável",
    input_type="grid",
    requires_params=['sigma']
)
def filtro_gaussiano_custom(
    data: GeophysicalData, 
    sigma: float = 2.0
) -> ProcessingResult:
    """
    Aplica filtro Gaussiano 2D para suavização.
    
    Parameters:
    -----------
    data : GeophysicalData
        Dados em grid
    sigma : float
        Desvio padrão do kernel (quanto maior, mais suave)
    
    Returns:
    --------
    ProcessingResult
    """
    start_time = datetime.now()
    
    Xi, Yi, Zi = data.to_grid()
    
    # Aplicar filtro Gaussiano
    from scipy.ndimage import gaussian_filter
    Zi_smooth = gaussian_filter(Zi, sigma=sigma)
    
    # ... resto do código (criar GeophysicalData, plots, etc.)
    
    return ProcessingResult(...)
```

### Exemplo 2: Detector de Anomalias

```python
@register_processing(
    category="Geral",
    description="Detecta anomalias usando Z-score",
    input_type="points",
    requires_params=['threshold']
)
def detectar_anomalias(
    data: GeophysicalData,
    threshold: float = 3.0
) -> ProcessingResult:
    """
    Identifica outliers usando método Z-score.
    
    Parameters:
    -----------
    data : GeophysicalData
        Dados brutos (pontos)
    threshold : float
        Limiar Z-score (padrão: 3 = 99.7% confiança)
    
    Returns:
    --------
    ProcessingResult
        Dados marcados com coluna 'is_anomaly'
    """
    start_time = datetime.now()
    
    values = data.data[data.value_column].to_numpy()
    
    # Calcula Z-score
    mean = np.mean(values)
    std = np.std(values)
    z_scores = np.abs((values - mean) / std)
    
    # Marca anomalias
    is_anomaly = z_scores > threshold
    
    # Adiciona coluna ao DataFrame
    data_with_anomalies = data.data.with_columns([
        pl.Series("z_score", z_scores),
        pl.Series("is_anomaly", is_anomaly)
    ])
    
    # ... criar visualização destacando anomalias ...
    
    return ProcessingResult(...)
```

### Exemplo 3: Interpolação Customizada

```python
@register_processing(
    category="Geral",
    description="Interpolação usando diferentes métodos",
    input_type="points",
    requires_params=['method', 'resolution']
)
def interpolar_custom(
    data: GeophysicalData,
    method: str = "cubic",
    resolution: int = 100
) -> ProcessingResult:
    """
    Interpola pontos irregulares para grid regular.
    
    Parameters:
    -----------
    data : GeophysicalData
        Pontos irregulares
    method : str
        Método: 'linear', 'cubic', 'rbf'
    resolution : int
        Número de células na grade
    
    Returns:
    --------
    ProcessingResult
        Grid interpolado
    """
    from scipy.interpolate import griddata, RBFInterpolator
    
    start_time = datetime.now()
    
    x = data.data[data.coords['x']].to_numpy()
    y = data.data[data.coords['y']].to_numpy()
    z = data.data[data.value_column].to_numpy()
    
    # Criar grid regular
    xi = np.linspace(x.min(), x.max(), resolution)
    yi = np.linspace(y.min(), y.max(), resolution)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolar
    if method in ['linear', 'cubic']:
        Zi = griddata((x, y), z, (Xi, Yi), method=method)
    elif method == 'rbf':
        rbf = RBFInterpolator(np.column_stack([x, y]), z)
        Zi = rbf(np.column_stack([Xi.flatten(), Yi.flatten()])).reshape(Xi.shape)
    
    # ... criar GeophysicalData e visualizações ...
    
    return ProcessingResult(...)
```

---

## ✅ Boas Práticas

### 1. Sempre Valide Entradas

```python
# ❌ RUIM (sem validação)
def minha_funcao(data, param):
    result = data.value / param
    
# ✅ BOM (com validação)
def minha_funcao(data, param):
    if param == 0:
        raise ProcessingError("param não pode ser zero!")
    if data.dimension not in ['2D', '3D']:
        raise ProcessingError("Requer dados 2D ou 3D")
    result = data.value / param
```

### 2. Use Try-Except

```python
try:
    # Código que pode falhar
    result = processar_dados(data)
except Exception as e:
    logger.error(f"Erro: {str(e)}")
    raise ProcessingError(f"Falha: {str(e)}")
```

### 3. Documente com Referências

```python
"""
Implementa redução ao polo.

Referências:
------------
BLAKELY, R. J. **Potential Theory in Gravity and Magnetic Applications**. 
Cambridge University Press, 1995. DOI: 10.1017/CBO9780511549816
"""
```

### 4. Aproveite GPU Quando Possível

```python
if GPU_INFO['available']:
    import torch
    # Converter para tensor
    tensor = torch.from_numpy(array).to(GPU_INFO['device'])
    # Processar em GPU
    result_tensor = processar_gpu(tensor)
    # Voltar para NumPy
    result = result_tensor.cpu().numpy()
else:
    # Fallback CPU
    result = processar_cpu(array)
```

### 5. Crie Visualizações Informativas

```python
# Use a função auxiliar create_comparison_plots
figures = create_comparison_plots(original, processed, "Meu Processamento")

# Ou crie gráficos customizados
fig = go.Figure()
fig.add_trace(go.Heatmap(z=data_processed))
fig.update_layout(title="Resultado")
figures.append(fig)
```

---

## 🐛 Debugging e Testes

### Logs

Use `logger` para debugging:

```python
logger.info("Iniciando processamento...")
logger.debug(f"Shape dos dados: {data.shape}")
logger.warning("Valores negativos detectados")
logger.error("Falha crítica!")
```

### Teste sua Função

```python
# No final do arquivo geobot.py, adicione:
if __name__ == "__main__":
    # Criar dados de teste
    test_data = GeophysicalData(...)
    
    # Testar função
    result = minha_funcao(test_data, param1=10)
    
    print(f"✅ Sucesso! Tempo: {result.execution_time}s")
```

### Verificar Registro

```python
# Verificar se a função foi registrada
print(PROCESSING_REGISTRY)

# Deve aparecer:
# {
#   'minha_funcao': {
#     'category': 'Minha Categoria',
#     'description': '...',
#     ...
#   }
# }
```

---

## 🎓 Conceitos Avançados

### 1. Processamento em Pipeline

Combine múltiplas funções:

```python
# Usuário pode pedir:
# "Aplique RTP seguido de derivada vertical"

# Sistema executa:
result1 = reduction_to_pole(data, ...)
result2 = vertical_derivative(result1.processed_data)
```

### 2. Parâmetros Dinâmicos

Extraia parâmetros do comando do usuário:

```python
# "Aplique filtro com sigma 3.5"
# → detect_processing_command() extrai sigma=3.5

def detect_processing_command(user_input):
    if 'filtro' in user_input:
        sigma_match = re.search(r'sigma\s*(\d+\.?\d*)', user_input)
        if sigma_match:
            params['sigma'] = float(sigma_match.group(1))
```

### 3. Processamento Adaptativo

Ajuste parâmetros automaticamente:

```python
def processar_adaptativo(data):
    # Escolhe parâmetros baseado nos dados
    if data.metadata['std'] > 10:
        # Dados ruidosos → mais suavização
        sigma = 5.0
    else:
        # Dados limpos → menos suavização
        sigma = 1.0
    
    return filtro_gaussiano(data, sigma)
```

---

## 📚 Recursos Adicionais

### Bibliotecas Úteis

- **NumPy:** Arrays e operações matriciais
- **SciPy:** FFT, interpolação, filtros
- **Harmonica:** Processamento geofísico específico
- **Plotly:** Visualizações interativas
- **PyTorch:** Aceleração GPU

### Referências Científicas

- **Blakely (1995):** Teoria de campos potenciais
- **Telford et al. (1990):** Geofísica aplicada
- **Fatiando a Terra:** [fatiando.org](https://www.fatiando.org/)

### Comunidade

- Issues no GitHub
- Discussões no README
- Email: allansoares@id.uff.br

---

## 🎉 Parabéns!

Você agora sabe como expandir o GeoBot! 

**Próximos passos:**

1. ✍️ Escreva sua primeira função
2. 🧪 Teste localmente
3. 📤 Faça um Pull Request
4. 🌟 Ajude a comunidade!

**Dúvidas?** Consulte [CONTRIBUTING.md](CONTRIBUTING.md) ou abra uma issue!

---

<div align="center">

**Happy Coding! 🚀🐍**

Made with ❤️ by PPG DOT-UFF

</div>
