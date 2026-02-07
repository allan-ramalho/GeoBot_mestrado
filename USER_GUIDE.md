# 📘 Manual do Usuário - GeoBot

> **Guia completo para usar o GeoBot mesmo sem conhecimento técnico!** 🌍

---

## 📑 Índice

1. [Início Rápido](#-início-rápido)
2. [Interface do GeoBot](#-interface-do-geobot)
3. [Carregando Dados](#-carregando-dados)
4. [Conversando com o GeoBot](#-conversando-com-o-geobot)
5. [Processamentos Disponíveis](#-processamentos-disponíveis)
6. [Interpretando Resultados](#-interpretando-resultados)
7. [Dicas e Truques](#-dicas-e-truques)
8. [Solução de Problemas](#-solução-de-problemas)

---

## 🚀 Início Rápido

### Instalação em 3 Passos (Windows)

1. **Baixe o GeoBot** do GitHub
2. **Clique duas vezes** em `INSTALAR.bat`
3. **Clique duas vezes** em `INICIAR_GEOBOT.bat`

Pronto! O GeoBot abrirá no seu navegador 🎉

### Primeira Vez Usando

Ao abrir pela primeira vez, você verá a **página de boas-vindas**. Siga os passos:

1. **Configure sua API Key da Groq**
   - Acesse [console.groq.com/keys](https://console.groq.com/keys)
   - Crie conta gratuita
   - Gere uma chave (começa com `gsk_...`)
   - Cole no campo indicado
   
2. **Selecione o modelo LLM**
   - Recomendado: `llama-3.3-70b-versatile`
   - Clique em "Confirmar e Iniciar"

3. **Você está pronto!** ✅

---

## 🎨 Interface do GeoBot

A interface é dividida em 3 áreas principais:

```
┌──────────────────────────────────────────────┐
│ 📁 SIDEBAR (Esquerda)                        │
│  • Upload de dados                           │
│  • Sugestões de comandos                     │
│  • Ajuda rápida                              │
├──────────────────────────────────────────────┤
│ 📊 PAINEL DE DADOS (Centro-Esquerda)        │
│  • Estatísticas descritivas                  │
│  • Preview da tabela                         │
│  • Mapas e gráficos                          │
├──────────────────────────────────────────────┤
│ 💬 CHAT (Centro-Direita)                    │
│  • Conversação com o GeoBot                  │
│  • Resultados de processamento               │
│  • Visualizações interativas                 │
└──────────────────────────────────────────────┘
```

### Sidebar (Barra Lateral)

A sidebar fica **sempre visível** no lado esquerdo. Você pode:

- **📂 Carregar dados:** Arraste ou clique em "Browse files"
- **💡 Ver sugestões:** Expandir "Sugestões de Comandos"
- **ℹ️ Obter ajuda:** Expandir "Ajuda"

### Painel de Dados

Após carregar um arquivo, você verá:

- **📊 Estatísticas:**
  - Número de pontos
  - Média, mediana, desvio padrão
  - Mínimo e máximo
  
- **📋 Preview dos Dados:**
  - Tabela com as primeiras 10 linhas
  
- **🗺️ Visualizações:**
  - Scatter plot colorido
  - Mapa interativo com os pontos

### Chat

É aqui que a mágica acontece! Converse com o GeoBot:

- Digite comandos em **linguagem natural**
- Veja respostas instantâneas
- Visualize gráficos de processamento
- Receba citações científicas automáticas

---

## 📂 Carregando Dados

### Formatos Aceitos

O GeoBot aceita diversos formatos:

| Formato | Extensão | Exemplo |
|---------|----------|---------|
| CSV | `.csv` | `dados_gravidade.csv` |
| TXT | `.txt` | `survey_magnetico.txt` |
| Excel | `.xlsx`, `.xls` | `dados_campo.xlsx` |

### Estrutura dos Dados

Seus dados devem ter **pelo menos 3 colunas**:

1. **Coordenada X** (longitude, x, easting)
2. **Coordenada Y** (latitude, y, northing)
3. **Valor medido** (gravity, bouguer, magnetic, tmi)

**Exemplo de CSV válido:**

```csv
longitude,latitude,gravity
-43.2156,-22.9035,980.234
-43.2145,-22.9028,980.189
-43.2167,-22.9041,980.301
...
```

**Exemplo de TXT válido:**

```
x        y        magnetic
500123   7801234  48523.45
500145   7801256  48519.12
500167   7801278  48534.78
...
```

### Passo a Passo: Carregando Arquivo

1. **Clique em "Browse files"** na sidebar
2. **Selecione seu arquivo** (CSV, TXT ou Excel)
3. **Aguarde o processamento** (barra de progresso aparece)
4. **✅ Sucesso!** Estatísticas e mapa aparecem automaticamente

> 💡 **Dica:** O GeoBot detecta automaticamente:
> - Tipo de dado (gravimetria ou magnetometria)
> - Nomes das colunas
> - Sistema de coordenadas
> - Unidades de medida

---

## 💬 Conversando com o GeoBot

### Comandos Básicos

O GeoBot entende linguagem natural! Experimente:

#### Estatísticas e Análise

```
"Mostre as estatísticas dos dados"
"Calcule a média da gravidade"
"Quais são os outliers?"
"Faça um histograma"
"Mostre a distribuição dos valores"
```

#### Processamento Geofísico

```
"Aplique correção de Bouguer"
"Calcule a derivada vertical"
"Faça redução ao polo"
"Aplique continuação ascendente de 1000 metros"
"Calcule o sinal analítico"
"Aplique filtro passa-baixa com comprimento de onda 5000"
```

#### Perguntas Educacionais

```
"O que é correção de Bouguer?"
"Qual a diferença entre RTP e sinal analítico?"
"Para que serve a derivada vertical?"
"Como funciona a continuação ascendente?"
```

### Exemplos de Conversação

#### Exemplo 1: Análise Exploratória

```
Você: "Olá! Carreguei dados de gravidade da Baía de Guanabara"

GeoBot: "Olá! Vi que você carregou 10.000 pontos de gravimetria.
Os valores variam de 980.2 a 985.7 mGal. O que gostaria de fazer?"

Você: "Mostre as estatísticas descritivas"

GeoBot: "📊 Estatísticas:
- Média: 982.45 mGal
- Mediana: 982.38 mGal
- Desvio padrão: 1.23 mGal
- IQR: 1.67 mGal

Os dados parecem bem distribuídos sem outliers significativos."

Você: "Faça um histograma"

GeoBot: "✅ Histograma gerado!
[Exibe gráfico interativo]
A distribuição é aproximadamente normal, indicando dados de
qualidade."
```

#### Exemplo 2: Processamento Completo

```
Você: "Preciso fazer correção de Bouguer"

GeoBot: "Para aplicar correção de Bouguer, preciso da densidade.
Qual densidade deseja usar? (padrão: 2.67 g/cm³)"

Você: "Use a densidade padrão"

GeoBot: "✅ Correção de Bouguer aplicada!

**Parâmetros:**
- Densidade: 2670 kg/m³

**Resultado:**
- Original: 980.2 a 985.7 mGal
- Anomalia Bouguer: -15.3 a +8.9 mGal
- Tempo: 0.34s

[Exibe mapas comparativos]

📖 **Referência:**
BLAKELY, R. J. **Potential Theory in Gravity and Magnetic 
Applications**. Cambridge University Press, 1995."

Você: "Agora calcule a derivada vertical"

GeoBot: "✅ Derivada vertical calculada!

A derivada vertical realça anomalias rasas...
[Exibe resultados]"
```

---

## 🔬 Processamentos Disponíveis

### Gravimetria

#### Correção de Bouguer
**O que faz:** Remove o efeito da topografia e da Lua/Sol

**Comando:**
```
"Aplique correção de Bouguer"
"Bouguer com densidade 2.67"
```

**Resultado:** Anomalia Bouguer (mGal)

---

#### Anomalia Ar-Livre
**O que faz:** Corrige apenas pela elevação

**Comando:**
```
"Calcule anomalia ar-livre"
"Free-air"
```

---

### Magnetometria

#### Redução ao Polo (RTP)
**O que faz:** Transforma campo magnético para latitude magnética 90°

**Comando:**
```
"Faça redução ao polo"
"Aplique RTP"
```

**Quando usar:** Facilita interpretação em baixas latitudes magnéticas

---

#### Sinal Analítico
**O que faz:** Calcula amplitude independente da magnetização

**Comando:**
```
"Calcule sinal analítico"
"Aplique ASA"
```

**Quando usar:** Delinear corpos sem saber direção de magnetização

---

#### Ângulo de Tilt
**O que faz:** Normaliza gradientes para delinear bordas

**Comando:**
```
"Calcule ângulo de tilt"
"Aplique tilt angle"
```

**Quando usar:** Encontrar bordas precisas de corpos

---

### Processamentos Gerais

#### Continuação Ascendente
**O que faz:** Simula medição em altitude maior

**Comando:**
```
"Continuação ascendente de 1000 metros"
"Upward continuation 500m"
```

**Quando usar:** Realçar fontes profundas, remover ruído

---

#### Derivada Vertical
**O que faz:** Calcula taxa de variação vertical do campo

**Comando:**
```
"Calcule derivada vertical"
"Primeira derivada"
```

**Quando usar:** Realçar anomalias rasas, bordas

---

#### Derivada Horizontal Total (THD)
**O que faz:** Magnitude do gradiente horizontal

**Comando:**
```
"Calcule THD"
"Derivada horizontal total"
```

**Quando usar:** Encontrar bordas horizontais

---

#### Filtros Gaussianos
**O que faz:** Suaviza (passa-baixa) ou realça (passa-alta)

**Comando:**
```
"Aplique filtro passa-baixa com lambda 5000"
"Filtro gaussiano sigma 2"
```

**Quando usar:** Remoção de ruído, separação regional-residual

---

## 📊 Interpretando Resultados

### Mapas de Calor

O GeoBot gera automaticamente **mapas comparativos**:

- **Esquerda:** Dados originais
- **Centro:** Dados processados
- **Direita:** Diferença

**Como interpretar as cores:**

| Cor | Significado |
|-----|-------------|
| 🔴 Vermelho | Valores altos (positivos) |
| 🔵 Azul | Valores baixos (negativos) |
| ⚪ Branco | Valores neutros |

---

### Histogramas

Mostram a **distribuição de valores**:

- **Gaussiana:** Dados bem distribuídos ✅
- **Bimodal:** Duas populações distintas
- **Assimétrica:** Possível tendência ou outliers

---

### Estatísticas

**Mean (Média):** Valor central dos dados  
**Median (Mediana):** Valor que divide ao meio  
**Std (Desvio Padrão):** Quão dispersos são os dados  
**IQR:** Intervalo interquartil (50% centrais)

> 💡 **Dica:** Se `std` é muito alto, pode haver outliers ou ruído

---

## 💡 Dicas e Truques

### 1. Use Nomes Descritivos

```
❌ "Processe os dados"
✅ "Aplique correção de Bouguer com densidade 2.67"
```

### 2. Peça Explicações

```
"Explique o que é sinal analítico"
"Por que usar derivada vertical?"
"Qual a diferença entre RTP e continuação?"
```

### 3. Pipeline de Processamento

Combine múltiplos processamentos:

```
"Faça RTP seguido de derivada vertical"
"Aplique Bouguer e depois filtro passa-baixa"
```

### 4. Salve Resultados

Após o processamento, **anote os parâmetros usados**:

```
✅ Correção de Bouguer aplicada
   Densidade: 2670 kg/m³
   Tempo: 0.34s
```

### 5. Explore os Dados Primeiro

Antes de processar:

1. Veja estatísticas
2. Analise histograma
3. Identifique outliers
4. Escolha processamento adequado

---

## 🚨 Solução de Problemas

### Erro: "Arquivo não reconhecido"

**Causa:** Formato inválido ou colunas faltando

**Solução:**
- Verifique se há colunas X, Y e valor
- Confirme delimitador (`,` ou `;` ou tab)
- Remova linhas vazias no início do arquivo

---

### Erro: "GPU não detectada"

**Causa:** PyTorch não instalado ou GPU incompatível

**Solução:**
- Instale PyTorch: `pip install torch`
- O GeoBot funciona normalmente em CPU (mais lento)

---

### Chat não responde

**Causa:** API Key inválida ou rate limit

**Solução:**
- Verifique se a chave está correta
- Aguarde 1 minuto (rate limit da Groq)
- O sistema tentará outros modelos automaticamente

---

### Mapas não aparecem

**Causa:** Coordenadas fora do range geográfico

**Solução:**
- Verifique se X e Y estão em graus (-180 a 180)
- Para coordenadas UTM, converta para lat/lon primeiro

---

### Processamento muito lento

**Causa:** Muitos pontos ou CPU sem GPU

**Solução:**
- Reduza resolução do grid
- Instale PyTorch para aceleração GPU
- Use filtros para reduzir ruído antes

---

## 📚 Recursos Adicionais

### Tutoriais em Vídeo

(Em breve - contribua!)

### Exemplos de Dados

O GeoBot inclui 3 datasets de exemplo em `example_data/`:

1. `gravity_basin_example.csv` - Bacia sedimentar
2. `gravity_profile_sphere.csv` - Anomalia esférica
3. `magnetic_dike_example.csv` - Dique magnético

---

## 📞 Precisa de Ajuda?

- 📧 Email: allansoares@id.uff.br
- 💬 Issues: [GitHub Issues](https://github.com/seu-usuario/GeoBot/issues)
- 📖 Documentação: [DOCUMENTACAO.md](DOCUMENTACAO.md)

---

<div align="center">

**Divirta-se explorando seus dados geofísicos! 🌍🚀**

Made with ❤️ by PPG DOT-UFF

</div>
