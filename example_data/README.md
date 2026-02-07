# Datasets de Exemplo - GeoBot

Este diretório contém datasets sintéticos para demonstração e testes do GeoBot.

## Arquivos Disponíveis

### 1. gravity_basin_example.csv
**Tipo:** Gravimetria 2D  
**Cenário:** Bacia sedimentar elíptica  
**Dimensões:** 50 km × 50 km  
**Pontos:** 10.000  

**Colunas:**
- `longitude`: Coordenada X (m)
- `latitude`: Coordenada Y (m)
- `elevation`: Elevação (m)
- `gravity_mgal`: Anomalia gravimétrica total (mGal)
- `regional_mgal`: Componente regional (mGal)

**Parâmetros do Modelo:**
- Profundidade máxima: 3000 m
- Contraste de densidade: -400 kg/m³
- Ruído: ±0.5 mGal

**Processamentos Sugeridos:**
- Correção de Bouguer (densidade 2200 kg/m³)
- Separação regional-residual
- Continuação ascendente
- Derivadas direcionais

---

### 2. magnetic_dike_example.csv
**Tipo:** Magnetometria 2D  
**Cenário:** Dique básico vertical (N45E)  
**Dimensões:** 30 km × 30 km  
**Pontos:** 6.400  

**Colunas:**
- `longitude`: Coordenada X (m)
- `latitude`: Coordenada Y (m)
- `elevation`: Elevação (m)
- `tmi_nt`: Campo magnético total (nT)
- `igrf_nt`: Campo regional IGRF (nT)
- `anomaly_nt`: Anomalia magnética (nT)

**Parâmetros do Modelo:**
- Largura: 500 m
- Profundidade topo: 100 m
- Profundidade base: 2000 m
- Susceptibilidade: 0.05 SI
- Ruído: ±5 nT

**Processamentos Sugeridos:**
- Redução ao polo
- Amplitude do sinal analítico
- Derivada Tilt
- Matched filter

---

### 3. gravity_profile_sphere.csv
**Tipo:** Gravimetria 1D (perfil)  
**Cenário:** Esfera enterrada  
**Extensão:** 10 km  
**Pontos:** 100  

**Colunas:**
- `x`: Distância ao longo do perfil (m)
- `gravity_mgal`: Anomalia gravimétrica (mGal)
- `elevation`: Elevação constante (300 m)

**Parâmetros do Modelo:**
- Profundidade do centro: 1000 m
- Raio: 500 m
- Contraste de densidade: 300 kg/m³
- Ruído: ±0.3 mGal

**Processamentos Sugeridos:**
- Inversão de profundidade (half-width)
- Deconvolução de Euler
- Modelagem forward

---

## Como Usar

1. Abra o GeoBot
2. Carregue um dos arquivos CSV na sidebar
3. Converse com o bot sobre os dados
4. Solicite processamentos específicos

## Exemplos de Comandos

```
"Carregue o arquivo gravity_basin_example.csv e mostre um mapa da anomalia"

"Aplique correção de Bouguer com densidade 2200 kg/m³"

"Faça continuação ascendente para 1000 metros e compare com o original"

"Calcule a primeira derivada vertical e identifique bordas da bacia"
```

## Dados Reais

Para usar seus próprios dados, garanta que o arquivo CSV contenha:

**Mínimo requerido:**
- Colunas de coordenadas (x, y ou lon, lat)
- Coluna com valores do campo potencial
- Nomes de colunas descritivos

**Opcional mas recomendado:**
- Coluna de elevação (para correções)
- Metadados em comentários no cabeçalho
- Informação de unidades

---

*Datasets gerados sinteticamente para fins didáticos.*
*Não representam levantamentos reais.*
