"""
Script para gerar datasets sintéticos de exemplo para o GeoBot.

Gera dados realistas de:
1. Gravimetria - Anomalia Bouguer de uma bacia sedimentar
2. Magnetometria - Campo magnético anômalo de um dique
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Diretório de saída
output_dir = Path("example_data")
output_dir.mkdir(exist_ok=True)

# ============================================================================
# DATASET 1: GRAVIMETRIA - BACIA SEDIMENTAR
# ============================================================================

print("Gerando dataset de gravimetria...")

# Define grid
x = np.linspace(0, 50000, 100)  # 50 km, 100 pontos
y = np.linspace(0, 50000, 100)
X, Y = np.meshgrid(x, y)

# Parâmetros da bacia
x_center = 25000
y_center = 25000
a = 15000  # semi-eixo maior (m)
b = 10000  # semi-eixo menor (m)
depth_max = 3000  # profundidade máxima (m)
density_contrast = -400  # kg/m³ (sedimentos menos densos)

# Modelo da bacia (elíptica)
dist = np.sqrt(((X - x_center)/a)**2 + ((Y - y_center)/b)**2)
depth = np.where(dist < 1, depth_max * (1 - dist**2), 0)

# Calcula anomalia gravimétrica (aproximação de placa)
G = 6.674e-11
bouguer_anomaly = 2 * np.pi * G * density_contrast * depth * 1e5  # Converte para mGal

# Adiciona tendência regional
regional = 0.02 * X + 0.01 * Y - 1000

# Anomalia total
gravity = regional + bouguer_anomaly

# Adiciona ruído
noise = np.random.normal(0, 0.5, gravity.shape)
gravity += noise

# Cria elevação sintética
elevation = 500 + 0.001 * (X + Y) + np.random.normal(0, 10, X.shape)

# Converte para DataFrame
df_gravity = pd.DataFrame({
    'longitude': X.flatten(),
    'latitude': Y.flatten(),
    'elevation': elevation.flatten(),
    'gravity_mgal': gravity.flatten(),
    'regional_mgal': regional.flatten()
})

# Salva
df_gravity.to_csv(output_dir / "gravity_basin_example.csv", index=False)
print(f"✓ Gravimetria salvo: {len(df_gravity)} pontos")
print(f"  Range: {df_gravity['gravity_mgal'].min():.2f} a {df_gravity['gravity_mgal'].max():.2f} mGal")

# ============================================================================
# DATASET 2: MAGNETOMETRIA - DIQUE MAGNÉTICO
# ============================================================================

print("\nGerando dataset de magnetometria...")

# Define grid
x = np.linspace(0, 30000, 80)  # 30 km, 80 pontos
y = np.linspace(0, 30000, 80)
X, Y = np.meshgrid(x, y)

# Parâmetros do dique
strike = np.deg2rad(45)  # Direção N45E
width = 500  # largura (m)
depth_top = 100  # topo (m)
depth_bottom = 2000  # base (m)
susceptibility = 0.05  # SI
field_strength = 50000  # nT
inclination = np.deg2rad(-45)  # Campo magnético (hemisfério sul)
declination = np.deg2rad(-20)

# Geometria do dique
x_rot = (X - 15000) * np.cos(strike) + (Y - 15000) * np.sin(strike)
y_rot = -(X - 15000) * np.sin(strike) + (Y - 15000) * np.cos(strike)

# Anomalia magnética (aproximação de dipolo 2D)
def magnetic_anomaly_dike(x_rot, y_rot, width, depth_top, depth_bottom, 
                          susceptibility, field, inc, dec):
    """Calcula anomalia de um dique vertical."""
    # Simplificação: perfil 2D
    distance = np.abs(x_rot)
    anomaly = np.zeros_like(distance)
    
    # Dentro do dique
    inside = distance < width/2
    anomaly[inside] = 1000 * susceptibility * np.cos(inc)
    
    # Fora do dique (decaimento)
    outside = ~inside
    r = distance[outside] - width/2
    anomaly[outside] = 1000 * susceptibility * np.cos(inc) / (1 + (r/depth_top)**2)
    
    return anomaly * field / 50000  # Normaliza

magnetic = magnetic_anomaly_dike(x_rot, y_rot, width, depth_top, depth_bottom,
                                  susceptibility, field_strength, inclination, declination)

# Adiciona campo regional (IGRF simplificado)
igrf = field_strength + 0.0001 * X - 0.00005 * Y

# Anomalia total
magnetic_total = igrf + magnetic

# Adiciona ruído
noise = np.random.normal(0, 5, magnetic_total.shape)
magnetic_total += noise

# Elevação
elevation = 800 + np.random.normal(0, 20, X.shape)

# Converte para DataFrame
df_magnetic = pd.DataFrame({
    'longitude': X.flatten(),
    'latitude': Y.flatten(),
    'elevation': elevation.flatten(),
    'tmi_nt': magnetic_total.flatten(),
    'igrf_nt': igrf.flatten(),
    'anomaly_nt': magnetic.flatten()
})

# Salva
df_magnetic.to_csv(output_dir / "magnetic_dike_example.csv", index=False)
print(f"✓ Magnetometria salvo: {len(df_magnetic)} pontos")
print(f"  Range: {df_magnetic['tmi_nt'].min():.2f} a {df_magnetic['tmi_nt'].max():.2f} nT")

# ============================================================================
# DATASET 3: PERFIL 1D - GRAVIMETRIA
# ============================================================================

print("\nGerando perfil 1D de gravimetria...")

# Perfil sobre esfera enterrada
x_profile = np.linspace(-5000, 5000, 100)
depth_sphere = 1000  # m
radius = 500  # m
density_contrast_sphere = 300  # kg/m³

# Anomalia de esfera (fórmula de Newton)
r = np.sqrt(x_profile**2 + depth_sphere**2)
gravity_profile = (4/3) * np.pi * G * radius**3 * density_contrast_sphere * depth_sphere / r**3 * 1e5

# Ruído
gravity_profile += np.random.normal(0, 0.3, len(x_profile))

df_profile = pd.DataFrame({
    'x': x_profile,
    'gravity_mgal': gravity_profile,
    'elevation': np.ones_like(x_profile) * 300
})

df_profile.to_csv(output_dir / "gravity_profile_sphere.csv", index=False)
print(f"✓ Perfil 1D salvo: {len(df_profile)} pontos")

# ============================================================================
# README
# ============================================================================

readme_content = """# Datasets de Exemplo - GeoBot

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
"""

with open(output_dir / "README.md", 'w', encoding='utf-8') as f:
    f.write(readme_content)

print("\n✓ README gerado")
print(f"\nDatasets criados com sucesso em: {output_dir.absolute()}")
