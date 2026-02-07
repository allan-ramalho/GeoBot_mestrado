"""
Script para gerar dados sintéticos realistas de gravimetria
Região: Baía de Guanabara, Rio de Janeiro
"""
import numpy as np
import pandas as pd

# Coordenadas da Baía de Guanabara
# Longitude: -43.25 a -43.05 (oeste)
# Latitude: -22.95 a -22.75 (sul)

np.random.seed(42)

# Grid regular
n_points_x = 100
n_points_y = 100
n_total = n_points_x * n_points_y

# Criar grid
lon = np.linspace(-43.25, -43.05, n_points_x)
lat = np.linspace(-22.95, -22.75, n_points_y)
LON, LAT = np.meshgrid(lon, lat)

# Achatar para vetores
longitude = LON.flatten()
latitude = LAT.flatten()

# Elevação (topografia/batimetria)
# Baía é mais baixa (negativa), continente mais alto
elevation = 10 * np.sin(2 * np.pi * (longitude + 43.15) / 0.2) + \
            5 * np.cos(2 * np.pi * (latitude + 22.85) / 0.2) + \
            np.random.normal(0, 2, n_total)

# Gravidade (mGal)
# Regional: campo gravitacional regional (diminui para sul)
regional_mgal = -978000 + 800 * (latitude + 22.85) / 0.2

# Anomalia: devido à bacia sedimentar e variações de densidade
# Bacia causa anomalia negativa
bacia_anomaly = -15 * np.exp(-((longitude + 43.15)**2 + (latitude + 22.85)**2) / 0.01)

# Ruído
noise = np.random.normal(0, 1, n_total)

# Gravidade observada
gravity_mgal = regional_mgal + bacia_anomaly + noise

# Criar DataFrame
df = pd.DataFrame({
    'longitude': longitude,
    'latitude': latitude,
    'elevation': elevation,
    'gravity_mgal': gravity_mgal,
    'regional_mgal': regional_mgal
})

# Salvar
output_file = 'example_data/gravity_basin_example.csv'
df.to_csv(output_file, index=False)
print(f"✅ Dados salvos em {output_file}")
print(f"📊 {len(df):,} pontos gerados")
print(f"📍 Longitude: {df['longitude'].min():.4f} a {df['longitude'].max():.4f}")
print(f"📍 Latitude: {df['latitude'].min():.4f} a {df['latitude'].max():.4f}")
print(f"⚖️ Gravidade: {df['gravity_mgal'].min():.2f} a {df['gravity_mgal'].max():.2f} mGal")
