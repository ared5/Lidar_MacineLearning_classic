# Wind Turbine ML - Blade Load Prediction

Machine Learning pipeline para predicción de cargas en palas de turbinas eólicas usando datos de simulaciones Bladed y mediciones LIDAR.

## 📋 Descripción del Proyecto

Este proyecto implementa un pipeline completo de Machine Learning para predecir momentos flectores en las raíces de las palas de turbinas eólicas utilizando:

- **Datos de entrada**: Simulaciones Bladed (velocidad del viento LIDAR, ángulos de pitch, azimuth, velocidad del rotor)
- **Targets**: Momentos flectores en raíces de palas (componentes 0P, 1P, 2P en marco fijo)
- **Modelos**: XGBoost, Random Forest, Ridge Regression

### 🎯 Objetivos

- Predecir cargas en palas usando mediciones upstream del viento (LIDAR)
- Implementar transformación Coleman para extraer componentes de frecuencia
- Comparar diferentes arquitecturas de modelos (MultiOutput vs Individual)
- Evaluar el impacto de normalización y early stopping

## 🏗️ Estructura del Proyecto

```
project/
├── configs/                    # Archivos de configuración YAML
│   ├── features.yaml          # Configuración de features
│   ├── models.yaml            # Configuración de modelos
│   └── paths.yaml             # Rutas del proyecto
├── data/
│   ├── raw/                   # Datos crudos de Bladed
│   ├── interim/               # Datos intermedios
│   ├── processed/             # Datos con features engineered
│   └── ml_traditional/        # Datasets finales para ML
├── models/                    # Modelos entrenados y scalers
│   ├── XGBoost/
│   ├── XGBoost_NoNorm/
│   ├── XGBoost_NoNorm_Individual/
│   ├── Random_Forest/
│   └── scalers/
├── notebooks/                 # Jupyter notebooks
│   ├── 01_data_import.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_eda.ipynb
│   └── 04_modeling.ipynb
├── reports/                   # Reportes y visualizaciones
│   ├── figures/
│   ├── tables/
│   └── eda/
├── scripts/                   # Scripts ejecutables
│   ├── 00_make_timeseries.py
│   ├── 01_make_features.py
│   ├── 02_build_dataset.py
│   ├── 03_split_and_scale.py
│   ├── 04_train_rf.py
│   ├── 05_train_xgb.py
│   └── 06_validate.py
├── src/windml/                # Código fuente (módulos)
│   ├── __init__.py
│   ├── config/
│   │   └── settings.py        # Gestión de configuración
│   ├── data/
│   │   ├── bladed_io.py       # Lectura de archivos Bladed
│   │   ├── assemble.py        # Ensamblado de datasets
│   │   └── split.py           # Train/test split
│   ├── features/
│   │   ├── coleman.py         # Transformación Coleman
│   │   ├── signal.py          # Lags y transformaciones
│   │   ├── angles.py          # Features angulares
│   │   └── vlos.py            # Procesamiento VLOS
│   ├── modeling/
│   │   ├── train_rf.py        # Random Forest
│   │   ├── train_xgb.py       # XGBoost
│   │   ├── validate.py        # Validación
│   │   └── metrics.py         # Métricas
│   ├── preprocessing/
│   │   ├── normalize.py       # Normalización
│   │   └── memory.py          # Optimización de memoria
│   └── eda/
│       ├── plots.py           # Visualizaciones
│       └── lag_analysis.py    # Análisis de lags
├── pyproject.toml             # Configuración del proyecto
└── README.md                  # Este archivo
```

## 🚀 Inicio Rápido

### 1. Instalación

```bash
# Clonar repositorio
git clone <repo_url>
cd project

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -e .
```

### 2. Configuración

Edita los archivos de configuración en `configs/`:

**`configs/paths.yaml`**: Rutas del proyecto
**`configs/features.yaml`**: Features a crear y usar
**`configs/models.yaml`**: Hiperparámetros de modelos

### 3. Pipeline Completo

#### Opción A: Scripts secuenciales

```bash
# 1. Generar CSVs desde Bladed
python scripts/01_make_timeseries.py

# 2. Crear features engineered
python scripts/02_make_features.py

# 3. Ensamblar dataset completo
python scripts/03_build_dataset.py

# 4. Split train/test y normalizar
python scripts/04_split_and_scale.py

# 5. Entrenar modelos
python scripts/05_train_rf.py
python scripts/06_train_xgb.py

# 6. Validar modelos
python scripts/07_validate.py
```

#### Opción B: Notebooks interactivos

Abre y ejecuta los notebooks en orden:

1. `notebooks/01_data_import.ipynb`
2. `notebooks/02_feature_engineering.ipynb`
3. `notebooks/03_eda.ipynb`
4. `notebooks/04_modeling.ipynb`

## 📊 Feature Engineering

El proyecto implementa múltiples transformaciones de features:

### 1. **Transformación Coleman** (0P, 1P, 2P)

Convierte momentos flectores del marco rotante al marco fijo:

```python
from windml import create_frequency_components_1P_2P

df = create_frequency_components_1P_2P(df, apply_filtering=True)
# Crea: M_0, M_1c, M_1s, M_2c, M_2s
```

- **M_0**: Componente 0P (lento, DC)
- **M_1c, M_1s**: Componente 1P (coseno y seno)
- **M_2c, M_2s**: Componente 2P (coseno y seno)

### 2. **Lags de VLOS**

Crea versiones retardadas de mediciones de viento:

```python
from windml import create_vlos_lags

df = create_vlos_lags(
    df, 
    lag_seconds_list=[2, 5, 8, 11, 14, 17, 20, 23, 26],
    range_values=[5]  # Solo RANGE5
)
```

### 3. **Componentes Trigonométricas**

Convierte ángulos a sin/cos para evitar discontinuidades:

```python
from windml import create_azimuth_components, create_yawerror_components

df = create_azimuth_components(df)      # sin/cos azimuth
df = create_yawerror_components(df)     # sin/cos yaw error
```

### 4. **Estadísticas de Viento**

Shear vertical/horizontal, media, desviación estándar, etc.

## 🤖 Modelos Implementados

### XGBoost (3 variantes)

1. **XGBoost MultiOutput** (normalizado)
   - Un modelo para todas las salidas
   - Datos normalizados
   
2. **XGBoost MultiOutput** (sin normalizar)
   - Un modelo para todas las salidas
   - Datos en escala original
   
3. **XGBoost Individual** (sin normalizar + early stopping)
   - Un modelo por cada target
   - Early stopping independiente
   - Óptimo número de iteraciones por target

### Random Forest

- Modelo baseline robusto
- Sin necesidad de normalización
- Interpretable

### Ridge Regression

- Modelo lineal con regularización L2
- Baseline simple y rápido

## 📈 Evaluación de Modelos

Métricas calculadas automáticamente:

- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **R²** (R-squared Score)

Visualizaciones generadas:

- Predicciones vs Real
- Análisis de residuos
- Series temporales (Real vs Predicho)
- Feature importance
- Métricas por target

## 🔧 Uso de Módulos

### Ejemplo básico

```python
from windml import get_config, create_frequency_components_1P_2P
from windml.modeling.train_xgb import train_xgboost_individual
import pandas as pd

# Cargar configuración
config = get_config()

# Cargar datos
df = pd.read_csv(config.get_path('data', 'processed') / 'simulation_001.csv')

# Feature engineering
df = create_frequency_components_1P_2P(df)

# Preparar datos
features = [...] # Lista de columnas de features
targets = ['M_0', 'M_1c', 'M_1s', 'M_2c', 'M_2s']

X = df[features]
y = df[targets]

# Entrenar modelo
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

models, metrics = train_xgboost_individual(
    X_tr, y_tr, 
    X_val, y_val,
    params={
        'n_estimators': 2000,
        'max_depth': 6,
        'learning_rate': 0.05,
        'early_stopping_rounds': 300
    },
    save_path=config.get_path('models', 'xgboost_individual')
)

print(metrics)
```

## 📝 Configuración Detallada

### features.yaml

```yaml
engineered_features:
  coleman_transform:
    enabled: true
    apply_filtering: true
  
  vlos_lags:
    enabled: true
    lag_seconds: [2, 5, 8, 11, 14, 17, 20, 23, 26]
    range_filter:
      enabled: true
      range_values: [5]
```

### models.yaml

```yaml
models:
  xgboost_individual:
    enabled: true
    normalize_data: false
    params:
      n_estimators: 2000
      max_depth: 6
      learning_rate: 0.05
      early_stopping_rounds: 300
```

## 🎓 Conceptos Clave

### Transformación Coleman

Convierte cargas del **marco rotante** (blade root moments) al **marco fijo** (tower/nacelle):

- **Marco rotante**: Las palas giran → momentos oscilan a frecuencia 1P, 2P, 3P...
- **Marco fijo**: Proyección al suelo → componentes estacionarios

**Ventajas**:
- Separa efectos gravitacionales (1P) de aerodinámicos (2P+)
- Facilita control individual de cargas
- Mejor interpretación física

### Series Temporales y Split

El split train/test se hace a **nivel de series completas** para evitar data leakage:

```python
# ✓ CORRECTO: Cada serie completa en train O test
series_0: train
series_1: train
series_2: test
series_3: train
...

# ✗ INCORRECTO: Mezclar muestras de una misma serie
series_0: [samples 0-100] → train, [samples 101-200] → test
```

## 🐛 Troubleshooting

### Error: "Module not found: windml"

```bash
# Instalar el paquete en modo desarrollo
pip install -e .
```

### Error: "postprocessbladed not found"

El módulo `postprocessbladed` debe estar instalado para leer archivos Bladed:

```bash
# Añadir ruta al sys.path o instalar el paquete
```

### Error: Memory issues con datasets grandes

Usa lectura por chunks y dtypes optimizados:

```python
# El código ya incluye optimizaciones de memoria
chunk_size = 5000
dtype_dict = {col: 'float32' for col in usecols}
```

## 📚 Referencias

- **XGBoost**: [Documentación oficial](https://xgboost.readthedocs.io/)
- **Coleman Transform**: Application of Multi-Blade Coordinate Transformation to Wind Turbine Applications
- **Scikit-learn**: [Documentación](https://scikit-learn.org/)

## 👥 Autores

Wind ML Team

## 📄 Licencia

[Tu licencia aquí]

## 🙏 Agradecimientos

- Equipo de desarrollo
- Proveedores de datos de simulaciones Bladed
- Comunidad open-source

---

**Última actualización**: Febrero 2026
