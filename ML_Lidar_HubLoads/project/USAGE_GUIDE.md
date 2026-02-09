# 📖 GUÍA DE USO - ML_Lidar_HubLoads Project

**Guía completa para ejecutar el proyecto paso a paso**

---

## 📋 Tabla de Contenidos

1. [Visión General](#visión-general)
2. [Requisitos Previos](#requisitos-previos)
3. [Pipeline Completo](#pipeline-completo)
4. [Guía de Scripts](#guía-de-scripts)
5. [Guía de Módulos](#guía-de-módulos)
6. [Ejemplos de Uso](#ejemplos-de-uso)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 Visión General

Este proyecto implementa un pipeline completo de Machine Learning para predecir cargas en palas de aerogeneradores usando:
- **Datos**: Simulaciones Bladed + mediciones LIDAR
- **Features**: Transformación Coleman, lags temporales, estadísticas de viento
- **Modelos**: XGBoost (3 variantes), Random Forest, Ridge Regression

### Flujo del Pipeline

```
Simulaciones Bladed (*.bin, *..$PJ)
    ↓
[01_make_timeseries.py] → CSVs crudos
    ↓
[02_make_features.py] → CSVs con features engineered
    ↓
[Manual: Combinar CSVs] → Dataset completo
    ↓
[Manual: Split + Normalize] → Train/Test sets
    ↓
[Manual: Entrenar modelos] → Modelos guardados
    ↓
[Manual: Validación] → Métricas + Gráficas
```

---

## ⚙️ Requisitos Previos

### 1. Instalación

```bash
# Crear entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt

# Instalar package en modo desarrollo
pip install -e .
```

### 2. Verificar Instalación

```python
# Probar que todos los módulos importan correctamente
python -c "from windml import *; print('✅ Instalación OK!')"
```

### 3. Configurar Rutas

Editar `configs/paths.yaml` con tus rutas locales:

```yaml
project: "C:/tu/ruta/project"
data:
  raw: "C:/tu/ruta/data/raw"
  processed: "C:/tu/ruta/data/processed"
# ... etc
```

---

## 🔄 Pipeline Completo

### Orden de Ejecución Recomendado

#### **PASO 1: Generar Time Series desde Bladed**

```bash
python scripts/01_make_timeseries.py
```

**¿Qué hace?**
- Lee archivos binarios de Bladed (`.bin`, `.$PJ`)
- Extrae señales configuradas en `VAR_DICTS`
- Guarda CSVs individuales por simulación en `data/raw/`

**Configuración necesaria:**
Editar dentro del script:
```python
LOADPATH = Path("C:/ruta/a/simulaciones/bladed")
FILE_NAMES = ['0040_DLC12a', '0041_DLC12a', ...]  # Lista de archivos
```

**Salida:**
```
data/raw/
├── 0040_DLC12a_150_000.csv
├── 0041_DLC12a_160_000.csv
└── ...
```

---

#### **PASO 2: Aplicar Feature Engineering**

```bash
python scripts/02_make_features.py
```

**¿Qué hace?**
- Lee CSVs de `data/raw/`
- Aplica transformaciones según `configs/features.yaml`:
  - Coleman transform (M_0, M_1c, M_1s, M_2c, M_2s)
  - VLOS lags (2-26 segundos)
  - Componentes trigonométricas (sin/cos de azimuth, yaw)
  - Estadísticas de viento (U_mean, U_std, shear)
  - Pitch Coleman (pitch_0, pitch_1c, pitch_1s + rates)
- Guarda CSVs procesados en `data/processed/`

**Configuración:**
Editar `configs/features.yaml` para activar/desactivar features:

```yaml
ml_features:
  use_trigonometric: true
  use_vlos_lags: true
  use_coleman_transform: true
  use_wind_statistics: true
  use_pitch_coleman: true
```

**Salida:**
```
data/processed/
├── 0040_DLC12a_150_000.csv  (con nuevas columnas)
├── 0041_DLC12a_160_000.csv
└── ...
```

---

#### **PASO 3: Combinar CSVs en Dataset Completo**

```python
# Crear script personalizado o ejecutar en notebook
from windml import assemble_csvs, get_path

# Combinar todos los CSVs procesados
df_complete = assemble_csvs(
    csv_folder=get_path('data', 'processed'),
    output_path=get_path('data', 'processed') / '0000_Complete_dataset.csv',
    batch_size=10,
    optimize_memory=True,
    verbose=True
)

print(f"Dataset completo: {df_complete.shape}")
```

**¿Qué hace?**
- Carga todos los CSVs de `data/processed/`
- Los concatena en un solo DataFrame
- Optimiza tipos de datos (float64→float32) para ahorrar memoria
- Guarda dataset completo

**Salida:**
```
data/processed/
└── 0000_Complete_dataset.csv  (dataset único con todos los datos)
```

---

#### **PASO 4: Split Train/Test por Series Temporales**

```python
from windml import train_test_split_series, print_split_summary
import pandas as pd

# Cargar dataset completo
df = pd.read_csv('data/processed/0000_Complete_dataset.csv')

# Split inteligente por series completas (no aleatorio)
df_train, df_test, info = train_test_split_series(
    df,
    test_size=0.2,  # 20% de series van a test
    split_strategy='uniform',  # Cada 5ta serie a test
    time_col='Time',
    random_state=42
)

# Ver resumen del split
print_split_summary(info)

# Separar features (X) y targets (y)
feature_cols = [...]  # Lista de columnas de features
target_cols = ['M_0', 'M_1c', 'M_1s', 'M_2c', 'M_2s']  # Targets Coleman

X_train = df_train[feature_cols]
y_train = df_train[target_cols]
X_test = df_test[feature_cols]
y_test = df_test[target_cols]

# Guardar splits
import joblib
joblib.dump(X_train, 'data/processed/X_train.pkl')
joblib.dump(y_train, 'data/processed/y_train.pkl')
joblib.dump(X_test, 'data/processed/X_test.pkl')
joblib.dump(y_test, 'data/processed/y_test.pkl')
```

**¿Qué hace?**
- Identifica series temporales individuales (detecta reinicios en columna `Time`)
- Asigna series **completas** a train o test (evita data leakage)
- Garantiza que ninguna simulación está dividida entre train y test

**Importante:** Este método es CRÍTICO para datos de series temporales. Un split aleatorio crearía data leakage.

---

#### **PASO 5: Normalizar con Scalers Independientes**

```python
from windml import fit_and_transform_independent, save_scalers
import joblib

# Cargar splits
X_train = joblib.load('data/processed/X_train.pkl')
y_train = joblib.load('data/processed/y_train.pkl')
X_test = joblib.load('data/processed/X_test.pkl')
y_test = joblib.load('data/processed/y_test.pkl')

# Crear scalers y normalizar (SOLO desde train)
X_train_norm, y_train_norm, scalers_X, scalers_y, X_test_norm, y_test_norm = \
    fit_and_transform_independent(
        X_train, y_train, X_test, y_test,
        scaler_type='standard'  # StandardScaler (mean=0, std=1)
    )

# Guardar datos normalizados
joblib.dump(X_train_norm, 'data/processed/X_train_norm.pkl')
joblib.dump(y_train_norm, 'data/processed/y_train_norm.pkl')
joblib.dump(X_test_norm, 'data/processed/X_test_norm.pkl')
joblib.dump(y_test_norm, 'data/processed/y_test_norm.pkl')

# Guardar scalers (necesarios para desnormalizar predicciones)
save_scalers(scalers_X, scalers_y, output_dir='models/scalers')
```

**¿Por qué scalers independientes?**
- Cada variable tiene su propio scaler
- Permite desnormalizar targets individuales correctamente
- Crítico para calcular métricas en escala original

---

#### **PASO 6: Entrenar Modelos**

##### **Opción A: XGBoost MultiOutput**

```python
from windml import train_xgboost_multioutput, get_model_config
import joblib

# Cargar datos normalizados
X_train_norm = joblib.load('data/processed/X_train_norm.pkl')
y_train_norm = joblib.load('data/processed/y_train_norm.pkl')

# Obtener parámetros desde configuración
params = get_model_config('xgboost')

# Entrenar
model = train_xgboost_multioutput(
    X_train_norm, 
    y_train_norm,
    params=params,
    model_name='XGBoost_MultiOutput',
    save_path='models/XGBoost/xgboost_multioutput.pkl'
)

print("✅ Modelo XGBoost MultiOutput entrenado")
```

##### **Opción B: XGBoost Individual (con Early Stopping)**

```python
from windml import train_xgboost_individual
import joblib

# Cargar datos (NO normalizados - el modelo maneja bien sin normalizar)
X_train = joblib.load('data/processed/X_train.pkl')
y_train = joblib.load('data/processed/y_train.pkl')
X_test = joblib.load('data/processed/X_test.pkl')  # Para early stopping
y_test = joblib.load('data/processed/y_test.pkl')

# Parámetros
params = get_model_config('xgboost_individual')

# Entrenar (un modelo por target con early stopping independiente)
models_dict, metrics_dict = train_xgboost_individual(
    X_train, y_train,
    X_val=X_test, y_val=y_test,  # Early stopping en test
    params=params,
    save_path='models/XGBoost/individual/'
)

print(f"✅ Entrenados {len(models_dict)} modelos individuales")
print("Best iterations:", {k: v['best_iteration'] for k, v in metrics_dict.items()})
```

##### **Opción C: Random Forest**

```python
from windml import train_random_forest
import joblib

# Cargar datos normalizados
X_train_norm = joblib.load('data/processed/X_train_norm.pkl')
y_train_norm = joblib.load('data/processed/y_train_norm.pkl')

# Parámetros
params = get_model_config('random_forest')

# Entrenar
model_rf = train_random_forest(
    X_train_norm,
    y_train_norm,
    params=params,
    verbose=True
)

# Guardar
from windml import save_random_forest
save_random_forest(
    model_rf,
    output_path='models/Random_Forest/rf_model.pkl',
    metadata={'params': params, 'date': '2026-02-09'}
)
```

---

#### **PASO 7: Validación y Métricas**

```python
from windml import (
    predict_xgboost,
    denormalize_with_independent_scalers,
    load_scalers,
    evaluate_random_forest
)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd
import joblib

# Cargar modelo y datos
model = joblib.load('models/XGBoost/xgboost_multioutput.pkl')
X_test_norm = joblib.load('data/processed/X_test_norm.pkl')
y_test_norm = joblib.load('data/processed/y_test_norm.pkl')
y_test = joblib.load('data/processed/y_test.pkl')  # Original scale

# Predicciones (en escala normalizada)
y_pred_norm = model.predict(X_test_norm)
y_pred_norm = pd.DataFrame(y_pred_norm, columns=y_test_norm.columns, index=y_test_norm.index)

# Desnormalizar predicciones
scalers_X, scalers_y = load_scalers('models/scalers')
y_pred = denormalize_with_independent_scalers(y_pred_norm, scalers_y)

# Calcular métricas (en escala original)
print("\n" + "="*70)
print("MÉTRICAS DE VALIDACIÓN")
print("="*70)

for target in y_test.columns:
    y_true = y_test[target]
    y_p = y_pred[target]
    
    rmse = np.sqrt(mean_squared_error(y_true, y_p))
    mae = mean_absolute_error(y_true, y_p)
    r2 = r2_score(y_true, y_p)
    
    print(f"\n{target}:")
    print(f"  RMSE: {rmse:10.2f} kNm")
    print(f"  MAE:  {mae:10.2f} kNm")
    print(f"  R²:   {r2:10.4f}")

print("="*70)
```

---

## 📚 Guía de Módulos

### 🗂️ **src/windml/config/**

#### `settings.py`

```python
from windml import get_config, get_path, get_feature_config, get_model_config

# Obtener configuración global
config = get_config()

# Obtener rutas
data_folder = get_path('data', 'raw')
models_folder = get_path('models')

# Obtener configuración de features
vlos_lags = get_feature_config('vlos', 'lag_seconds')  # [2, 4, 6, ..., 26]

# Obtener hiperparámetros de modelo
xgb_params = get_model_config('xgboost')  # Dict con n_estimators, max_depth, etc.
```

---

### 📂 **src/windml/data/**

#### `bladed_io.py` - Importar desde Bladed

```python
from windml import load_bladed_data
from pathlib import Path

# Cargar simulaciones Bladed
data_dict = load_bladed_data(
    loadpath=Path("C:/simulaciones/bladed"),
    file_names=['0040_DLC12a', '0041_DLC12a'],
    var_dicts={
        'hub_rotating': ['Stationary hub Mx', 'Stationary hub My', ...],
        'Aero': ['AeroTq', 'AeroFx', ...]
    },
    resultspath=Path("data/raw"),
    add_units=True,
    aero_positions=[0.0, 6.0, 18.0, 30.0, 46.0, 59.0, 68.25]
)

# Retorna dict con DataFrames por tipo de señal
print(data_dict.keys())  # ['hub_rotating', 'Aero', ...]
```

#### `assemble.py` - Combinar CSVs

```python
from windml import assemble_csvs, optimize_dataframe_dtypes
import pandas as pd

# Combinar múltiples CSVs
df = assemble_csvs(
    csv_folder='data/processed',
    output_path='data/complete.csv',
    pattern='*.csv',  # o 'DLC12a*.csv' para filtrar
    batch_size=10,
    optimize_memory=True
)

# Optimizar DataFrame existente
df_optimized = optimize_dataframe_dtypes(df)
print(f"Memoria ahorrada: {df.memory_usage().sum() - df_optimized.memory_usage().sum()}")
```

#### `split.py` - Split por Series Temporales

```python
from windml import train_test_split_series, identify_time_series, create_series_mapping

# Split automático
df_train, df_test, info = train_test_split_series(
    df,
    test_size=0.2,
    split_strategy='uniform',  # 'uniform', 'random', 'end', 'start'
    time_col='Time'
)

# O especificar series manualmente
df_train, df_test, info = train_test_split_series(
    df,
    test_series_ids=[3, 7, 12, 18, 25],  # Series específicas a test
    time_col='Time'
)

# Crear mapeo series_id ↔ Name_DLC
mapping = create_series_mapping(
    df,
    time_col='Time',
    name_col='Name_DLC',
    output_path='data/series_mapping.csv'
)
print(mapping)
```

---

### 🔧 **src/windml/features/**

#### `coleman.py` - Transformación Coleman para Cargas

```python
from windml import create_frequency_components_1P_2P
import pandas as pd

# Aplicar Coleman transform
df_with_coleman = create_frequency_components_1P_2P(
    df,
    apply_filtering=True  # Aplica filtros bandpass y lowpass
)

# Nuevas columnas creadas:
# - M_0: Componente 0P (DC, promedio)
# - M_1c, M_1s: Componentes 1P (gravedad, once-per-revolution)
# - M_2c, M_2s: Componentes 2P (twice-per-revolution)
```

**¿Cuándo usar?**
- Para predecir cargas de palas individuales transformadas a sistema fijo
- Reduce dimensionalidad (2 palas → 5 componentes independientes)
- Captura efectos gravitacionales (1P) y aerodinámicos (2P)

#### `signal.py` - Lags y Componentes Trigonométricas

```python
from windml import create_vlos_lags, create_azimuth_components

# VLOS lags (crucial para predicción anticipada)
df = create_vlos_lags(
    df,
    lag_seconds_list=[2, 4, 6, 8, 10, 14, 18, 22, 26],
    range_values=[5],  # RANGE5 típicamente
    range_min=1,
    range_max=7,
    include_all_ranges=False
)

# Componentes trigonométricas
df = create_azimuth_components(df)  # azimuth_cos, azimuth_sin
df = create_yawerror_components(df)  # yaw_error_cos, yaw_error_sin
```

**Lags de VLOS:**
- Permiten usar mediciones LIDAR de 2-26 segundos antes
- Crítico: la turbulencia tarda tiempo en llegar desde la posición LIDAR hasta el rotor
- Típicamente lag óptimo: 6-14 segundos según velocidad de viento

#### `windfield.py` - Estadísticas de Viento

```python
from windml import calculate_wind_statistics, create_turbulence_intensity

# Estadísticas con ventana temporal
df = calculate_wind_statistics(
    df,
    wind_speed_col='Hub wind speed Ux',
    window_size=100,  # 100 muestras (~10s con fs=10Hz)
    include_shear=True
)

# Intensidad de turbulencia
df = create_turbulence_intensity(
    df,
    wind_speed_col='Hub wind speed Ux',
    window_size=100
)

# Nuevas columnas:
# - U_mean: Velocidad de viento media
# - U_std: Desviación estándar
# - U_shear: Corte de viento vertical
# - TI: Intensidad de turbulencia (std/mean)
```

#### `angles.py` - Pitch Coleman Transform

```python
from windml import create_pitch_coleman_transform, detect_ipc_presence

# Detectar si hay IPC (Individual Pitch Control)
has_ipc, stats = detect_ipc_presence(df)
if has_ipc:
    print("✅ IPC detectado - pitch_1c/pitch_1s serán significativos")
else:
    print("❌ Solo pitch colectivo - componentes 1P ≈ 0")

# Aplicar Coleman a pitch
df = create_pitch_coleman_transform(
    df,
    pitch_blade1_col='Pitch angle 1',
    pitch_blade2_col='Pitch angle 2',
    azimuth_col='Rotor azimuth',
    include_rates=True,  # Incluye derivadas temporales
    include_2p=False
)

# Nuevas columnas:
# - pitch_0: Pitch colectivo
# - pitch_1c, pitch_1s: Componentes 1P
# - pitch_0_rate, pitch_1c_rate, pitch_1s_rate: Tasas de cambio
```

---

### ⚙️ **src/windml/preprocessing/**

#### `normalize.py` - Normalización con Scalers Independientes

```python
from windml import (
    create_independent_scalers,
    normalize_with_independent_scalers,
    denormalize_with_independent_scalers,
    save_scalers,
    load_scalers
)

# TRAIN: Crear scalers
scalers_X = create_independent_scalers(X_train, scaler_type='standard')
scalers_y = create_independent_scalers(y_train, scaler_type='standard')

# TRAIN: Normalizar
X_train_norm = normalize_with_independent_scalers(X_train, scalers_X)
y_train_norm = normalize_with_independent_scalers(y_train, scalers_y)

# TEST: Normalizar con los MISMOS scalers (no crear nuevos)
X_test_norm = normalize_with_independent_scalers(X_test, scalers_X)
y_test_norm = normalize_with_independent_scalers(y_test, scalers_y)

# PREDICCIÓN: Desnormalizar
y_pred_norm = model.predict(X_test_norm)
y_pred = denormalize_with_independent_scalers(y_pred_norm, scalers_y)

# Guardar/Cargar scalers
save_scalers(scalers_X, scalers_y, output_dir='models/scalers')
scalers_X, scalers_y = load_scalers('models/scalers')
```

**¿Por qué independientes?**
- Cada variable tiene su media/std propia
- Fundamental para desnormalizar predicciones correctamente
- Permite calcular métricas en escala original

---

### 🤖 **src/windml/modeling/**

#### `train_xgb.py` - XGBoost

```python
from windml import (
    train_xgboost_multioutput,
    train_xgboost_individual,
    predict_xgboost
)

# MultiOutput: Un modelo para todos los targets
model_multi = train_xgboost_multioutput(
    X_train, y_train,
    params={'n_estimators': 1000, 'max_depth': 6, 'learning_rate': 0.05},
    model_name='XGB_Multi',
    save_path='models/xgb_multi.pkl'
)

# Individual: Un modelo por target con early stopping
models, metrics = train_xgboost_individual(
    X_train, y_train,
    X_val, y_val,
    params={'n_estimators': 2000, 'early_stopping_rounds': 300},
    save_path='models/xgb_individual/'
)

# Predicciones
y_pred = predict_xgboost(model_multi, X_test, model_type='multioutput')
# o
y_pred = predict_xgboost(models, X_test, model_type='individual')
```

**Cuándo usar cada uno:**
- **MultiOutput**: Más rápido, compartido entre targets, bueno para targets correlacionados
- **Individual**: Mejor accuracy, early stopping independiente por target, más flexible

#### `train_rf.py` - Random Forest

```python
from windml import (
    train_random_forest,
    evaluate_random_forest,
    get_feature_importance
)

# Entrenar
model_rf = train_random_forest(
    X_train, y_train,
    params={
        'n_estimators': 200,
        'max_depth': 20,
        'max_features': 'sqrt',
        'oob_score': True
    }
)

# Evaluar
metrics = evaluate_random_forest(model_rf, X_test, y_test, 'Test')
print(metrics)

# Feature importance
importance = get_feature_importance(model_rf, X_train.columns, top_n=15)
print(importance)
```

---

## 💡 Ejemplos de Uso

### Ejemplo 1: Pipeline Completo desde Cero

```python
# Script completo: pipeline_completo.py
from windml import *
import pandas as pd
import numpy as np
from pathlib import Path

# 1. CONFIGURACIÓN
project_root = Path("C:/project")
config = get_config()

# 2. CARGAR DATOS
print("Cargando dataset completo...")
df = pd.read_csv(project_root / "data/processed/0000_Complete_dataset.csv")

# 3. SPLIT POR SERIES
print("Dividiendo train/test...")
df_train, df_test, info = train_test_split_series(
    df, test_size=0.2, split_strategy='uniform', time_col='Time'
)
print_split_summary(info)

# 4. DEFINIR FEATURES Y TARGETS
features = [col for col in df_train.columns if col not in ['Time', 'series_id', 'Name_DLC']]
targets = ['M_0', 'M_1c', 'M_1s', 'M_2c', 'M_2s']

X_train = df_train[features]
y_train = df_train[targets]
X_test = df_test[features]
y_test = df_test[targets]

# 5. NORMALIZAR
print("Normalizando...")
X_tr_n, y_tr_n, sc_X, sc_y, X_te_n, y_te_n = fit_and_transform_independent(
    X_train, y_train, X_test, y_test
)

# 6. ENTRENAR XGBOOST
print("Entrenando XGBoost...")
model_xgb = train_xgboost_multioutput(
    X_tr_n, y_tr_n,
    params=get_model_config('xgboost'),
    save_path=project_root / 'models/xgb_model.pkl'
)

# 7. VALIDAR
print("Validando...")
y_pred_norm = model_xgb.predict(X_te_n)
y_pred_norm = pd.DataFrame(y_pred_norm, columns=targets, index=y_te_n.index)
y_pred = denormalize_with_independent_scalers(y_pred_norm, sc_y)

# 8. MÉTRICAS
from sklearn.metrics import mean_squared_error, r2_score
for target in targets:
    rmse = np.sqrt(mean_squared_error(y_test[target], y_pred[target]))
    r2 = r2_score(y_test[target], y_pred[target])
    print(f"{target}: RMSE={rmse:.2f}, R²={r2:.4f}")

print("✅ Pipeline completado!")
```

### Ejemplo 2: Comparar Múltiples Modelos

```python
from windml import *
import pandas as pd
import numpy as np

# Cargar datos
X_train = ...
y_train = ...
X_test = ...
y_test = ...

# Normalizar
X_tr_n, y_tr_n, sc_X, sc_y, X_te_n, y_te_n = fit_and_transform_independent(
    X_train, y_train, X_test, y_test
)

# =========== MODELOS ===========

# 1. XGBoost MultiOutput
print("\n[1/3] XGBoost MultiOutput...")
model_xgb_multi = train_xgboost_multioutput(X_tr_n, y_tr_n, params=get_model_config('xgboost'))
y_pred_xgb_multi = denormalize_with_independent_scalers(
    pd.DataFrame(model_xgb_multi.predict(X_te_n), columns=y_test.columns), sc_y
)

# 2. XGBoost Individual
print("\n[2/3] XGBoost Individual...")
models_indiv, _ = train_xgboost_individual(X_train, y_train, X_test, y_test, 
                                           params=get_model_config('xgboost_individual'))
y_pred_xgb_indiv = predict_xgboost(models_indiv, X_test, model_type='individual')

# 3. Random Forest
print("\n[3/3] Random Forest...")
model_rf = train_random_forest(X_tr_n, y_tr_n, params=get_model_config('random_forest'))
y_pred_rf = denormalize_with_independent_scalers(
    pd.DataFrame(model_rf.predict(X_te_n), columns=y_test.columns), sc_y
)

# =========== COMPARACIÓN ===========
from sklearn.metrics import mean_squared_error, r2_score

print("\n" + "="*80)
print("COMPARACIÓN DE MODELOS")
print("="*80)

models_results = {
    'XGBoost MultiOutput': y_pred_xgb_multi,
    'XGBoost Individual': y_pred_xgb_indiv,
    'Random Forest': y_pred_rf
}

for model_name, y_pred in models_results.items():
    print(f"\n{model_name}:")
    for target in y_test.columns:
        rmse = np.sqrt(mean_squared_error(y_test[target], y_pred[target]))
        r2 = r2_score(y_test[target], y_pred[target])
        print(f"  {target}: RMSE={rmse:8.2f} kNm, R²={r2:7.4f}")

print("="*80)
```

---

## 🔧 Troubleshooting

### Problema 1: Error de importación

```
ImportError: cannot import name 'create_vlos_lags' from 'windml'
```

**Solución:**
```bash
# Reinstalar package en modo desarrollo
pip install -e .
```

### Problema 2: KeyError en configuración

```
KeyError: 'data'
```

**Solución:** Verificar que `configs/paths.yaml` tiene todas las claves necesarias.

### Problema 3: MemoryError al combinar CSVs

```
MemoryError: Unable to allocate array
```

**Solución:** Usar batch_size más pequeño y optimize_memory=True:
```python
df = assemble_csvs(..., batch_size=5, optimize_memory=True)
```

### Problema 4: Scalers no coinciden con columnas

```
ValueError: Column 'X' not found in scalers
```

**Solución:** Asegurarse de usar los MISMOS scalers creados del training set para test set.

### Problema 5: NaN en predicciones después de desnormalizar

**Causa:** Intentar desnormalizar con scalers incorrectos.

**Solución:**
```python
# Verificar que las columnas coinciden
print(y_pred_norm.columns)
print(list(scalers_y.keys()))

# Deben ser idénticas
```

---

## 📞 Soporte

Para más información:
- Ver `README.md` para arquitectura general
- Ver `configs/` para configuraciones disponibles
- Ver docstrings en el código: `help(create_frequency_components_1P_2P)`

---

**Última actualización:** 9 febrero 2026
