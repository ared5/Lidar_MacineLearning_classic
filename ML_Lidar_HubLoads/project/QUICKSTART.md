# ⚡ QUICKSTART - Ejecución Rápida

Guía ultrarrápida para ejecutar el proyecto en 5 minutos.

---

## 🚀 Instalación Rápida

```bash
# 1. Crear entorno virtual
python -m venv venv
venv\Scripts\activate

# 2. Instalar
pip install -r requirements.txt
pip install -e .

# 3. Verificar
python -c "from windml import *; print('✅ OK!')"
```

---

## 📝 Configuración Mínima

Editar `configs/paths.yaml`:
```yaml
project: "C:/tu/ruta/project"
data:
  raw: "C:/tu/ruta/data/raw"
  processed: "C:/tu/ruta/data/processed"
```

---

## ⚡ Pipeline en 3 Pasos

### **PASO 1: Generar CSVs desde Bladed**

Editar `scripts/01_make_timeseries.py` con tus rutas:
```python
LOADPATH = Path("C:/ruta/simulaciones/bladed")
FILE_NAMES = ['0040_DLC12a', '0041_DLC12a', ...]
```

Ejecutar:
```bash
python scripts/01_make_timeseries.py
```

**✅ Output:** CSVs en `data/raw/`

---

### **PASO 2: Aplicar Feature Engineering**

```bash
python scripts/02_make_features.py
```

**✅ Output:** CSVs procesados en `data/processed/`

---

### **PASO 3: Entrenar Modelo** 

Crear `scripts/03_train_quick.py`:

```python
from windml import *
import pandas as pd
import numpy as np
from pathlib import Path

# ========== CARGAR DATOS ==========
# Combinar CSVs
df = assemble_csvs(
    csv_folder='data/processed',
    output_path='data/processed/complete.csv',
    batch_size=10
)

# Split train/test por series
df_train, df_test, info = train_test_split_series(
    df, test_size=0.2, split_strategy='uniform', time_col='Time'
)

# Seleccionar features y targets
feature_cols = [col for col in df.columns 
                if col not in ['Time', 'series_id', 'Name_DLC']]
target_cols = ['M_0', 'M_1c', 'M_1s', 'M_2c', 'M_2s']

X_train = df_train[feature_cols]
y_train = df_train[target_cols]
X_test = df_test[feature_cols]
y_test = df_test[target_cols]

# ========== NORMALIZAR ==========
results = fit_and_transform_independent(X_train, y_train, X_test, y_test)
X_tr_n, y_tr_n, sc_X, sc_y, X_te_n, y_te_n = results

save_scalers(sc_X, sc_y, 'models/scalers')

# ========== ENTRENAR ==========
print("\n🚀 Entrenando XGBoost...")
model = train_xgboost_multioutput(
    X_tr_n, y_tr_n,
    params=get_model_config('xgboost'),
    save_path='models/xgb_model.pkl'
)

# ========== VALIDAR ==========
print("\n📊 Validando...")
y_pred_norm = pd.DataFrame(
    model.predict(X_te_n), 
    columns=target_cols, 
    index=y_te_n.index
)
y_pred = denormalize_with_independent_scalers(y_pred_norm, sc_y)

# ========== MÉTRICAS ==========
from sklearn.metrics import mean_squared_error, r2_score

print("\n" + "="*60)
print("RESULTADOS")
print("="*60)
for target in target_cols:
    rmse = np.sqrt(mean_squared_error(y_test[target], y_pred[target]))
    r2 = r2_score(y_test[target], y_pred[target])
    print(f"{target:15s}: RMSE={rmse:8.2f} kNm  |  R²={r2:7.4f}")
print("="*60)
print("\n✅ ¡Entrenamiento completado!")
```

Ejecutar:
```bash
python scripts/03_train_quick.py
```

**✅ Output:** Modelo entrenado en `models/` + métricas en consola

---

## 📊 Ver Resultados

Los archivos generados:
```
project/
├── data/
│   ├── raw/              # CSVs crudos desde Bladed
│   ├── processed/        # CSVs con features
│   └── complete.csv      # Dataset unificado
├── models/
│   ├── xgb_model.pkl     # Modelo entrenado
│   └── scalers/          # Scalers para normalización
└── reports/              # Gráficas y reportes
```

---

## 🔍 Resumen de Comandos

```bash
# 1. Instalación
pip install -r requirements.txt && pip install -e .

# 2. Generar datos
python scripts/01_make_timeseries.py
python scripts/02_make_features.py

# 3. Entrenar
python scripts/03_train_quick.py

# 4. (Opcional) Comparar modelos
python scripts/04_compare_models.py
```

---

## 📚 Más Información

Para documentación completa, ver:
- **USAGE_GUIDE.md** - Guía detallada con todos los módulos
- **README.md** - Arquitectura y descripción general
- **configs/** - Configuraciones disponibles

---

## ❓ Ayuda Rápida

**Error de import?**
```bash
pip install -e .
```

**Falta memoria?**
```python
assemble_csvs(..., batch_size=5, optimize_memory=True)
```

**¿Qué features usar?**
Ver `configs/features.yaml` y activar/desactivar según necesites.

---

**¡Listo para empezar! 🎉**
