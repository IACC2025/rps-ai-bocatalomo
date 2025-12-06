# 📚 GUÍA COMPLETA DEL CÓDIGO - modelo.py

## 🎯 Objetivo General del Código

Este archivo implementa un **sistema de Inteligencia Artificial** que aprende a predecir las jugadas de un oponente en Piedra, Papel o Tijera, utilizando **Machine Learning**.

---

## 📦 1. IMPORTACIONES Y CONFIGURACIÓN

### Librerías Importadas

```python
import os
import pickle
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
```

**¿Para qué sirve cada una?**

| Librería | Uso |
|----------|-----|
| `os` | Crear carpetas (models/) |
| `pickle` | Guardar/cargar el modelo entrenado |
| `warnings` | Silenciar mensajes de advertencia |
| `Path` | Manejar rutas de archivos de forma segura |
| `pandas` | Manipular datos (DataFrames) |
| `numpy` | Operaciones matemáticas y arrays |

### Librerías de Machine Learning

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
```

**¿Para qué?**

- **train_test_split**: Divide datos en entrenamiento (80%) y prueba (20%)
- **accuracy_score**: Calcula el % de aciertos del modelo
- **KNeighborsClassifier**: Modelo KNN (vecinos más cercanos)
- **RandomForestClassifier**: Modelo de bosques aleatorios
- **GradientBoostingClassifier**: Modelo de boosting
- **compute_class_weight**: Balancea clases desbalanceadas

### Configuración de Rutas

```python
RUTA_PROYECTO = Path(__file__).parent.parent
RUTA_DATOS = RUTA_PROYECTO / "data" / "resultados_juego.csv"
RUTA_MODELO = RUTA_PROYECTO / "models" / "modelo_entrenado.pkl"
```

**Explicación:**
- `__file__`: Ubicación del archivo actual (modelo.py)
- `.parent.parent`: Sube 2 niveles (de src/ a rps-ai-bocatalomo/)
- Construye rutas a: `data/resultados_juego.csv` y `models/modelo_entrenado.pkl`

### Diccionarios de Mapeo

```python
JUGADA_A_NUM = {"piedra": 0, "papel": 1, "tijera": 2}
NUM_A_JUGADA = {0: "piedra", 1: "papel", 2: "tijera"}
GANA_A = {"piedra": "tijera", "papel": "piedra", "tijera": "papel"}
PIERDE_CONTRA = {"piedra": "papel", "papel": "tijera", "tijera": "piedra"}
```

**¿Por qué?**

Los modelos de ML solo entienden **números**, no texto. Necesitamos:
- **JUGADA_A_NUM**: Convertir "piedra" → 0, "papel" → 1, "tijera" → 2
- **NUM_A_JUGADA**: Convertir de vuelta 0 → "piedra"
- **GANA_A**: Saber qué jugada le gana a cuál
- **PIERDE_CONTRA**: Saber qué jugada pierde contra cuál

---

## 🗂️ 2. CARGA Y PREPARACIÓN DE DATOS

### Función: `cargar_datos()`

```python
def cargar_datos(ruta_csv: str = None) -> pd.DataFrame:
    """Carga y renombra columnas del CSV."""
```

**¿Qué hace?**

1. Lee el archivo CSV con pandas
2. Renombra las columnas a nombres estándar
3. Si el CSV solo tiene 3 columnas, añade las que faltan

**Ejemplo:**

```python
# Entrada: CSV con columnas desconocidas
# 1,piedra,papel,Jugador 2,0.5,0.6

# Salida: DataFrame con columnas estándar
# numero_ronda | jugada_j1 | jugada_j2 | ganador | tiempo_j1 | tiempo_j2
# 1            | piedra    | papel     | J2      | 0.5       | 0.6
```

**Código clave:**

```python
if len(df.columns) == 3:
    # CSV mínimo: solo tiene ronda, j1, j2
    df.columns = NOMBRES[:3]
    df['tiempo_j1'] = 0.5  # Añadir columnas que faltan
    df['tiempo_j2'] = 0.5
```

---

### Función: `preparar_datos()`

```python
def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara datos: convierte jugadas a números y crea target."""
```

**¿Qué hace? (Paso a paso)**

#### Paso 1: Convertir jugadas a números

```python
df['jugada_j1_num'] = df['jugada_j1'].map(JUGADA_A_NUM)
df['jugada_j2_num'] = df['jugada_j2'].map(JUGADA_A_NUM)
```

**Antes:**
```
jugada_j1: piedra, papel, tijera
```

**Después:**
```
jugada_j1_num: 0, 1, 2
```

#### Paso 2: Crear el TARGET (objetivo a predecir)

```python
df['proxima_jugada_j2'] = df['jugada_j2_num'].shift(-1)
```

**¿Qué hace `shift(-1)`?**

Desplaza los valores hacia **arriba**, así cada fila tiene la jugada **siguiente**:

```
Ronda | jugada_j2 | proxima_jugada_j2
  1   | piedra    | papel            ← Shift trajo el valor de la ronda 2
  2   | papel     | tijera           ← Shift trajo el valor de la ronda 3
  3   | tijera    | NaN              ← No hay ronda 4
```

**¿Por qué es importante?**

Esto es el **corazón del modelo**: Queremos predecir **"¿qué jugará el oponente EN LA PRÓXIMA RONDA?"**

#### Paso 3: Calcular resultado de cada ronda

```python
def calcular_resultado(row):
    j1, j2 = row['jugada_j1'], row['jugada_j2']
    if j1 == j2: return 0        # Empate
    elif GANA_A.get(j1) == j2: return 1   # Gana J1
    else: return -1                        # Pierde J1

df['resultado'] = df.apply(calcular_resultado, axis=1)
```

**Resultado:**
- `1` = J1 ganó
- `0` = Empate
- `-1` = J1 perdió

---

## ⚙️ 3. FEATURE ENGINEERING (Lo Más Importante)

### Función: `crear_features()`

```python
def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features sin introducir patrones cíclicos."""
```

**¿Qué son las "features"?**

Son **características** que ayudan al modelo a predecir. Cuantas mejores features, mejor predicción.

### Feature 1: Frecuencias Acumulativas

```python
df['freq_j2_piedra'] = (df['jugada_j2_num'] == 0).expanding().mean()
df['freq_j2_papel'] = (df['jugada_j2_num'] == 1).expanding().mean()
df['freq_j2_tijera'] = (df['jugada_j2_num'] == 2).expanding().mean()
```

**¿Qué hace `.expanding().mean()`?**

Calcula el **promedio acumulativo**:

```
Ronda | jugada_j2 | freq_j2_piedra
  1   | piedra    | 1.00 (100% ha sido piedra hasta ahora)
  2   | papel     | 0.50 (50% piedra de 2 rondas)
  3   | piedra    | 0.67 (67% piedra de 3 rondas)
  4   | tijera    | 0.50 (50% piedra de 4 rondas)
```

**¿Por qué es útil?**

Si alguien juega piedra el 60% del tiempo, **probablemente seguirá haciéndolo**.

---

### Feature 2: Lag Features (Memoria)

```python
df['jugada_j2_lag1'] = df['jugada_j2_num'].shift(1)
df['jugada_j2_lag2'] = df['jugada_j2_num'].shift(2)
df['jugada_j2_lag3'] = df['jugada_j2_num'].shift(3)
```

**¿Qué hace `shift(1)`?**

Trae el valor de la fila **anterior**:

```
Ronda | jugada_j2 | lag1  | lag2  | lag3
  4   | tijera    | papel | piedra| papel
             ↑        ↑       ↑       ↑
           actual   ronda3  ronda2  ronda1
```

**¿Por qué es útil?**

Detecta patrones como: **"Siempre juega tijera después de papel"**

---

### Feature 3: Resultado Anterior

```python
df['resultado_anterior'] = df['resultado'].shift(1)
```

**¿Para qué?**

Detecta si el oponente **reacciona** a ganar o perder:

```
Ronda | resultado_anterior | jugada_j2
  2   | -1 (perdió)        | papel     ← ¿Cambia después de perder?
  3   | 1  (ganó)          | papel     ← ¿Repite cuando gana?
```

---

### Feature 4: Racha

```python
def calcular_racha(resultados):
    racha = 0
    for r in resultados:
        if r == 1: racha = racha + 1 if racha >= 0 else 1
        elif r == -1: racha = racha - 1 if racha <= 0 else -1
        else: racha = 0
    return racha

df['racha'] = df['resultado'].expanding().apply(calcular_racha, raw=False)
```

**¿Qué hace?**

Cuenta victorias/derrotas **consecutivas**:

```
Resultados:   1,  1, -1, -1, -1,  0,  1
Racha:        1,  2, -1, -2, -3,  0,  1
              ↑   ↑   ↑   ↑   ↑   ↑   ↑
            +1  +2  -1  -2  -3  reset +1
```

**¿Por qué es útil?**

Detecta si el oponente cambia estrategia tras una racha de derrotas.

---

### Feature 5: Patrones de Cambio

```python
df['cambio_j2'] = (df['jugada_j2_num'] != df['jugada_j2_lag1']).astype(int)
df['cambio_tras_perder'] = ((df['resultado_anterior'] == -1) & (df['cambio_j2'] == 1)).astype(int)
```

**¿Qué detecta?**

- **cambio_j2**: ¿Cambió su jugada? (1=sí, 0=no)
- **cambio_tras_perder**: ¿Cambió DESPUÉS de perder?

**Ejemplo:**

```
Ronda | jugada_j2 | resultado_anterior | cambio_j2 | cambio_tras_perder
  2   | papel     | -1 (perdió)        | 1 (cambió)| 1 (SÍ)
  3   | papel     | 1  (ganó)          | 0 (repite)| 0 (NO)
```

---

### Feature 6: Fase del Juego

```python
df['fase_juego'] = pd.cut(df['numero_ronda'], bins=3, labels=[0, 1, 2])
```

**¿Qué hace `pd.cut()`?**

Divide las rondas en 3 grupos:

```
Rondas 1-5:   fase_juego = 0 (inicio)
Rondas 6-10:  fase_juego = 1 (medio)
Rondas 11-15: fase_juego = 2 (final)
```

**¿Por qué es útil?**

La gente juega diferente al principio (explorando) vs al final (patrones establecidos).

---

### Feature 7: Tendencias Recientes

```python
df['freq_j2_piedra_reciente'] = (df['jugada_j2_num'] == 0).rolling(5, min_periods=1).mean()
```

**¿Qué hace `.rolling(5)`?**

Calcula el promedio de las **últimas 5 rondas** (ventana móvil):

```
Rondas:     P  P  T  P  P  P  P
Ventana:   [P  P  T  P  P]
Promedio:   80% piedra en últimas 5

Siguiente:    [P  T  P  P  P]
Promedio:      80% piedra
```

**¿Por qué es útil?**

Detecta **cambios de estrategia**: "Antes jugaba tijera, ahora juega papel"

---

### Feature 8: Análisis de Tiempos

```python
df['tiempo_j2_promedio'] = df['tiempo_j2'].expanding().mean()
df['tiempo_j2_relativo'] = df['tiempo_j2'] - df['tiempo_j2_promedio']
df['tiempo_j2_rapido'] = (df['tiempo_j2'] < 0.5).astype(int)
```

**¿Qué detecta?**

- **tiempo_j2_promedio**: Velocidad promedio del oponente
- **tiempo_j2_relativo**: ¿Jugó más rápido o lento que su promedio?
- **tiempo_j2_rapido**: ¿Jugó en menos de 0.5 segundos?

**¿Por qué es útil?**

Las **jugadas rápidas son instintivas** y más predecibles. Si alguien juega rápido, probablemente use su jugada "por defecto".

---

### Función: `seleccionar_features()`

```python
def seleccionar_features(df: pd.DataFrame) -> tuple:
    """Selecciona features para el modelo."""
    feature_cols = [
        'jugada_j2_lag1', 'jugada_j2_lag2', 'jugada_j2_lag3',
        'freq_j2_piedra', 'freq_j2_papel', 'freq_j2_tijera',
        # ... (21 features en total)
    ]
    
    X = df_clean[feature_cols]  # Features (entrada)
    y = df_clean['proxima_jugada_j2']  # Target (salida)
    
    return X, y
```

**¿Qué hace?**

Separa los datos en:
- **X** (features): Las 21 características que el modelo usará para aprender
- **y** (target): Lo que queremos predecir (próxima jugada)

---

## 🎓 4. ENTRENAMIENTO DEL MODELO

### Función: `entrenar_modelo()`

```python
def entrenar_modelo(X, y, test_size: float = 0.2):
    """Entrena y selecciona el mejor modelo."""
```

#### Paso 1: Dividir Datos

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)
```

**¿Qué hace?**

Divide los datos:
- **80% Entrenamiento**: Para que el modelo aprenda
- **20% Prueba**: Para evaluar qué tan bien aprendió

**shuffle=False**: No mezcla (mantiene orden temporal)

```
Datos totales: 100 rondas
├─ Train: Rondas 1-80  (aprender)
└─ Test:  Rondas 81-100 (evaluar)
```

---

#### Paso 2: Balancear Clases

```python
clases = np.unique(y_train)
pesos = compute_class_weight(class_weight='balanced', classes=clases, y=y_train)
pesos_dict = dict(zip(clases, pesos))
```

**¿Por qué?**

Si tienes datos desbalanceados:

```
Piedra: 60 veces
Papel: 30 veces
Tijera: 10 veces ← El modelo ignoraría tijera
```

**Los pesos corrigen esto:**

```
Peso Piedra: 0.5  (baja importancia)
Peso Papel:  1.0  (normal)
Peso Tijera: 3.0  (alta importancia)
```

---

#### Paso 3: Entrenar Múltiples Modelos

```python
modelos = {
    'Random Forest': RandomForestClassifier(...),
    'Gradient Boosting': GradientBoostingClassifier(...),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
}
```

**¿Por qué 3 modelos?**

Cada modelo tiene **fortalezas diferentes**:

| Modelo | Bueno para |
|--------|-----------|
| **Random Forest** | Patrones complejos, robusto |
| **Gradient Boosting** | Accuracy alta, aprende errores |
| **KNN** | Patrones simples, similar a casos anteriores |

---

#### Paso 4: Evaluar y Seleccionar el Mejor

```python
for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)  # Entrenar
    y_pred = modelo.predict(X_test)  # Predecir
    acc = accuracy_score(y_test, y_pred)  # Evaluar
    
    if acc > mejor_accuracy:
        mejor_modelo = modelo  # Guardar el mejor
```

**Salida:**

```
📊 Evaluando modelos...
  Random Forest: 52.30%
  Gradient Boosting: 48.70%
  KNN (k=5): 46.20%

🏆 Mejor: Random Forest (52.30%)
```

---

#### Paso 5: Reentrenar con Todos los Datos

```python
mejor_modelo.fit(X, y)  # Usar TODOS los datos (100%)
```

**¿Por qué?**

Ahora que sabemos que Random Forest es el mejor, lo entrenamos con **todos los datos** (no solo el 80%) para que aprenda más.

---

### Funciones: `guardar_modelo()` y `cargar_modelo()`

```python
def guardar_modelo(modelo, ruta=None):
    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)

def cargar_modelo(ruta=None):
    with open(ruta, "rb") as f:
        return pickle.load(f)
```

**¿Qué hace pickle?**

Guarda el modelo entrenado en un archivo `.pkl` para usarlo después sin tener que reentrenar.

---

## 🤖 5. CLASE JUGADOR IA (Lo Más Complejo)

### Inicialización

```python
class JugadorIA:
    def __init__(self, ruta_modelo: str = None):
        self.modelo = None
        self.historial = []
        self.feature_cols = [...]  # Lista de 21 features
        
        self.modelo = cargar_modelo(ruta_modelo)
```

**¿Qué guarda?**

- **modelo**: El modelo entrenado (Random Forest, etc.)
- **historial**: Lista de todas las rondas jugadas
- **feature_cols**: Nombres de las 21 features (deben coincidir con entrenamiento)

---

### Método: `registrar_ronda()`

```python
def registrar_ronda(self, jugada_j1: str, jugada_j2: str, 
                    tiempo_j1: float = 0, tiempo_j2: float = 0):
    self.historial.append((jugada_j1, jugada_j2, tiempo_j1, tiempo_j2))
```

**¿Qué hace?**

Añade cada ronda jugada al historial:

```python
historial = [
    ('piedra', 'papel', 0.5, 0.6),
    ('tijera', 'piedra', 0.8, 0.4),
    ('papel', 'tijera', 0.3, 0.7),
]
```

---

### Método: `obtener_features_actuales()` ⭐

```python
def obtener_features_actuales(self) -> np.ndarray:
    """Genera features del historial actual."""
    df_hist = pd.DataFrame(self.historial, ...)
    df = preparar_datos(df_hist)
    df = crear_features(df)
    
    ultima_fila = df.iloc[-1]
    features = ultima_fila[self.feature_cols].values
    return features
```

**¿Qué hace? (Paso a paso)**

1. Convierte `historial` en DataFrame
2. Llama a `preparar_datos()` (convierte a números)
3. Llama a `crear_features()` (calcula las 21 features)
4. Toma la **última fila** (estado actual)
5. Extrae solo las 21 features que el modelo necesita

**Ejemplo:**

```python
Historial: 3 rondas jugadas
→ Convierte a DataFrame
→ Crea features (freq_piedra=0.66, lag1=1, ...)
→ Última fila: [0.66, 1, 0, 0.33, ...] ← 21 números
→ Estos 21 números van al modelo para predecir
```

---

### Método: `es_jugador_aleatorio()` 🎲

```python
def es_jugador_aleatorio(self) -> bool:
    """Detecta si el oponente juega aleatorio."""
```

**3 Criterios:**

1. **Frecuencias equilibradas**: ~33% cada jugada
2. **Cambios frecuentes**: >75% tasa de cambio
3. **Sin patrón reciente**: Ninguna jugada >50% en últimas 5

**Si cumple 2 de 3 → Jugador ALEATORIO**

---

### Método: `predecir_jugada_oponente()` 🧠 (EL MÁS IMPORTANTE)

```python
def predecir_jugada_oponente(self) -> str:
    """Predice la próxima jugada SIN crear bucles cíclicos."""
```

#### **Flujo de Decisión:**

```
1. ¿Hay modelo? NO → jugar aleatorio
                ↓ SÍ
2. ¿IA jugó lo mismo 5 veces? SÍ → CAMBIAR (anti-bucle)
                              ↓ NO
3. ¿Oponente es aleatorio? SÍ → Estrategia anti-aleatorio
                           ↓ NO
4. ¿Hay patrón MUY claro (>65%)? SÍ → Usar patrón
                                 ↓ NO
5. ¿Hay patrón claro (>55%)? SÍ (70%) → Usar patrón
                              ↓ NO (30%)
6. Usar predicción del modelo
```

#### **Detector Anti-Bucle** 🚨

```python
if len(set(ultimas_5_ia)) == 1:  # Si las 5 son iguales
    print("🚨 ANTI-BUCLE")
    opciones = [j for j in ["piedra", "papel", "tijera"] if j != repetida]
    return np.random.choice(opciones)
```

**¿Qué previene?**

```
❌ ANTES (sin anti-bucle):
IA: Piedra, Piedra, Piedra, Piedra, Piedra... (infinito)

✅ AHORA (con anti-bucle):
IA: Piedra, Piedra, Piedra, Piedra, Piedra, Papel ← CAMBIA
```

---

### Método: `decidir_jugada()` 🎯

```python
def decidir_jugada(self) -> str:
    prediccion_oponente = self.predecir_jugada_oponente()
    return PIERDE_CONTRA[prediccion_oponente]
```

**¿Qué hace?**

1. Predice qué jugará el oponente
2. Devuelve la jugada que **le gana**

**Ejemplo:**

```python
prediccion = "tijera"  ← IA predice que jugarás tijera
return PIERDE_CONTRA["tijera"]  = "piedra"
→ IA juega PIEDRA (gana a tijera)
```

---

## 🏁 6. FUNCIÓN MAIN (Flujo Completo)

```python
def main():
    df = cargar_datos()           # 1. Cargar CSV
    df = preparar_datos(df)       # 2. Convertir a números
    df = crear_features(df)       # 3. Crear 21 features
    X, y = seleccionar_features(df)  # 4. Separar X e y
    modelo = entrenar_modelo(X, y)   # 5. Entrenar modelos
    guardar_modelo(modelo)        # 6. Guardar el mejor
```

---

## 📊 RESUMEN: Flujo Completo de Uso

### Entrenamiento (una vez)

```
CSV (150 rondas)
    ↓ cargar_datos()
DataFrame con columnas estándar
    ↓ preparar_datos()
Jugadas convertidas a números + target creado
    ↓ crear_features()
21 features calculadas
    ↓ seleccionar_features()
X (21 features), y (target)
    ↓ entrenar_modelo()
3 modelos entrenados → Mejor seleccionado
    ↓ guardar_modelo()
modelo_entrenado.pkl (guardado)
```

### Uso en Juego (cada ronda)

```
Ronda 1-3: IA juega aleatorio (no hay historial)

Ronda 4+:
    Tu jugada anterior registrada en historial
        ↓ obtener_features_actuales()
    21 features calculadas del historial actual
        ↓ predecir_jugada_oponente()
    Modelo predice: "Jugará TIJERA"
        ↓ decidir_jugada()
    IA decide: "Jugaré PIEDRA" (gana a tijera)
        ↓
    Ronda se juega
        ↓ registrar_ronda()
    Se añade al historial
        ↓
    Volver a Ronda siguiente
```

---

## 🎯 Conceptos Clave Para Entender

1. **Target (y)**: Lo que queremos predecir = próxima jugada
2. **Features (X)**: Características que ayudan a predecir (21 en total)
3. **Train/Test Split**: 80% aprende, 20% evalúa
4. **Expanding**: Promedio acumulativo (toda la historia)
5. **Rolling**: Promedio de ventana móvil (últimas N rondas)
6. **Shift**: Trae valores de filas anteriores/siguientes
7. **Anti-Bucle**: Evita que la IA se quede atascada

---

## 💡 ¿Por Qué Funciona?

1. **Muchas features (21)**: El modelo ve muchos patrones
2. **Datos históricos**: Aprende de 150+ rondas previas
3. **Detección de patrones**: Frecuencias, lag, rachas
4. **Anti-bucle**: No se queda atascado
5. **Detección de aleatoridad**: Cambia estrategia si no hay patrón
6. **Múltiples modelos**: Elige el que mejor funciona

**Resultado:** 50-70% winrate contra humanos 🎯