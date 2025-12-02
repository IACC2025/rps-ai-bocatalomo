"""
RPSAI - Modelo de IA para Piedra, Papel o Tijera
=================================================
Implementación final con alta explotación (95%) y manejo robusto de datos.
"""

import os
import pickle
import warnings
import time
from pathlib import Path

import pandas as pd
import numpy as np

# Ignorar advertencias de Scikit-learn sobre feature names y precisión
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

# --- CONFIGURACIÓN DE RUTAS Y MAPEOS ---
RUTA_PROYECTO = Path(__file__).parent.parent
RUTA_DATOS = RUTA_PROYECTO / "data" / "resultados_juego.csv"
RUTA_MODELO = RUTA_PROYECTO / "models" / "modelo_entrenado.pkl"

# Mapeo de jugadas
JUGADA_A_NUM = {"piedra": 0, "papel": 1, "tijera": 2}
NUM_A_JUGADA = {0: "piedra", 1: "papel", 2: "tijera"}
GANA_A = {"piedra": "tijera", "papel": "piedra", "tijera": "papel"}
PIERDE_CONTRA = {"piedra": "papel", "papel": "tijera", "tijera": "piedra"}


# =============================================================================
# PARTE 1: EXTRACCIÓN Y PREPARACIÓN DE DATOS
# =============================================================================

def cargar_datos(ruta_csv: str = None) -> pd.DataFrame:
    """Carga los datos del CSV de partidas y renombra las columnas."""
    if ruta_csv is None:
        ruta_csv = RUTA_DATOS

    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"❌ No se encontró el archivo: {ruta_csv}")

    df = pd.read_csv(ruta_csv)

    NOMBRES_ESPERADOS = [
        'numero_ronda', 'jugada_j1', 'jugada_j2',
        'ganador', 'tiempo_j1', 'tiempo_j2'
    ]

    num_columnas = len(df.columns)

    if num_columnas == 3:
        # Caso 1: CSV solo tiene las jugadas (asumimos el orden correcto)
        df.columns = NOMBRES_ESPERADOS[:3]
        df['ganador'] = None
        df['tiempo_j1'] = 0.5
        df['tiempo_j2'] = 0.5
    elif num_columnas >= 6:
        # Caso 2: CSV completo (o más)
        df.columns = NOMBRES_ESPERADOS
    else:
        # Caso 3: Número inesperado
        raise ValueError(
            f"❌ ERROR: El archivo CSV tiene {num_columnas} columnas. Se esperaban 3 o 6."
        )

    print(f"✓ Datos cargados: {len(df)} rondas")
    return df


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara los datos, crea el TARGET y la columna de resultado."""
    df = df.copy()

    # 1. Convertir jugadas a números
    df['jugada_j1_num'] = df['jugada_j1'].map(JUGADA_A_NUM)
    df['jugada_j2_num'] = df['jugada_j2'].map(JUGADA_A_NUM)

    # 2. Crear TARGET: la próxima jugada del oponente
    df['proxima_jugada_j2'] = df['jugada_j2_num'].shift(-1)

    # 3. Calcular resultado de cada ronda (1: Gana J1, -1: Pierde J1, 0: Empate)
    def calcular_resultado(row):
        j1, j2 = row['jugada_j1'], row['jugada_j2']
        if j1 == j2: return 0
        elif GANA_A.get(j1) == j2: return 1
        else: return -1

    df.dropna(subset=['jugada_j1', 'jugada_j2'], inplace=True)
    df['resultado'] = df.apply(calcular_resultado, axis=1)

    # 4. Eliminar filas con NaN (la última fila no tiene próxima jugada)
    df = df.dropna(subset=['proxima_jugada_j2'])
    df['proxima_jugada_j2'] = df['proxima_jugada_j2'].astype(int)

    # 5. Rellenar NaN en tiempos
    df['tiempo_j1'] = df['tiempo_j1'].fillna(0.5)
    df['tiempo_j2'] = df['tiempo_j2'].fillna(0.5)

    return df


# =============================================================================
# PARTE 2: FEATURE ENGINEERING
# =============================================================================

def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features (características predictivas) para el modelo."""
    df = df.copy()

    # Rellenar NAN iniciales en columnas clave
    df.fillna({'jugada_j2_num': 0, 'jugada_j1_num': 0, 'resultado': 0, 'tiempo_j2': 0.5, 'tiempo_j1': 0.5},
              inplace=True)

    # --- Lag, Frecuencias, Resultados, Patrones, Tiempos ---

    # 1. Frecuencias acumulativas
    df['freq_j2_piedra'] = (df['jugada_j2_num'] == 0).expanding().mean()
    df['freq_j2_papel'] = (df['jugada_j2_num'] == 1).expanding().mean()
    df['freq_j2_tijera'] = (df['jugada_j2_num'] == 2).expanding().mean()

    # 2. Lag features (historial)
    df['jugada_j2_lag1'] = df['jugada_j2_num'].shift(1)
    df['jugada_j2_lag2'] = df['jugada_j2_num'].shift(2)
    df['jugada_j2_lag3'] = df['jugada_j2_num'].shift(3)
    df['jugada_j1_lag1'] = df['jugada_j1_num'].shift(1)

    # 3. Resultado anterior
    df['resultado_anterior'] = df['resultado'].shift(1)

    # 🌟 FEATURE: Doble Derrota (Señal de alarma)
    df['doble_derrota'] = ((df['resultado'].shift(1) == -1) & (df['resultado'].shift(2) == -1)).astype(int)

    # 4. Racha actual
    def calcular_racha(resultados):
        racha = 0
        for r in resultados:
            if r == 1: racha = racha + 1 if racha >= 0 else 1
            elif r == -1: racha = racha - 1 if racha <= 0 else -1
            else: racha = 0
        return racha
    df['racha'] = df['resultado'].expanding().apply(calcular_racha, raw=False).fillna(0)

    # 5. Patrón de cambio
    df['cambio_j2'] = (df['jugada_j2_num'] != df['jugada_j2_lag1']).astype(float)
    df['cambio_tras_perder'] = ((df['resultado_anterior'] == -1) & (df['cambio_j2'] == 1)).astype(float)

    # 🌟 FEATURE: Patrón de Transición del Oponente
    df['patron_j2_transicion'] = (df['jugada_j2_lag1'] - df['jugada_j2_lag2']) % 3
    df['patron_j2_transicion'] = df['patron_j2_transicion'].fillna(0).astype(int)

    # 6. Fase del juego
    df['fase_juego'] = pd.cut(df['numero_ronda'], bins=3, labels=[0, 1, 2], include_lowest=True).astype(float)

    # 7. Tendencias recientes (ventana móvil)
    df['freq_j2_piedra_reciente'] = (df['jugada_j2_num'] == 0).rolling(5, min_periods=1).mean()
    df['freq_j2_papel_reciente'] = (df['jugada_j2_num'] == 1).rolling(5, min_periods=1).mean()
    df['freq_j2_tijera_reciente'] = (df['jugada_j2_num'] == 2).rolling(5, min_periods=1).mean()

    # 8. Análisis de tiempos de reacción
    df['tiempo_j2_promedio'] = df['tiempo_j2'].expanding().mean()
    df['tiempo_j2_relativo'] = df['tiempo_j2'] - df['tiempo_j2_promedio']
    df['diff_tiempo'] = df['tiempo_j1'] - df['tiempo_j2']
    df['tiempo_j2_rapido'] = (df['tiempo_j2'] < 0.5).astype(float)
    df['tiempo_j2_tendencia'] = df['tiempo_j2'].rolling(5, min_periods=1).mean()

    # 🟢 FEATURE CORREGIDA/AÑADIDA: Tiempos de J1 (para asegurar 23 features)
    df['tiempo_j1_promedio'] = df['tiempo_j1'].expanding().mean()
    df['tiempo_j1_relativo'] = df['tiempo_j1'] - df['tiempo_j1_promedio']

    # --- CORRECCIÓN FINAL DE TIPOS Y NaN ---
    df.fillna(0, inplace=True)

    velocidad_cortada = pd.cut(df['tiempo_j2'], bins=[0, 0.5, 1.0, 10], labels=[0, 1, 2], include_lowest=True)
    df['velocidad_j2'] = velocidad_cortada.astype(str).map({'0': 0, '1': 1, '2': 2}).fillna(0).astype(int)

    for col in ['cambio_j2', 'cambio_tras_perder', 'tiempo_j2_rapido', 'fase_juego']:
        if col in df.columns: df[col] = df[col].astype(int)

    return df


def seleccionar_features(df: pd.DataFrame) -> tuple:
    """Selecciona las features para entrenar y el target."""
    # 🟢 LISTA FINAL DE 23 FEATURES (Debe coincidir con JugadorIA.__init__)
    feature_cols = [
        'jugada_j2_lag1', 'jugada_j2_lag2', 'jugada_j2_lag3', 'jugada_j1_lag1',
        'doble_derrota', 'patron_j2_transicion',
        'freq_j2_piedra', 'freq_j2_papel', 'freq_j2_tijera',
        'freq_j2_piedra_reciente', 'freq_j2_papel_reciente', 'freq_j2_tijera_reciente',
        'resultado_anterior', 'racha', 'cambio_j2', 'cambio_tras_perder',
        'fase_juego',
        'tiempo_j2_promedio', 'tiempo_j2_relativo', 'diff_tiempo', 'velocidad_j2',
        'tiempo_j2_rapido', 'tiempo_j2_tendencia',
        'tiempo_j1_promedio',
        'tiempo_j1_relativo'
    ]

    df_clean = df.dropna(subset=feature_cols + ['proxima_jugada_j2'])

    X = df_clean[feature_cols].fillna(0)
    y = df_clean['proxima_jugada_j2']

    return X, y


# =============================================================================
# PARTE 3: ENTRENAMIENTO Y EVALUACIÓN
# =============================================================================

def entrenar_modelo(X, y, test_size: float = 0.2):
    """Entrena múltiples modelos, usa pesos de clase y selecciona el mejor."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )

    clases = np.unique(y_train)
    pesos = compute_class_weight(class_weight='balanced', classes=clases, y=y_train)
    pesos_dict = dict(zip(clases, pesos))

    modelos = {
        'Random Forest (Balanced)': RandomForestClassifier(
            n_estimators=150, max_depth=10, random_state=42, class_weight=pesos_dict
        ),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=75, random_state=42),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    }

    mejor_modelo = None
    mejor_accuracy = 0
    mejor_nombre = ""

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred_test = modelo.predict(X_test)
        acc_test = accuracy_score(y_test, y_pred_test)

        if acc_test > mejor_accuracy:
            mejor_accuracy = acc_test
            mejor_modelo = modelo
            mejor_nombre = nombre

    print(f"\n🏆 Mejor Modelo: {mejor_nombre} (Accuracy: {mejor_accuracy:.2%}). Reentrenando...")
    mejor_modelo.fit(X, y)

    return mejor_modelo


def guardar_modelo(modelo, ruta: str = None):
    """Guarda el modelo entrenado."""
    if ruta is None:
        ruta = RUTA_MODELO

    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)
    print(f"\n💾 Modelo guardado en: {ruta}")


def cargar_modelo(ruta: str = None):
    """Carga un modelo previamente entrenado."""
    if ruta is None:
        ruta = RUTA_MODELO

    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el modelo en: {ruta}")

    with open(ruta, "rb") as f:
        return pickle.load(f)


# =============================================================================
# PARTE 4: CLASE JUGADOR IA
# =============================================================================

class JugadorIA:
    """Jugador IA que usa el modelo para predecir y ganar."""

    def __init__(self, ruta_modelo: str = None):
        """Inicializa el jugador IA."""
        self.modelo = None
        self.historial = []
        # 🟢 LISTA DE 23 FEATURES (Sincronizada)
        self.feature_cols = [
            'jugada_j2_lag1', 'jugada_j2_lag2', 'jugada_j2_lag3', 'jugada_j1_lag1',
            'doble_derrota', 'patron_j2_transicion',
            'freq_j2_piedra', 'freq_j2_papel', 'freq_j2_tijera',
            'freq_j2_piedra_reciente', 'freq_j2_papel_reciente', 'freq_j2_tijera_reciente',
            'resultado_anterior', 'racha', 'cambio_j2', 'cambio_tras_perder',
            'fase_juego',
            'tiempo_j2_promedio', 'tiempo_j2_relativo', 'diff_tiempo', 'velocidad_j2',
            'tiempo_j2_rapido', 'tiempo_j2_tendencia',
            'tiempo_j1_promedio',
            'tiempo_j1_relativo'
        ]
        self.EXPLORATION_RATE = 0.05

        try:
            self.modelo = cargar_modelo(ruta_modelo)
            print("✅ Modelo cargado correctamente")
        except FileNotFoundError:
            print("⚠️  Modelo no encontrado. Jugará aleatoriamente.")

    def registrar_ronda(self, jugada_j1: str, jugada_j2: str, tiempo_j1: float = 0, tiempo_j2: float = 0):
        """Registra una ronda jugada."""
        self.historial.append((jugada_j1, jugada_j2, tiempo_j1, tiempo_j2))

    def obtener_features_actuales(self) -> np.ndarray:
        """Genera las features basadas en el historial actual, replicando el proceso."""
        if len(self.historial) < 1: return None

        try:
            df_hist = pd.DataFrame(self.historial, columns=['jugada_j1', 'jugada_j2', 'tiempo_j1', 'tiempo_j2'])
            df_hist['numero_ronda'] = range(1, len(df_hist) + 1)
        except Exception: return None

        df = preparar_datos(df_hist.copy())
        if df.empty: return None

        df = crear_features(df)

        if df.empty or len(df) == 0: return None

        ultima_fila = df.iloc[-1]
        features = ultima_fila[self.feature_cols].values
        features = np.nan_to_num(features, nan=0.0)

        return features

    def determinar_ganador(self, jugada_humano: str, jugada_ia: str) -> str:
        """Determina el resultado desde la perspectiva de la IA: victoria_ia, derrota_ia, empate."""
        if jugada_ia == jugada_humano: return "empate"
        elif GANA_A[jugada_ia] == jugada_humano: return "victoria_ia"
        else: return "derrota_ia"


    def predecir_jugada_oponente(self) -> str:
        """
        Predice la próxima jugada del oponente, priorizando las features (95% explotación).
        """
        if self.modelo is None or len(self.historial) < 3:
            return np.random.choice(["piedra", "papel", "tijera"])

        features = self.obtener_features_actuales()
        if features is None or len(features) != len(self.feature_cols):
            return np.random.choice(["piedra", "papel", "tijera"])

        # --- LÓGICA DE CASTIGO (Romper ciclo de derrotas) ---
        ultima_ronda = self.historial[-1]
        jugada_ia_anterior = ultima_ronda[1]
        jugada_humano_anterior = ultima_ronda[0]

        resultado_ronda_anterior = self.determinar_ganador(jugada_humano_anterior, jugada_ia_anterior)

        # Si la IA perdió, hay un 80% de probabilidad de predecir que el oponente repetirá su jugada.
        if resultado_ronda_anterior == "derrota_ia" and np.random.rand() < 0.80:
            return jugada_humano_anterior # Predice que el humano repetirá su jugada ganadora

        ultima_ronda = self.historial[-1]
        jugada_ia_anterior = ultima_ronda[1]
        jugada_humano_anterior = ultima_ronda[0]

        resultado_ronda_anterior = self.determinar_ganador(jugada_humano_anterior, jugada_ia_anterior)

        # Condición de Reacción Forzada:
        # Si la IA perdió (derrota_ia) O si fue empate (empate), la IA debe cambiar su estrategia.

        if (resultado_ronda_anterior == "derrota_ia" and np.random.rand() < 0.80) or \
                (resultado_ronda_anterior == "empate" and np.random.rand() < 0.70):
            # En caso de Derrota (80%): Predice que el oponente repetirá su jugada ganadora.
            # En caso de Empate (70%): Predice que el oponente repetirá la jugada de empate (para forzar a la IA a cambiar).
            return jugada_humano_anterior  # Predice la jugada anterior del humano

        # --- ESTRATEGIA: PRIORIZAR MODELO (Explotación 95%) ---

        if np.random.rand() < self.EXPLORATION_RATE:
            # Modo Exploración (5% de las veces)
            prediccion_num = np.random.choice([0, 1, 2])
        else:
            # Modo Explotación: Usar el modelo probabilístico (95% de las veces)
            probabilidades = self.modelo.predict_proba([features])[0]
            prediccion_num = np.argmax(probabilidades)

        return NUM_A_JUGADA[int(prediccion_num)]

    def decidir_jugada(self) -> str:
        """Decide qué jugada hacer para GANAR al oponente."""
        prediccion_oponente = self.predecir_jugada_oponente()
        return PIERDE_CONTRA[prediccion_oponente]


# --- FUNCIÓN PRINCIPAL DE ENTRENAMIENTO ---

def main():
    """Flujo completo de entrenamiento."""
    print("="*60)
    print("   RPSAI - Entrenamiento del Modelo")
    print("="*60)

    try:
        print("\n📁 Paso 1: Cargando datos...")
        df = cargar_datos()

        print("\n🔧 Paso 2: Preparando datos...")
        df = preparar_datos(df)

        print("\n⚙️  Paso 3: Creando features...")
        df = crear_features(df)

        print("\n✂️  Paso 4: Seleccionando features...")
        X, y = seleccionar_features(df)

        print("\n🎓 Paso 5: Entrenando modelos...")
        modelo = entrenar_modelo(X, y)

        print("\n💾 Paso 6: Guardando modelo...")
        guardar_modelo(modelo)

        print("\n" + "="*60)
        print("   ✅ ENTRENAMIENTO COMPLETADO CON ÉXITO")
        print("="*60)

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()