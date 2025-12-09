"""
RPSAI - Modelo OPTIMIZADO para mayor winrate
==============================================
Versión optimizada: Features reducidas y más relevantes + nuevas features estratégicas
Incluye: Detección avanzada de jugadores cíclicos - CORREGIDO
"""

import os
import pickle
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

# --- CONFIGURACIÓN ---
RUTA_PROYECTO = Path(__file__).parent.parent
RUTA_DATOS = RUTA_PROYECTO / "data" / "resultados_juego.csv"
RUTA_MODELO = RUTA_PROYECTO / "models" / "modelo_entrenado.pkl"

JUGADA_A_NUM = {"piedra": 0, "papel": 1, "tijera": 2}
NUM_A_JUGADA = {0: "piedra", 1: "papel", 2: "tijera"}
GANA_A = {"piedra": "tijera", "papel": "piedra", "tijera": "papel"}
PIERDE_CONTRA = {"piedra": "papel", "papel": "tijera", "tijera": "piedra"}


# =============================================================================
# CARGA Y PREPARACIÓN DE DATOS
# =============================================================================

def cargar_datos(ruta_csv: str = None) -> pd.DataFrame:
    """Carga y renombra columnas del CSV."""
    if ruta_csv is None:
        ruta_csv = RUTA_DATOS

    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"❌ No se encontró: {ruta_csv}")

    df = pd.read_csv(ruta_csv)

    # Renombrar columnas
    NOMBRES = ['numero_ronda', 'jugada_j1', 'jugada_j2', 'ganador', 'tiempo_j1', 'tiempo_j2']

    if len(df.columns) == 3:
        df.columns = NOMBRES[:3]
        df['ganador'] = None
        df['tiempo_j1'] = 0.5
        df['tiempo_j2'] = 0.5
    elif len(df.columns) >= 6:
        df.columns = NOMBRES[:len(df.columns)]

    print(f"✓ Datos cargados: {len(df)} rondas")
    return df


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara datos: convierte jugadas a números y crea target."""
    df = df.copy()

    # Convertir jugadas a números
    df['jugada_j1_num'] = df['jugada_j1'].map(JUGADA_A_NUM)
    df['jugada_j2_num'] = df['jugada_j2'].map(JUGADA_A_NUM)

    # TARGET: próxima jugada del oponente
    df['proxima_jugada_j2'] = df['jugada_j2_num'].shift(-1)

    # Calcular resultado
    def calcular_resultado(row):
        j1, j2 = row['jugada_j1'], row['jugada_j2']
        if j1 == j2: return 0
        elif GANA_A.get(j1) == j2: return 1
        else: return -1

    df.dropna(subset=['jugada_j1', 'jugada_j2'], inplace=True)
    df['resultado'] = df.apply(calcular_resultado, axis=1)

    # Limpiar
    df = df.dropna(subset=['proxima_jugada_j2'])
    df['proxima_jugada_j2'] = df['proxima_jugada_j2'].astype(int)
    df['tiempo_j1'] = df['tiempo_j1'].fillna(0.5)
    df['tiempo_j2'] = df['tiempo_j2'].fillna(0.5)

    return df


# =============================================================================
# FEATURE ENGINEERING OPTIMIZADO
# =============================================================================

def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features OPTIMIZADAS - Solo las más relevantes + nuevas estratégicas."""
    df = df.copy()

    # Rellenar NaN iniciales
    df.fillna({
        'jugada_j2_num': 0, 'jugada_j1_num': 0,
        'resultado': 0, 'tiempo_j2': 0.5, 'tiempo_j1': 0.5
    }, inplace=True)

    # ========== GRUPO 1: LAGS (Patrones secuenciales) ==========
    df['jugada_j2_lag1'] = df['jugada_j2_num'].shift(1).fillna(0)
    df['jugada_j2_lag2'] = df['jugada_j2_num'].shift(2).fillna(0)
    df['jugada_j2_lag3'] = df['jugada_j2_num'].shift(3).fillna(0)
    df['jugada_j1_lag1'] = df['jugada_j1_num'].shift(1).fillna(0)

    # ========== GRUPO 2: FRECUENCIAS GLOBALES ==========
    df['freq_j2_piedra'] = (df['jugada_j2_num'] == 0).expanding().mean().fillna(0)
    df['freq_j2_papel'] = (df['jugada_j2_num'] == 1).expanding().mean().fillna(0)
    df['freq_j2_tijera'] = (df['jugada_j2_num'] == 2).expanding().mean().fillna(0)

    # ========== GRUPO 3: FRECUENCIAS RECIENTES (ventana 5) ==========
    df['freq_j2_piedra_reciente'] = (df['jugada_j2_num'] == 0).rolling(5, min_periods=1).mean().fillna(0)
    df['freq_j2_papel_reciente'] = (df['jugada_j2_num'] == 1).rolling(5, min_periods=1).mean().fillna(0)
    df['freq_j2_tijera_reciente'] = (df['jugada_j2_num'] == 2).rolling(5, min_periods=1).mean().fillna(0)

    # ========== GRUPO 4: FRECUENCIAS MUY RECIENTES (ventana 3) ==========
    df['freq_j2_piedra_muy_reciente'] = (df['jugada_j2_num'] == 0).rolling(3, min_periods=1).mean().fillna(0)
    df['freq_j2_papel_muy_reciente'] = (df['jugada_j2_num'] == 1).rolling(3, min_periods=1).mean().fillna(0)
    df['freq_j2_tijera_muy_reciente'] = (df['jugada_j2_num'] == 2).rolling(3, min_periods=1).mean().fillna(0)

    # ========== GRUPO 5: RESULTADOS Y RACHAS ==========
    df['resultado_anterior'] = df['resultado'].shift(1).fillna(0)
    df['resultado_lag2'] = df['resultado'].shift(2).fillna(0)

    # Racha optimizada
    def calcular_racha(resultados):
        racha = 0
        for r in resultados:
            if r == 1: racha = racha + 1 if racha >= 0 else 1
            elif r == -1: racha = racha - 1 if racha <= 0 else -1
            else: racha = 0
        return racha

    df['racha'] = df['resultado'].expanding().apply(calcular_racha, raw=False).fillna(0)

    # ========== GRUPO 6: PATRONES DE CAMBIO ==========
    df['cambio_j2'] = (df['jugada_j2_num'] != df['jugada_j2_lag1']).astype(int).fillna(0)

    # Tasa de cambios reciente (ventana 5)
    df['tasa_cambios_reciente'] = df['cambio_j2'].rolling(5, min_periods=1).mean().fillna(0)

    # ========== GRUPO 7: PATRONES CÍCLICOS - MEJORADO ==========
    # Detectar si el oponente sigue un patrón cíclico

    def detectar_ciclo_ascendente(j_actual, j1, j2):
        """Detecta ciclo: 0->1->2 (piedra->papel->tijera)"""
        if pd.isna(j_actual) or pd.isna(j1) or pd.isna(j2):
            return 0
        # Ciclo completo: 0->1->2 o 1->2->0 o 2->0->1
        if (j2 == 0 and j1 == 1 and j_actual == 2) or \
           (j2 == 1 and j1 == 2 and j_actual == 0) or \
           (j2 == 2 and j1 == 0 and j_actual == 1):
            return 1
        return 0

    def detectar_ciclo_descendente(j_actual, j1, j2):
        """Detecta ciclo inverso: 2->1->0 (tijera->papel->piedra)"""
        if pd.isna(j_actual) or pd.isna(j1) or pd.isna(j2):
            return 0
        # Ciclo inverso: 2->1->0 o 1->0->2 o 0->2->1
        if (j2 == 2 and j1 == 1 and j_actual == 0) or \
           (j2 == 1 and j1 == 0 and j_actual == 2) or \
           (j2 == 0 and j1 == 2 and j_actual == 1):
            return 1
        return 0

    # Aplicar detección
    df['ciclo_ascendente'] = df.apply(
        lambda row: detectar_ciclo_ascendente(
            row['jugada_j2_num'], row['jugada_j2_lag1'], row['jugada_j2_lag2']
        ), axis=1
    )

    df['ciclo_descendente'] = df.apply(
        lambda row: detectar_ciclo_descendente(
            row['jugada_j2_num'], row['jugada_j2_lag1'], row['jugada_j2_lag2']
        ), axis=1
    )

    # Cualquier tipo de ciclo
    df['patron_ciclico'] = ((df['ciclo_ascendente'] == 1) | (df['ciclo_descendente'] == 1)).astype(int)

    # Contador de ciclos consecutivos (cuántos ciclos seguidos)
    def contar_ciclos_consecutivos(serie_ciclos):
        """Cuenta cuántos ciclos ha hecho consecutivamente"""
        contador = 0
        for val in reversed(serie_ciclos):
            if val == 1:
                contador += 1
            else:
                break
        return contador

    df['ciclos_consecutivos'] = df['patron_ciclico'].rolling(
        window=10, min_periods=1
    ).apply(lambda x: contar_ciclos_consecutivos(x.values), raw=False).fillna(0)

    # Tasa de ciclos en ventana reciente
    df['tasa_ciclos_reciente'] = df['patron_ciclico'].rolling(6, min_periods=1).mean().fillna(0)

    # Predecir siguiente jugada si está en ciclo
    def predecir_siguiente_en_ciclo(ciclo_asc, ciclo_desc, ultima_jugada):
        """Si está en un ciclo, predice la siguiente jugada del ciclo"""
        if pd.isna(ultima_jugada):
            return -1

        ultima = int(ultima_jugada)

        # Si detectó ciclo ascendente, la siguiente será +1 (mod 3)
        if ciclo_asc == 1:
            return (ultima + 1) % 3

        # Si detectó ciclo descendente, la siguiente será -1 (mod 3)
        if ciclo_desc == 1:
            return (ultima - 1) % 3

        return -1  # No hay ciclo

    df['prediccion_ciclo'] = df.apply(
        lambda row: predecir_siguiente_en_ciclo(
            row['ciclo_ascendente'],
            row['ciclo_descendente'],
            row['jugada_j2_num']
        ), axis=1
    )

    # ========== GRUPO 8: REPETICIONES ==========
    # ¿El oponente repite la misma jugada?
    df['repite_jugada'] = (df['jugada_j2_lag1'] == df['jugada_j2_lag2']).astype(int).fillna(0)

    # ========== GRUPO 9: REACCIÓN A RESULTADOS ==========
    # ¿Cambia después de perder?
    df['cambio_tras_victoria_ia'] = ((df['resultado_anterior'] == 1) & (df['cambio_j2'] == 1)).astype(int)
    df['repite_tras_derrota_ia'] = ((df['resultado_anterior'] == -1) & (df['repite_jugada'] == 1)).astype(int)

    # ========== GRUPO 10: DIVERSIDAD ==========
    # ¿Cuántas jugadas diferentes ha usado en las últimas 5 rondas?
    def calcular_diversidad(serie):
        return len(set(serie)) if len(serie) > 0 else 1

    df['diversidad_reciente'] = df['jugada_j2_num'].rolling(5, min_periods=1).apply(calcular_diversidad, raw=False).fillna(1)

    # ========== GRUPO 11: CONTRA-PREDICCIÓN ==========
    # ¿El oponente juega lo que gana a la jugada anterior de la IA?
    def es_contra_prediccion(jugada_j2, jugada_j1_anterior):
        if pd.isna(jugada_j2) or pd.isna(jugada_j1_anterior):
            return 0
        jugada_j1_ant_str = NUM_A_JUGADA.get(int(jugada_j1_anterior), None)
        jugada_j2_str = NUM_A_JUGADA.get(int(jugada_j2), None)
        if jugada_j1_ant_str and jugada_j2_str:
            return 1 if jugada_j2_str == PIERDE_CONTRA.get(jugada_j1_ant_str) else 0
        return 0

    df['es_contra_prediccion'] = df.apply(
        lambda row: es_contra_prediccion(row['jugada_j2_num'], row['jugada_j1_lag1']),
        axis=1
    )

    # Tasa de contra-predicción reciente
    df['tasa_contra_prediccion'] = df['es_contra_prediccion'].rolling(5, min_periods=1).mean().fillna(0)

    # Rellenar cualquier NaN restante
    df.fillna(0, inplace=True)

    return df


def seleccionar_features(df: pd.DataFrame) -> tuple:
    """Selecciona features OPTIMIZADAS para el modelo."""
    feature_cols = [
        # Lags (patrones secuenciales)
        'jugada_j2_lag1', 'jugada_j2_lag2', 'jugada_j2_lag3', 'jugada_j1_lag1',

        # Frecuencias globales
        'freq_j2_piedra', 'freq_j2_papel', 'freq_j2_tijera',

        # Frecuencias recientes
        'freq_j2_piedra_reciente', 'freq_j2_papel_reciente', 'freq_j2_tijera_reciente',

        # Frecuencias muy recientes
        'freq_j2_piedra_muy_reciente', 'freq_j2_papel_muy_reciente', 'freq_j2_tijera_muy_reciente',

        # Resultados
        'resultado_anterior', 'resultado_lag2', 'racha',

        # Patrones de cambio
        'cambio_j2', 'tasa_cambios_reciente',

        # Patrones cíclicos MEJORADOS
        'patron_ciclico', 'ciclo_ascendente', 'ciclo_descendente',
        'ciclos_consecutivos', 'tasa_ciclos_reciente', 'prediccion_ciclo',

        # Repeticiones
        'repite_jugada',

        # Reacciones
        'cambio_tras_victoria_ia', 'repite_tras_derrota_ia',

        # Diversidad
        'diversidad_reciente',

        # Contra-predicción
        'es_contra_prediccion', 'tasa_contra_prediccion'
    ]

    df_clean = df.dropna(subset=feature_cols + ['proxima_jugada_j2'])
    X = df_clean[feature_cols].fillna(0)
    y = df_clean['proxima_jugada_j2']

    print(f"✓ Features OPTIMIZADAS: {len(feature_cols)}")
    print(f"✓ Muestras: {len(X)}")

    return X, y


# =============================================================================
# ENTRENAMIENTO OPTIMIZADO
# =============================================================================

def entrenar_modelo(X, y, test_size: float = 0.2):
    """Entrena y selecciona el mejor modelo con hiperparámetros optimizados."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )

    clases = np.unique(y_train)
    pesos = compute_class_weight(class_weight='balanced', classes=clases, y=y_train)
    pesos_dict = dict(zip(clases, pesos))

    modelos = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight=pesos_dict
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        ),
        'KNN (k=7)': KNeighborsClassifier(n_neighbors=7),
    }

    mejor_modelo = None
    mejor_accuracy = 0
    mejor_nombre = ""

    print(f"\n📊 Evaluando modelos...")
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"  {nombre}: {acc:.2%}")

        if acc > mejor_accuracy:
            mejor_accuracy = acc
            mejor_modelo = modelo
            mejor_nombre = nombre

    print(f"\n🏆 Mejor: {mejor_nombre} ({mejor_accuracy:.2%})")
    print(f"♻️  Reentrenando con todos los datos...")
    mejor_modelo.fit(X, y)

    return mejor_modelo


def guardar_modelo(modelo, ruta: str = None):
    """Guarda el modelo."""
    if ruta is None:
        ruta = RUTA_MODELO

    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)
    print(f"💾 Guardado en: {ruta}")


def cargar_modelo(ruta: str = None):
    """Carga el modelo."""
    if ruta is None:
        ruta = RUTA_MODELO

    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No encontrado: {ruta}")

    with open(ruta, "rb") as f:
        return pickle.load(f)


# =============================================================================
# JUGADOR IA (OPTIMIZADO Y CORREGIDO)
# =============================================================================

class JugadorIA:
    """IA OPTIMIZADA con mejores features y lógica de decisión refinada."""

    def __init__(self, ruta_modelo: str = None):
        """Inicializa la IA."""
        self.modelo = None
        self.historial = []
        self.feature_cols = [
            'jugada_j2_lag1', 'jugada_j2_lag2', 'jugada_j2_lag3', 'jugada_j1_lag1',
            'freq_j2_piedra', 'freq_j2_papel', 'freq_j2_tijera',
            'freq_j2_piedra_reciente', 'freq_j2_papel_reciente', 'freq_j2_tijera_reciente',
            'freq_j2_piedra_muy_reciente', 'freq_j2_papel_muy_reciente', 'freq_j2_tijera_muy_reciente',
            'resultado_anterior', 'resultado_lag2', 'racha',
            'cambio_j2', 'tasa_cambios_reciente',
            'patron_ciclico', 'ciclo_ascendente', 'ciclo_descendente',
            'ciclos_consecutivos', 'tasa_ciclos_reciente', 'prediccion_ciclo',
            'repite_jugada',
            'cambio_tras_victoria_ia', 'repite_tras_derrota_ia',
            'diversidad_reciente',
            'es_contra_prediccion', 'tasa_contra_prediccion'
        ]

        try:
            self.modelo = cargar_modelo(ruta_modelo)
            print("✅ Modelo cargado")
        except FileNotFoundError:
            print("⚠️  Modelo no encontrado")

    def registrar_ronda(self, jugada_j1: str, jugada_j2: str, tiempo_j1: float = 0, tiempo_j2: float = 0):
        """Registra una ronda."""
        self.historial.append((jugada_j1, jugada_j2, tiempo_j1, tiempo_j2))

    def obtener_features_actuales(self) -> np.ndarray:
        """Genera features del historial actual."""
        if len(self.historial) < 1:
            return None

        try:
            df_hist = pd.DataFrame(self.historial,
                                   columns=['jugada_j1', 'jugada_j2', 'tiempo_j1', 'tiempo_j2'])
            df_hist['numero_ronda'] = range(1, len(df_hist) + 1)

            df = preparar_datos(df_hist.copy())
            if df.empty:
                return None

            df = crear_features(df)
            if df.empty:
                return None

            ultima_fila = df.iloc[-1]
            features = ultima_fila[self.feature_cols].values
            features = np.nan_to_num(features, nan=0.0)

            return features
        except Exception as e:
            print(f"⚠️  Error features: {e}")
            return None

    def obtener_stats_actuales(self) -> dict:
        """Estadísticas del historial."""
        if len(self.historial) == 0:
            return {}

        jugadas_oponente = [j[1] for j in self.historial]  # jugada_j2 (humano/oponente)
        jugadas_ia = [j[0] for j in self.historial]  # jugada_j1 (IA)

        total = len(jugadas_oponente)

        stats = {
            'total_rondas': total,
            'freq_piedra': jugadas_oponente.count('piedra') / total,
            'freq_papel': jugadas_oponente.count('papel') / total,
            'freq_tijera': jugadas_oponente.count('tijera') / total,
            'ultima_jugada': jugadas_oponente[-1] if jugadas_oponente else None,
        }

        if len(self.historial) >= 2:
            cambios = sum(1 for i in range(1, len(jugadas_oponente))
                         if jugadas_oponente[i] != jugadas_oponente[i-1])
            stats['cambios_jugada'] = cambios

        if len(jugadas_oponente) >= 5:
            ultimas_5 = jugadas_oponente[-5:]
            stats['freq_piedra_reciente'] = ultimas_5.count('piedra') / 5
            stats['freq_papel_reciente'] = ultimas_5.count('papel') / 5
            stats['freq_tijera_reciente'] = ultimas_5.count('tijera') / 5

        # Detección de meta-juego
        if len(self.historial) >= 5:
            contador_contra = 0
            contador_contra_reciente = 0

            for i in range(len(self.historial) - 1):
                jugada_ia_actual = jugadas_ia[i]
                jugada_humano_siguiente = jugadas_oponente[i + 1]

                if jugada_humano_siguiente == PIERDE_CONTRA.get(jugada_ia_actual):
                    contador_contra += 1
                    if i >= len(self.historial) - 6:
                        contador_contra_reciente += 1

            stats['tasa_contra_prediccion'] = contador_contra / (len(self.historial) - 1)
            stats['tasa_contra_prediccion_reciente'] = contador_contra_reciente / min(5, len(self.historial) - 1)

        return stats

    def es_jugador_aleatorio(self) -> bool:
        """Detecta si el oponente juega aleatorio."""
        if len(self.historial) < 10:
            return False

        stats = self.obtener_stats_actuales()

        freqs = [stats.get('freq_piedra', 0), stats.get('freq_papel', 0), stats.get('freq_tijera', 0)]
        diferencia = max(freqs) - min(freqs)
        equilibrado = diferencia < 0.17

        tasa_cambio = stats.get('cambios_jugada', 0) / (len(self.historial) - 1)
        cambios_frecuentes = tasa_cambio > 0.75

        sin_patron = False
        if 'freq_piedra_reciente' in stats:
            max_reciente = max(stats.get('freq_piedra_reciente', 0),
                              stats.get('freq_papel_reciente', 0),
                              stats.get('freq_tijera_reciente', 0))
            sin_patron = max_reciente < 0.5

        return sum([equilibrado, cambios_frecuentes, sin_patron]) >= 2

    def predecir_jugada_oponente(self) -> str:
        """Predice la próxima jugada con lógica optimizada - CORREGIDO."""
        if self.modelo is None or len(self.historial) < 3:
            return np.random.choice(["piedra", "papel", "tijera"])

        # DETECTOR ANTI-BUCLE
        if len(self.historial) >= 5:
            ultimas_5_ia = [j[0] for j in self.historial[-5:]]
            if len(set(ultimas_5_ia)) == 1:
                jugada_repetida_ia = ultimas_5_ia[0]
                print(f"ya se tu próxima jugada JIJIJI")
                opciones = [j for j in ["piedra", "papel", "tijera"] if j != jugada_repetida_ia]
                return np.random.choice(opciones)

        # DETECTOR DE JUGADOR CÍCLICO - ✅ CORREGIDO
        if len(self.historial) >= 6:
            features = self.obtener_features_actuales()
            if features is not None and len(features) == len(self.feature_cols):
                try:
                    # Índices de las features cíclicas
                    idx_ciclos_consec = self.feature_cols.index('ciclos_consecutivos')
                    idx_tasa_ciclos = self.feature_cols.index('tasa_ciclos_reciente')
                    idx_pred_ciclo = self.feature_cols.index('prediccion_ciclo')

                    ciclos_consecutivos = features[idx_ciclos_consec]
                    tasa_ciclos = features[idx_tasa_ciclos]
                    pred_ciclo = int(features[idx_pred_ciclo])

                    # Si ha hecho 3+ ciclos consecutivos O tasa > 70%
                    if ciclos_consecutivos >= 3 or tasa_ciclos > 0.7:
                        if pred_ciclo != -1:  # Si hay predicción de ciclo válida
                            jugada_predicha_humano = NUM_A_JUGADA[pred_ciclo]
                            # ✅ CONTRA-JUGAR: Jugar lo que GANA contra la predicción
                            jugada_ia = PIERDE_CONTRA[jugada_predicha_humano]
                            print(f"ya te estoy pillando MUEJEJE")
                            return jugada_ia
                except (ValueError, IndexError):
                    pass  # Si hay error, continuar con la lógica normal

        # DETECTOR DE META-JUEGO (REDUCIDO umbral a 55%)
        if len(self.historial) >= 5:
            stats = self.obtener_stats_actuales()
            tasa_contra = stats.get('tasa_contra_prediccion_reciente', 0)

            # Umbral reducido de 0.6 a 0.55
            if tasa_contra > 0.55:
                ultima_jugada_ia = self.historial[-1][0]
                prediccion_meta = PIERDE_CONTRA[ultima_jugada_ia]

                # Aumentado de 0.7 a 0.75
                if np.random.random() < 0.75:
                    print(f"te voy a ganar MUAJAJA")
                    return prediccion_meta
                else:
                    return np.random.choice(["piedra", "papel", "tijera"])

        # DETECCIÓN DE ALEATORIDAD
        if len(self.historial) >= 10 and self.es_jugador_aleatorio():
            stats = self.obtener_stats_actuales()
            freqs = {
                'piedra': stats.get('freq_piedra', 0),
                'papel': stats.get('freq_papel', 0),
                'tijera': stats.get('freq_tijera', 0)
            }
            jugada_menos_comun = min(freqs, key=freqs.get)

            if np.random.random() < 0.4:
                return jugada_menos_comun
            else:
                return np.random.choice(["piedra", "papel", "tijera"])

        # LÓGICA DE PATRONES (OPTIMIZADA - umbrales más bajos)
        features = self.obtener_features_actuales()
        if features is None or len(features) != len(self.feature_cols):
            return np.random.choice(["piedra", "papel", "tijera"])

        if len(self.historial) >= 6:
            stats = self.obtener_stats_actuales()

            if 'freq_piedra_reciente' in stats:
                freqs_recientes = {
                    'piedra': stats.get('freq_piedra_reciente', 0),
                    'papel': stats.get('freq_papel_reciente', 0),
                    'tijera': stats.get('freq_tijera_reciente', 0)
                }
                jugada_reciente = max(freqs_recientes, key=freqs_recientes.get)
                max_freq_reciente = freqs_recientes[jugada_reciente]

                # Reducido de 0.65 a 0.60
                if max_freq_reciente > 0.60:
                    return jugada_reciente

                # Reducido de 0.55 a 0.50, confianza aumentada a 0.75
                if max_freq_reciente > 0.50 and np.random.random() < 0.75:
                    return jugada_reciente

        # Por defecto: usar modelo ML
        prediccion = self.modelo.predict([features])[0]
        return NUM_A_JUGADA[int(prediccion)]

    def decidir_jugada(self) -> str:
        """Decide qué jugar para ganar."""
        prediccion_oponente = self.predecir_jugada_oponente()

        # Reducido de 0.15 a 0.10 (menos aleatoridad)
        if np.random.random() < 0.10:
            return np.random.choice(["piedra", "papel", "tijera"])

        return PIERDE_CONTRA[prediccion_oponente]


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Entrenamiento completo."""
    print("="*60)
    print("   RPSAI - Entrenamiento del Modelo OPTIMIZADO")
    print("="*60)

    try:
        df = cargar_datos()
        df = preparar_datos(df)
        df = crear_features(df)
        X, y = seleccionar_features(df)
        modelo = entrenar_modelo(X, y)
        guardar_modelo(modelo)

        print("\n✅ COMPLETADO")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()