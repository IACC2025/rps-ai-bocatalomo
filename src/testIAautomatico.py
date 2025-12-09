"""
Script de prueba automatizada para evaluar el winrate de la IA
================================================================
Simula 50 partidas con diferentes estrategias:
- Aleatorio puro
- Patrones cíclicos
- Sesgo hacia una jugada
- Anti-predicción (meta-juego)
- Mixto (cambia de estrategia)
"""

import sys
import random
import numpy as np
from pathlib import Path

# Añadir el directorio src al path para importar el modelo
RUTA_PROYECTO = Path(__file__).parent
sys.path.insert(0, str(RUTA_PROYECTO / "src"))

try:
    from modelo import JugadorIA, GANA_A, PIERDE_CONTRA
except ImportError:
    print("⚠️  No se encontró modelo_corregido.py, intentando con modelo.py")
    from modelo import JugadorIA, GANA_A, PIERDE_CONTRA


# =============================================================================
# ESTRATEGIAS DE JUEGO
# =============================================================================

class EstrategiaJugador:
    """Clase base para estrategias de juego"""

    def __init__(self):
        self.historial_propio = []
        self.historial_oponente = []

    def registrar(self, jugada_propia, jugada_oponente):
        self.historial_propio.append(jugada_propia)
        self.historial_oponente.append(jugada_oponente)

    def decidir_jugada(self) -> str:
        raise NotImplementedError


class Aleatorio(EstrategiaJugador):
    """Juega completamente al azar"""

    def decidir_jugada(self) -> str:
        return random.choice(['piedra', 'papel', 'tijera'])


class CiclicoAscendente(EstrategiaJugador):
    """Sigue el patrón: piedra → papel → tijera → piedra..."""

    def decidir_jugada(self) -> str:
        if not self.historial_propio:
            return 'piedra'

        ultima = self.historial_propio[-1]
        ciclo = {'piedra': 'papel', 'papel': 'tijera', 'tijera': 'piedra'}
        return ciclo[ultima]


class CiclicoDescendente(EstrategiaJugador):
    """Sigue el patrón: tijera → papel → piedra → tijera..."""

    def decidir_jugada(self) -> str:
        if not self.historial_propio:
            return 'tijera'

        ultima = self.historial_propio[-1]
        ciclo = {'tijera': 'papel', 'papel': 'piedra', 'piedra': 'tijera'}
        return ciclo[ultima]


class SesgoJugada(EstrategiaJugador):
    """Tiene preferencia por una jugada (70% de probabilidad)"""

    def __init__(self, jugada_favorita='piedra', probabilidad=0.7):
        super().__init__()
        self.jugada_favorita = jugada_favorita
        self.probabilidad = probabilidad

    def decidir_jugada(self) -> str:
        if random.random() < self.probabilidad:
            return self.jugada_favorita
        else:
            otras = [j for j in ['piedra', 'papel', 'tijera'] if j != self.jugada_favorita]
            return random.choice(otras)


class AntiPrediccion(EstrategiaJugador):
    """Intenta predecir qué jugará la IA y jugar lo que le gana"""

    def decidir_jugada(self) -> str:
        if len(self.historial_oponente) < 2:
            return random.choice(['piedra', 'papel', 'tijera'])

        # Analiza qué jugó la IA últimamente
        ultimas_3 = self.historial_oponente[-3:]

        # Encuentra la más común
        mas_comun = max(set(ultimas_3), key=ultimas_3.count)

        # 60% de las veces, juega lo que gana contra esa
        if random.random() < 0.6:
            return PIERDE_CONTRA[mas_comun]
        else:
            return random.choice(['piedra', 'papel', 'tijera'])


class Repetidor(EstrategiaJugador):
    """Repite la misma jugada si ganó, cambia si perdió"""

    def __init__(self):
        super().__init__()
        self.resultado_anterior = None

    def decidir_jugada(self) -> str:
        if not self.historial_propio:
            return random.choice(['piedra', 'papel', 'tijera'])

        # Si no hay resultado previo, aleatorio
        if self.resultado_anterior is None:
            return random.choice(['piedra', 'papel', 'tijera'])

        # Si ganó, repite
        if self.resultado_anterior == 'gana':
            return self.historial_propio[-1]

        # Si perdió o empató, cambia
        else:
            otras = [j for j in ['piedra', 'papel', 'tijera'] if j != self.historial_propio[-1]]
            return random.choice(otras)

    def registrar(self, jugada_propia, jugada_oponente):
        super().registrar(jugada_propia, jugada_oponente)

        # Calcular resultado
        if jugada_propia == jugada_oponente:
            self.resultado_anterior = 'empate'
        elif GANA_A[jugada_propia] == jugada_oponente:
            self.resultado_anterior = 'gana'
        else:
            self.resultado_anterior = 'pierde'


class Mixto(EstrategiaJugador):
    """Cambia de estrategia cada 10 rondas"""

    def __init__(self):
        super().__init__()
        self.estrategias = [
            Aleatorio(),
            CiclicoAscendente(),
            SesgoJugada('piedra', 0.6),
            AntiPrediccion()
        ]
        self.estrategia_actual = 0

    def decidir_jugada(self) -> str:
        # Cambiar estrategia cada 10 rondas
        if len(self.historial_propio) > 0 and len(self.historial_propio) % 10 == 0:
            self.estrategia_actual = (self.estrategia_actual + 1) % len(self.estrategias)

        return self.estrategias[self.estrategia_actual].decidir_jugada()

    def registrar(self, jugada_propia, jugada_oponente):
        super().registrar(jugada_propia, jugada_oponente)
        # Registrar en la estrategia actual
        self.estrategias[self.estrategia_actual].registrar(jugada_propia, jugada_oponente)


# =============================================================================
# SIMULADOR DE PARTIDA
# =============================================================================

def calcular_resultado(jugada_j1, jugada_j2):
    """Calcula quién gana. Retorna: 1=J1 gana, -1=J1 pierde, 0=empate"""
    if jugada_j1 == jugada_j2:
        return 0
    elif GANA_A[jugada_j1] == jugada_j2:
        return 1
    else:
        return -1


def simular_partida(estrategia: EstrategiaJugador, ia: JugadorIA, num_rondas: int = 50, verbose: bool = False):
    """
    Simula una partida completa.

    Returns:
        dict con estadísticas: victorias_ia, derrotas_ia, empates, winrate
    """
    victorias_ia = 0
    derrotas_ia = 0
    empates = 0

    for ronda in range(1, num_rondas + 1):
        # Decidir jugadas
        jugada_humano = estrategia.decidir_jugada()
        jugada_ia = ia.decidir_jugada()

        # Calcular resultado (desde perspectiva de la IA)
        resultado = calcular_resultado(jugada_ia, jugada_humano)

        if resultado == 1:
            victorias_ia += 1
            resultado_str = "✅ IA gana"
        elif resultado == -1:
            derrotas_ia += 1
            resultado_str = "❌ IA pierde"
        else:
            empates += 1
            resultado_str = "⚪ Empate"

        # Registrar en historial
        ia.registrar_ronda(jugada_ia, jugada_humano)
        estrategia.registrar(jugada_humano, jugada_ia)

        if verbose:
            print(f"  Ronda {ronda:2d}: Humano={jugada_humano:6s} | IA={jugada_ia:6s} | {resultado_str}")

    # Calcular winrate
    total_decisivos = victorias_ia + derrotas_ia
    winrate = (victorias_ia / total_decisivos * 100) if total_decisivos > 0 else 0

    return {
        'victorias_ia': victorias_ia,
        'derrotas_ia': derrotas_ia,
        'empates': empates,
        'winrate': winrate,
        'total_rondas': num_rondas
    }


# =============================================================================
# BATERÍA DE PRUEBAS
# =============================================================================

def ejecutar_bateria_pruebas(num_rondas: int = 50, verbose: bool = False):
    """
    Ejecuta pruebas contra todas las estrategias.
    """

    estrategias_a_probar = [
        ("Aleatorio Puro", Aleatorio()),
        ("Cíclico Ascendente (piedra→papel→tijera)", CiclicoAscendente()),
        ("Cíclico Descendente (tijera→papel→piedra)", CiclicoDescendente()),
        ("Sesgo Piedra (70%)", SesgoJugada('piedra', 0.7)),
        ("Sesgo Papel (70%)", SesgoJugada('papel', 0.7)),
        ("Sesgo Tijera (70%)", SesgoJugada('tijera', 0.7)),
        ("Anti-Predicción", AntiPrediccion()),
        ("Repetidor (repite si gana)", Repetidor()),
        ("Mixto (cambia cada 10 rondas)", Mixto()),
    ]

    print("=" * 70)
    print("  BATERÍA DE PRUEBAS - EVALUACIÓN DE IA")
    print("=" * 70)
    print(f"📊 Configuración: {num_rondas} rondas por estrategia\n")

    resultados_globales = []

    for nombre, estrategia in estrategias_a_probar:
        print(f"\n{'=' * 70}")
        print(f"🧪 PRUEBA: {nombre}")
        print('=' * 70)

        # Crear nueva instancia de IA para cada prueba
        ia = JugadorIA()

        # Simular partida
        resultado = simular_partida(estrategia, ia, num_rondas, verbose)

        # Mostrar resultados
        print(f"\n📊 RESULTADOS:")
        print(f"   Victorias IA: {resultado['victorias_ia']}")
        print(f"   Derrotas IA:  {resultado['derrotas_ia']}")
        print(f"   Empates:      {resultado['empates']}")
        print(f"   Winrate:      {resultado['winrate']:.1f}%")

        # Evaluación
        if resultado['winrate'] >= 60:
            evaluacion = "✅ Excelente adaptación"
        elif resultado['winrate'] >= 50:
            evaluacion = "⚠️  Adaptación aceptable"
        else:
            evaluacion = "❌ Adaptación insuficiente"

        print(f"   {evaluacion}")

        resultados_globales.append({
            'nombre': nombre,
            'winrate': resultado['winrate'],
            'victorias': resultado['victorias_ia'],
            'derrotas': resultado['derrotas_ia'],
            'empates': resultado['empates']
        })

    # Resumen global
    print("\n" + "=" * 70)
    print("📈 RESUMEN GLOBAL")
    print("=" * 70)

    winrate_promedio = np.mean([r['winrate'] for r in resultados_globales])

    print(f"\n{'Estrategia':<40} {'Winrate':>10} {'V':>4} {'D':>4} {'E':>4}")
    print("-" * 70)

    for res in resultados_globales:
        print(
            f"{res['nombre']:<40} {res['winrate']:>9.1f}% {res['victorias']:>4} {res['derrotas']:>4} {res['empates']:>4}")

    print("-" * 70)
    print(f"{'PROMEDIO GLOBAL':<40} {winrate_promedio:>9.1f}%")

    # Evaluación final
    print("\n" + "=" * 70)
    print("🎯 EVALUACIÓN FINAL")
    print("=" * 70)

    if winrate_promedio >= 55:
        print("✅ La IA muestra BUENA capacidad de adaptación")
        print("   Supera el 50% de winrate contra múltiples estrategias")
    elif winrate_promedio >= 45:
        print("⚠️  La IA muestra capacidad de adaptación MODERADA")
        print("   Cerca del equilibrio pero puede mejorar")
    else:
        print("❌ La IA necesita MEJORAS en su adaptación")
        print("   No logra superar consistentemente a oponentes con patrones")

    print("\n💡 Nota: Winrate > 50% indica que la IA es mejor que jugar al azar")
    print("   Winrate ~33% sería lo esperado si todos perdieran contra la IA")
    print("   Winrate ~50% indica equilibrio o juego aleatorio mutuo")
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Punto de entrada principal"""

    import argparse

    parser = argparse.ArgumentParser(description='Evaluar IA de Piedra, Papel o Tijera')
    parser.add_argument('--rondas', type=int, default=50,
                        help='Número de rondas por estrategia (default: 50)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Mostrar detalles de cada ronda')

    args = parser.parse_args()

    ejecutar_bateria_pruebas(num_rondas=args.rondas, verbose=args.verbose)


if __name__ == "__main__":
    main()