"""
Test de Adaptación de la IA
============================

Este script prueba que la IA se adapta correctamente a diferentes patrones sintéticos.
Es crucial para validar la efectividad de las Features y la lógica de Castigo/Exploración.

Ejecuta: python src/test_adaptacion.py
"""

import sys
import time
from pathlib import Path

# Agregar el directorio src al path para importar modelo
sys.path.insert(0, str(Path(__file__).parent))

# Importamos la clase JugadorIA y la constante GANA_A para la lógica de reglas
from modelo import JugadorIA, GANA_A

# Colores (ANSI)
COLOR_VERDE = "\033[92m"
COLOR_ROJO = "\033[91m"
COLOR_AMARILLO = "\033[93m"
COLOR_CYAN = "\033[96m"
COLOR_RESET = "\033[0m"


def determinar_resultado_ronda(jugada_ia: str, jugada_humano: str) -> str:
    """Determina el resultado de la ronda (victoria_ia, derrota_ia, empate)."""
    if jugada_ia == jugada_humano:
        return "empate"
    # Si la jugada de la IA gana a la del humano (usando la tabla GANA_A)
    elif GANA_A.get(jugada_ia) == jugada_humano:
        return "victoria_ia"
    else:
        return "derrota_ia"


def probar_patron(ia: JugadorIA, patron: list, nombre_patron: str):
    """
    Prueba la IA contra un patrón específico.

    Args:
        ia: Instancia de JugadorIA (debe ser una instancia limpia para cada prueba)
        patron: Lista de jugadas del "oponente"
        nombre_patron: Nombre descriptivo del patrón
    """
    print(f"\n{'=' * 70}")
    print(f"🧪 PRUEBA: {nombre_patron}")
    print(f"{'=' * 70}")
    print(f"Patrón: {' → '.join(patron)}")

    victorias_ia = 0
    derrotas_ia = 0
    empates = 0

    # Simular las rondas
    for i, jugada_humano in enumerate(patron):

        # 1. IA DECIDE (usando el historial hasta i-1)
        jugada_ia = ia.decidir_jugada()

        # 2. DETERMINAR RESULTADO (Ronda i)
        resultado = determinar_resultado_ronda(jugada_ia, jugada_humano)

        if resultado == "victoria_ia":
            victorias_ia += 1
            simbolo_resultado = "✅"
        elif resultado == "derrota_ia":
            derrotas_ia += 1
            simbolo_resultado = "❌"
        else:
            empates += 1
            simbolo_resultado = "⚪"

        # 3. MOSTRAR Y REGISTRAR
        print(f"  Ronda {i + 1:2d}: Humano={jugada_humano:6} | IA={jugada_ia:6} | {simbolo_resultado}")

        # 4. REGISTRAR: Añadir la jugada real de la ronda i al historial (tiempos ficticios)
        ia.registrar_ronda(jugada_ia, jugada_humano, 0.5, 0.5)

        # Pequeña pausa para visualización
        time.sleep(0.05)

    # Calcular winrate (solo sobre victorias y derrotas)
    total_decisivas = victorias_ia + derrotas_ia
    if total_decisivas > 0:
        winrate = victorias_ia / total_decisivas * 100
    else:
        winrate = 0

    # Mostrar resultados
    print(f"\n{'─' * 70}")
    print(f"📊 RESULTADOS:")
    print(f"   Victorias IA: {victorias_ia}")
    print(f"   Derrotas IA:  {derrotas_ia}")
    print(f"   Empates:      {empates}")
    print(f"   {COLOR_CYAN}Winrate (V / (V+D)): {winrate:.1f}%{COLOR_RESET}")

    # Análisis de adaptación (Objetivo > 42%)
    if winrate > 50:
        print(f"   {COLOR_VERDE}✅ Adaptación excelente (Winrate > 50%){COLOR_RESET}")
    elif winrate > 42:
        print(f"   {COLOR_AMARILLO}⚠️  Adaptación aceptable (Winrate > 42%){COLOR_RESET}")
    else:
        print(f"   {COLOR_ROJO}❌ Adaptación insuficiente{COLOR_RESET}")

    print(f"{'─' * 70}")

    return winrate


def main():
    """Ejecuta las pruebas de adaptación."""
    print("=" * 70)
    print("   TEST DE ADAPTACIÓN DE LA IA")
    print("=" * 70)

    # Cargar modelo (verificamos que exista)
    try:
        ia_modelo_base = JugadorIA()
        if ia_modelo_base.modelo is None:
            print(f"{COLOR_ROJO}❌ Error: No se encontró modelo entrenado{COLOR_RESET}")
            print(f"   Entrena primero: python src/modelo.py")
            return
    except Exception as e:
        print(f"{COLOR_ROJO}❌ Error al cargar modelo: {e}{COLOR_RESET}")
        return

    # Patrones de prueba
    patrones = [
        # Patrón 1: REPETITIVO (solo piedra) -> La IA debería predecir 'piedra' (jugar 'papel')
        {
            'nombre': 'Patrón 1: Jugador REPETITIVO (solo piedra)',
            'jugadas': ['piedra'] * 20
        },

        # Patrón 2: CÍCLICO (piedra-papel-tijera) -> La IA debería predecir la jugada siguiente
        {
            'nombre': 'Patrón 2: Jugador CÍCLICO (piedra→papel→tijera)',
            'jugadas': ['piedra', 'papel', 'tijera'] * 7
        },

        # Patrón 3: 2-2-2 (Lags cortos) -> La IA debe capturar el ciclo de repetición corta
        {
            'nombre': 'Patrón 3: Jugador con PATRÓN 2-2-2',
            'jugadas': ['piedra', 'piedra', 'papel', 'papel', 'tijera', 'tijera'] * 4
        },

        # Patrón 4: CAMBIO de estrategia (piedra→papel) -> La IA debe reaccionar al cambio de tendencia
        {
            'nombre': 'Patrón 4: Jugador que CAMBIA estrategia (piedra→papel)',
            'jugadas': ['piedra'] * 10 + ['papel'] * 10
        }
    ]

    resultados = []

    # Ejecutar todas las pruebas
    for patron_info in patrones:
        # Resetear IA para cada prueba (crucial para limpiar el historial)
        ia_prueba = JugadorIA()

        # Ejecutar prueba
        winrate = probar_patron(
            ia_prueba,
            patron_info['jugadas'],
            patron_info['nombre']
        )

        resultados.append({
            'nombre': patron_info['nombre'],
            'winrate': winrate
        })

        time.sleep(1)

    # Resumen final
    print(f"\n\n{'=' * 70}")
    print("   📊 RESUMEN DE TODAS LAS PRUEBAS")
    print(f"{'=' * 70}\n")

    winrate_promedio = sum(r['winrate'] for r in resultados) / len(resultados)

    for i, resultado in enumerate(resultados, 1):
        color = COLOR_VERDE if resultado['winrate'] > 50 else COLOR_AMARILLO if resultado[
                                                                                    'winrate'] > 42 else COLOR_ROJO
        print(f"{i}. {resultado['nombre']}")
        print(f"   {color}Winrate: {resultado['winrate']:.1f}%{COLOR_RESET}")

    print(f"\n{'─' * 70}")
    print(f"📈 Winrate promedio: {COLOR_CYAN}{winrate_promedio:.1f}%{COLOR_RESET}")
    print(f"{'─' * 70}")

    if winrate_promedio > 50:
        print(f"\n{COLOR_VERDE}✅ ÉXITO: La IA se adapta correctamente (Promedio > 50%){COLOR_RESET}")
    elif winrate_promedio > 42:
        print(f"\n{COLOR_AMARILLO}⚠️  ACEPTABLE: La IA muestra adaptación (Promedio > 42%){COLOR_RESET}")
    else:
        print(f"\n{COLOR_ROJO}❌ PROBLEMA: La IA no se adapta bien{COLOR_RESET}")


if __name__ == "__main__":
    main()