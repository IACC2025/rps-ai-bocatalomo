"""
RPSAI - JUEGO INTERACTIVO (Piedra, Papel o Tijera)
===================================================
Juego interactivo de RPS contra la IA, con temporizador y teclado.
Juega 15 rondas consecutivas sin interrupción.

WINRATE: Solo cuenta victorias y derrotas (empates NO cuentan)
"""

import time
import sys
import random
from pathlib import Path
from pynput import keyboard
from threading import Thread
import argparse

# Configuración de ruta mínima para importar modelo
sys.path.insert(0, str(Path(__file__).parent))

# Importamos solo lo esencial: la clase JugadorIA y reglas
try:
    from modelo import JugadorIA, GANA_A
except ImportError:
    class JugadorIA:
        def __init__(self): print("⚠️ Modelo no encontrado. Jugando ALEATORIO.")
        def registrar_ronda(self, *args): pass
        def decidir_jugada(self): return random.choice(['piedra', 'papel', 'tijera'])

# --- ARTE Y COLORES ---
MANOS_ASCII = {
    'piedra': """
╔═══════════╗
║  PIEDRA   ║
║     ✊    ║
╚═══════════╝
""",
    'papel': """
╔═══════════╗
║   PAPEL   ║
║     ✋    ║
╚═══════════╝
""",
    'tijera': """
╔═══════════╗
║  TIJERA   ║
║     ✌️    ║
╚═══════════╝
"""
}

COLOR_VERDE = "\033[92m"
COLOR_ROJO = "\033[91m"
COLOR_AMARILLO = "\033[93m"
COLOR_CYAN = "\033[96m"
COLOR_RESET = "\033[0m"


def obtener_nota(winrate: float) -> tuple:
    """
    Calcula la nota basada en el winrate.

    IMPORTANTE: El winrate se calcula SOLO con victorias y derrotas.
    Los empates NO cuentan para el winrate.
    """
    if winrate >= 55:
        return 10, f"{COLOR_VERDE}🎉 BONUS: Fallos en examen no restan!{COLOR_RESET}"
    elif winrate >= 50:
        return 10, None
    elif winrate >= 49:
        return 9, None
    elif winrate >= 48:
        return 8, None
    elif winrate >= 46:
        return 7, None
    elif winrate >= 44:
        return 6, None
    elif winrate >= 42:
        return 5, None
    elif winrate >= 40:
        return 4, None
    elif winrate >= 39:
        return 3, None
    elif winrate >= 37:
        return 2, None
    elif winrate >= 35:
        return 1, None
    else:
        return 0, None


class JuegoRPS:
    """Clase principal para ejecutar el juego interactivo."""

    def __init__(self, num_rondas: int = 15):
        self.num_rondas = num_rondas
        self.ia = JugadorIA()

        self.ronda_actual = 0
        self.victorias_ia = 0
        self.derrotas_ia = 0
        self.empates = 0

        self.jugada_humano = None
        self.tiempo_humano = None
        self.tiempo_inicio_jugada = None
        self.jugada_en_curso = False
        self.listener = None
        self.teclas_humano = {'1': 'piedra', '2': 'papel', '3': 'tijera'}

        sys.stdout.write(f"{COLOR_CYAN}="*60 + COLOR_RESET + "\n")
        sys.stdout.write(f"{COLOR_CYAN}   PIEDRA, PAPEL O TIJERA (Modo Interactivo){COLOR_RESET}\n")
        sys.stdout.write(f"{COLOR_CYAN}="*60 + COLOR_RESET + "\n")


    def on_press(self, key):
        """Captura la pulsación de tecla del humano."""
        if not self.jugada_en_curso: return
        try:
            tecla = key.char
            tiempo_actual = time.time()
            if tecla in self.teclas_humano and self.jugada_humano is None:
                self.jugada_humano = self.teclas_humano[tecla]
                self.tiempo_humano = round(tiempo_actual - self.tiempo_inicio_jugada, 3)
                self.jugada_en_curso = False
        except AttributeError:
            pass

    def cuenta_regresiva(self):
        """Muestra la cuenta regresiva."""
        print(f"\n{COLOR_CYAN}{'—'*60}{COLOR_RESET}")
        print(f"{COLOR_CYAN}   RONDA {self.ronda_actual}/{self.num_rondas}{COLOR_RESET}")
        print(f"{COLOR_CYAN}{'—'*60}{COLOR_RESET}")

        print("\n⏱️  Preparados...")
        time.sleep(1)

        for i in range(3, 0, -1):
            print(f"\n   {i}...", end='', flush=True)
            time.sleep(0.8)

        print(f"\n{COLOR_AMARILLO}   ¡YA!{COLOR_RESET}")

        self.tiempo_inicio_jugada = time.time()
        self.jugada_en_curso = True

    def esperar_jugada_humano(self):
        """Espera la jugada del humano con timeout."""
        timeout = 3.0
        tiempo_inicio = time.time()

        print(f"\n{COLOR_CYAN}[Teclas] 1=Piedra  2=Papel  3=Tijera{COLOR_RESET}")

        while self.jugada_humano is None and self.jugada_en_curso and (time.time() - tiempo_inicio < timeout):
            tiempo_restante = timeout - (time.time() - tiempo_inicio)
            estado = "✓" if self.jugada_humano else "⏳"
            print(f"\rHumano: {estado}  |  Tiempo: {tiempo_restante:.1f}s", end='', flush=True)
            time.sleep(0.1)

        print() # Nueva línea

        # Si no eligió
        if self.jugada_humano is None:
            self.jugada_humano = random.choice(['piedra', 'papel', 'tijera'])
            self.tiempo_humano = timeout
            print(f"{COLOR_AMARILLO}⚠️  ¡Tiempo agotado! Jugada automática: {self.jugada_humano.upper()}{COLOR_RESET}")
        else:
             print(f"{COLOR_VERDE}Humano eligió: {self.jugada_humano.upper()} en {self.tiempo_humano:.3f}s{COLOR_RESET}")


    def determinar_ganador(self, jugada_humano: str, jugada_ia: str) -> str:
        """Determina el resultado de la ronda y actualiza contadores."""
        try:
            if jugada_ia == jugada_humano:
                self.empates += 1
                return "empate"
            elif GANA_A[jugada_ia] == jugada_humano:
                self.victorias_ia += 1
                return "ia" # IA gana
            else:
                self.derrotas_ia += 1
                return "humano" # Humano gana
        except KeyError: return "error"

    def mostrar_resultado(self, jugada_humano: str, jugada_ia: str,
                          tiempo_humano: float, tiempo_ia: float, resultado: str):
        """Muestra el resultado de la ronda con arte ASCII."""
        print("\n" + "="*60)
        print("RESULTADO DE LA RONDA")
        print("="*60)

        # Mostrar jugadas lado a lado
        lineas_humano = MANOS_ASCII.get(jugada_humano, "J. Inválida").strip().split('\n')
        lineas_ia = MANOS_ASCII.get(jugada_ia, "J. Inválida").strip().split('\n')

        print(f"\n{'HUMANO (TÚ)':^30} {'IA':^30}")
        print(f"{tiempo_humano:.3f}s".center(30) + f"{tiempo_ia:.3f}s".center(30))

        for lh, li in zip(lineas_humano, lineas_ia):
            print(f"{lh:30} {li:30}")

        # Mostrar resultado final
        print()
        if resultado == "ia":
            print(f"{COLOR_ROJO}{'>>> IA GANA LA RONDA <<<':^60}{COLOR_RESET}")
        elif resultado == "humano":
            print(f"{COLOR_VERDE}{'>>> TÚ GANAS LA RONDA <<<':^60}{COLOR_RESET}")
        else:
            print(f"{COLOR_AMARILLO}{'>>> EMPATE <<<':^60}{COLOR_RESET}")

    def mostrar_progreso_final(self):
        """
        Muestra el resumen de estadísticas al final del juego.

        CORRECCIÓN CLAVE: Winrate se calcula SOLO con victorias y derrotas.
        Los empates NO cuentan.
        """
        # Total de rondas DECISIVAS (sin empates)
        total_decisivas = self.victorias_ia + self.derrotas_ia

        # Calcular winrate SOLO con rondas decisivas
        if total_decisivas > 0:
            winrate = (self.victorias_ia / total_decisivas) * 100
        else:
            winrate = 0

        print(f"\n{COLOR_CYAN}{'='*60}{COLOR_RESET}")
        print(f"{COLOR_CYAN}   RESULTADOS FINALES{COLOR_RESET}")
        print(f"{COLOR_CYAN}{'='*60}{COLOR_RESET}")

        print(f"\n📊 Estadísticas completas:")
        print(f"   Total de rondas jugadas: {self.ronda_actual}")
        print(f"   Victorias IA: {self.victorias_ia}")
        print(f"   Derrotas IA: {self.derrotas_ia}")
        print(f"   Empates: {self.empates}")

        print(f"\n📈 Cálculo de Winrate:")
        print(f"   Rondas decisivas (V + D): {total_decisivas}")
        print(f"   Empates ignorados: {self.empates}")
        print(f"   Fórmula: {self.victorias_ia} victorias / {total_decisivas} decisivas = {winrate:.1f}%")

        print(f"\n   {COLOR_CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{COLOR_RESET}")
        print(f"   {COLOR_CYAN}WINRATE FINAL DE LA IA: {winrate:.1f}%{COLOR_RESET}")
        print(f"   {COLOR_CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{COLOR_RESET}")

        # Calcular nota
        nota, bonus = obtener_nota(winrate)

        print(f"\n📝 Evaluación:")
        print(f"   Nota obtenida: {nota}/10")
        if bonus:
            print(f"   {bonus}")

        # Mensaje de logro
        print(f"\n🎯 Análisis:")
        if winrate >= 50:
            print(f"   {COLOR_VERDE}✅ ¡EXCELENTE! La IA supera el 50% de winrate{COLOR_RESET}")
        elif winrate >= 42:
            print(f"   {COLOR_VERDE}✅ ¡BIEN! Objetivo base del 42% superado{COLOR_RESET}")
        elif winrate >= 40:
            print(f"   {COLOR_AMARILLO}⚠️  Cerca del objetivo (40%+){COLOR_RESET}")
        else:
            print(f"   {COLOR_ROJO}❌ Por debajo del objetivo (40%){COLOR_RESET}")

        print(f"\n📋 Tabla de referencia:")
        print(f"   <35% = 0  |  35% = 1  |  37% = 2  |  39% = 3")
        print(f"   40% = 4   |  42% = 5  |  44% = 6  |  46% = 7")
        print(f"   48% = 8   |  49% = 9  |  50%+ = 10 | 55%+ = BONUS")

        print(f"{COLOR_CYAN}{'='*60}{COLOR_RESET}")


    def jugar_ronda(self):
        """Ejecuta una ronda completa."""
        self.ronda_actual += 1
        self.jugada_humano = None
        self.tiempo_humano = None
        self.jugada_en_curso = False

        # 1. IA decide su jugada
        tiempo_inicio_ia = time.time()
        jugada_ia = self.ia.decidir_jugada()
        tiempo_ia = round(time.time() - tiempo_inicio_ia, 3)

        # 2. Capturar jugada del humano
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        self.cuenta_regresiva()
        self.esperar_jugada_humano()

        # Detener la captura
        self.jugada_en_curso = False
        self.listener.stop()

        # 3. Evaluar y mostrar
        resultado = self.determinar_ganador(self.jugada_humano, jugada_ia)
        self.mostrar_resultado(self.jugada_humano, jugada_ia,
                               self.tiempo_humano, tiempo_ia, resultado)

        # 4. Registrar en el historial de la IA (para la siguiente predicción)
        self.ia.registrar_ronda(jugada_ia, self.jugada_humano, tiempo_ia, self.tiempo_humano)

        # 5. Mostrar progreso parcial
        if self.ronda_actual % 5 == 0:
            total_decisivas = self.victorias_ia + self.derrotas_ia
            if total_decisivas > 0:
                winrate_parcial = (self.victorias_ia / total_decisivas) * 100
                print(f"\n{COLOR_CYAN}📊 Progreso: {self.victorias_ia}V-{self.derrotas_ia}D-{self.empates}E | Winrate: {winrate_parcial:.1f}%{COLOR_RESET}")

        time.sleep(1)

    def iniciar_juego(self):
        """Inicia el bucle de juego para el número fijo de rondas."""
        print(f"\nSe jugarán {self.num_rondas} rondas consecutivas.")
        print(f"\n{COLOR_CYAN}ℹ️  IMPORTANTE: El winrate se calcula SOLO con victorias y derrotas.{COLOR_RESET}")
        print(f"{COLOR_CYAN}   Los empates NO cuentan para el winrate.{COLOR_RESET}")
        print("\nEl juego comenzará en 2 segundos...")
        time.sleep(2)

        try:
            for _ in range(self.num_rondas):
                self.jugar_ronda()

            self.mostrar_progreso_final()

        except KeyboardInterrupt:
            print(f"\n\n{COLOR_AMARILLO}Juego interrumpido en la ronda {self.ronda_actual}.{COLOR_RESET}")
            self.mostrar_progreso_final()


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Juego de Piedra, Papel o Tijera contra IA")
    parser.add_argument("-n", "--rondas", type=int, default=15,
                        help="Número de rondas a jugar (default: 15)")
    args = parser.parse_args()

    juego = JuegoRPS(num_rondas=args.rondas)
    juego.iniciar_juego()


if __name__ == "__main__":
    main()