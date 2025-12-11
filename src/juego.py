import time
import csv
import os
from pynput import keyboard
from threading import Thread

MANOS_ASCII = {
    'piedra': """
 ██████╗ ██╗███████╗██████╗ ██████╗  █████╗ 
 ██╔══██╗██║██╔════╝██╔══██╗██╔══██╗██╔══██╗
 ██████╔╝██║█████╗  ██║  ██║██████╔╝███████║
 ██╔═══╝ ██║██╔══╝  ██║  ██║██╔══██╗██╔══██║
 ██║     ██║███████╗██████╔╝██║  ██║██║  ██║
 ╚═╝     ╚═╝╚══════╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝
""",
    'papel': """
 ██████╗  █████╗ ██████╗ ███████╗██╗     
 ██╔══██╗██╔══██╗██╔══██╗██╔════╝██║     
 ██████╔╝███████║██████╔╝█████╗  ██║     
 ██╔═══╝ ██╔══██║██╔═══╝ ██╔══╝  ██║     
 ██║     ██║  ██║██║     ███████╗███████╗
 ╚═╝     ╚═╝  ╚═╝╚═╝     ╚══════╝╚══════╝
""",
    'tijera': """
 ████████╗██╗     ██╗███████╗██████╗  █████╗ 
 ╚══██╔══╝██║     ██║██╔════╝██╔══██╗██╔══██╗
    ██║   ██║     ██║█████╗  ██████╔╝███████║
    ██║   ██║██   ██║██╔══╝  ██╔══██╗██╔══██║
    ██║   ██║╚█████╔╝███████╗██║  ██║██║  ██║
    ╚═╝   ╚═╝ ╚════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝
"""
}

NUMEROS_ASCII = {
    3: """
 ██████╗ 
 ╚════██╗
  █████╔╝
  ╚═══██╗
 ██████╔╝
 ╚═════╝ 
""",
    2: """
 ██████╗ 
 ╚════██╗
  █████╔╝
 ██╔═══╝ 
 ███████╗
 ╚══════╝
""",
    1: """
 ██╗
███║
╚██║
 ██║
 ██║
 ╚═╝
"""
}

# Mensajes ASCII grandes
GANAS_ASCII = """
  ██████╗  █████╗ ███╗   ██╗ █████╗ ███████╗
 ██╔════╝ ██╔══██╗████╗  ██║██╔══██╗██╔════╝
 ██║  ███╗███████║██╔██╗ ██║███████║███████╗
 ██║   ██║██╔══██║██║╚██╗██║██╔══██║╚════██║
 ╚██████╔╝██║  ██║██║ ╚████║██║  ██║███████║
  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝
"""

PIERDES_ASCII = """
 ██████╗ ██╗███████╗██████╗ ██████╗ ███████╗███████╗
 ██╔══██╗██║██╔════╝██╔══██╗██╔══██╗██╔════╝██╔════╝
 ██████╔╝██║█████╗  ██████╔╝██║  ██║█████╗  ███████╗
 ██╔═══╝ ██║██╔══╝  ██╔══██╗██║  ██║██╔══╝  ╚════██║
 ██║     ██║███████╗██║  ██║██████╔╝███████╗███████║
 ╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝ ╚══════╝╚══════╝
"""

EMPATE_ASCII = """
 ███████╗███╗   ███╗██████╗  █████╗ ████████╗███████╗
 ██╔════╝████╗ ████║██╔══██╗██╔══██╗╚══██╔══╝██╔════╝
 █████╗  ██╔████╔██║██████╔╝███████║   ██║   █████╗  
 ██╔══╝  ██║╚██╔╝██║██╔═══╝ ██╔══██║   ██║   ██╔══╝  
 ███████╗██║ ╚═╝ ██║██║     ██║  ██║   ██║   ███████╗
 ╚══════╝╚═╝     ╚═╝╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚══════╝
"""

# Códigos de color ANSI
COLOR_VERDE = "\033[92m"
COLOR_ROJO = "\033[91m"
COLOR_AMARILLO = "\033[93m"
COLOR_RESET = "\033[0m"


class JuegoPiedraPapelTijera:
    def __init__(self):
        self.jugada_j1 = None
        self.jugada_j2 = None
        self.tiempo_j1 = None
        self.tiempo_j2 = None
        self.tiempo_inicio_jugada = None
        self.ronda = 0
        self.puntos_j1 = 0
        self.puntos_j2 = 0
        self.archivo_csv = 'resultados_juego.csv'
        self.listener = None
        self.jugada_en_curso = False

        # Mapeo de teclas a jugadas
        self.teclas_j1 = {'1': 'piedra', '2': 'papel', '3': 'tijera'}
        self.teclas_j2 = {',': 'piedra', '.': 'papel', '-': 'tijera'}

        # Inicializar archivo CSV
        self.inicializar_csv()

    def inicializar_csv(self):
        """Crea el archivo CSV si no existe"""
        if not os.path.exists(self.archivo_csv):
            with open(self.archivo_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Ronda', 'Jugador 1', 'Jugador 2', 'Ganador', 'Tiempo Jugador 1', 'Tiempo Jugador 2'])

    def on_press(self, key):
        """Captura las teclas presionadas"""
        if not self.jugada_en_curso:
            return

        try:
            tecla = key.char.lower()
            tiempo_actual = time.time()

            # Jugador 1
            if tecla in self.teclas_j1 and self.jugada_j1 is None:
                self.jugada_j1 = self.teclas_j1[tecla]
                self.tiempo_j1 = round(tiempo_actual - self.tiempo_inicio_jugada, 3)

            # Jugador 2
            if tecla in self.teclas_j2 and self.jugada_j2 is None:
                self.jugada_j2 = self.teclas_j2[tecla]
                self.tiempo_j2 = round(tiempo_actual - self.tiempo_inicio_jugada, 3)

        except AttributeError:
            pass

    def cuenta_regresiva(self):
        """Realiza la cuenta regresiva"""

        time.sleep(1)

        for i in range(3, 0, -1):
            print("\n" + NUMEROS_ASCII[i])
            time.sleep(1)



        # Activar captura de jugadas y marcar tiempo de inicio
        self.tiempo_inicio_jugada = time.time()
        self.jugada_en_curso = True

    def esperar_jugadas(self):
        """Espera a que ambos jugadores elijan"""
        timeout = 5
        tiempo_inicio = time.time()

        print("\nEsperando jugadas...")

        while self.jugada_j1 is None or self.jugada_j2 is None:
            tiempo_transcurrido = time.time() - tiempo_inicio
            tiempo_restante = timeout - tiempo_transcurrido

            if tiempo_restante <= 0:
                break

            # Mostrar estado de jugadas con tiempo restante
            estado_j1 = "✓" if self.jugada_j1 else "⏳"
            estado_j2 = "✓" if self.jugada_j2 else "⏳"
            print(f"\rJugador 1: {estado_j1}  |  Jugador 2: {estado_j2}  |  Tiempo: {tiempo_restante:.1f}s", end='',
                  flush=True)
            time.sleep(0.1)

        # Si ambos jugadores eligieron antes del timeout
        if self.jugada_j1 and self.jugada_j2:
            print("\n\n¡Ambos jugadores han elegido!")
            time.sleep(0.5)
            return

        # Si algún jugador no eligió, asignar una jugada aleatoria
        import random
        opciones = ['piedra', 'papel', 'tijera']

        if self.jugada_j1 is None:
            self.jugada_j1 = random.choice(opciones)
            self.tiempo_j1 = timeout
            print("\n⚠️ Jugador 1 no eligió a tiempo. Jugada aleatoria.")

        if self.jugada_j2 is None:
            self.jugada_j2 = random.choice(opciones)
            self.tiempo_j2 = timeout
            print("⚠️ Jugador 2 no eligió a tiempo. Jugada aleatoria.")

        time.sleep(1)

    def mostrar_jugadas(self):
        """Muestra las jugadas con arte ASCII"""
        print("\n" + "=" * 100)
        print("JUGADAS:")
        print("=" * 100)

        # Dividir las manos en líneas
        lineas_j1 = MANOS_ASCII[self.jugada_j1].strip().split('\n')
        lineas_j2 = MANOS_ASCII[self.jugada_j2].strip().split('\n')

        # Mostrar nombres de jugadores con tiempos de reacción
        print(
            f"\n{'JUGADOR 1: ' + self.jugada_j1.upper() + f' ({self.tiempo_j1}s)':^45}          {'JUGADOR 2: ' + self.jugada_j2.upper() + f' ({self.tiempo_j2}s)':^45}")

        # Mostrar las manos lado a lado con más separación
        max_lineas = max(len(lineas_j1), len(lineas_j2))
        for i in range(max_lineas):
            linea_j1 = lineas_j1[i] if i < len(lineas_j1) else ""
            linea_j2 = lineas_j2[i] if i < len(lineas_j2) else ""
            print(f"{linea_j1:45}          {linea_j2:45}")

    def determinar_ganador(self):
        """Determina el ganador de la ronda"""
        if self.jugada_j1 == self.jugada_j2:
            return "Empate"

        if ((self.jugada_j1 == 'piedra' and self.jugada_j2 == 'tijera') or
                (self.jugada_j1 == 'papel' and self.jugada_j2 == 'piedra') or
                (self.jugada_j1 == 'tijera' and self.jugada_j2 == 'papel')):
            self.puntos_j1 += 1
            return "Jugador 1"
        else:
            self.puntos_j2 += 1
            return "Jugador 2"

    def guardar_resultado(self, ganador):
        """Guarda el resultado en el archivo CSV"""
        with open(self.archivo_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([self.ronda, self.jugada_j1, self.jugada_j2, ganador,
                             self.tiempo_j1, self.tiempo_j2])

    def mostrar_resultado(self, ganador):
        """Muestra el resultado de la ronda"""
        print("\n" + "=" * 100)
        if ganador == "Empate":
            print(COLOR_AMARILLO + EMPATE_ASCII + COLOR_RESET)
        elif ganador == "Jugador 2":
            print(COLOR_VERDE + GANAS_ASCII + COLOR_RESET)
        else:  # Jugador 1 gana
            print(COLOR_ROJO + PIERDES_ASCII + COLOR_RESET)



    def jugar_ronda(self):
        """Ejecuta una ronda completa"""
        self.ronda += 1
        self.jugada_j1 = None
        self.jugada_j2 = None
        self.tiempo_j1 = None
        self.tiempo_j2 = None
        self.jugada_en_curso = False  # Desactivar captura durante cuenta regresiva


        # Iniciar listener de teclado
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

        # Cuenta regresiva (activará jugada_en_curso al final)
        self.cuenta_regresiva()

        # Esperar jugadas
        self.esperar_jugadas()

        # Detener listener
        self.jugada_en_curso = False
        self.listener.stop()

        # Mostrar jugadas
        self.mostrar_jugadas()

        # Determinar ganador
        ganador = self.determinar_ganador()

        # Mostrar resultado
        self.mostrar_resultado(ganador)

        # Guardar en CSV
        self.guardar_resultado(ganador)

        print({self.ronda})

    def jugar(self):
        """Bucle principal del juego"""

        try:
            while True:
                self.jugar_ronda()
                time.sleep(2)  # Pausa breve entre rondas
        except KeyboardInterrupt:
            print(f"\n\n{'=' * 100}")
            print("RESULTADO FINAL")
            print(f"{'=' * 100}")
            print(f"Jugador 1: {self.puntos_j1} puntos")
            print(f"Jugador 2: {self.puntos_j2} puntos")

            if self.puntos_j1 > self.puntos_j2:
                print("\n🎉 ¡JUGADOR 1 GANA LA PARTIDA!")
            elif self.puntos_j2 > self.puntos_j1:
                print("\n🎉 ¡JUGADOR 2 GANA LA PARTIDA!")
            else:
                print("\n🤝 ¡EMPATE EN LA PARTIDA!")

            print(f"\nResultados guardados en: {self.archivo_csv}")


# Ejecutar el juego
if __name__ == "__main__":
    juego = JuegoPiedraPapelTijera()
    juego.jugar()