"""
API Flask para servir el modelo de IA de Piedra, Papel o Tijera
=================================================================
Expone el modelo entrenado como API REST para integración con la app Tamagotchi.

Endpoints:
- GET  /health           - Estado del servidor y modelo
- POST /predict          - Obtener predicción de la IA
- POST /register-round   - Registrar una ronda jugada

Puerto: 5001
CORS: Habilitado para localhost:5173 (app React)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

# Importar clases del modelo entrenado
try:
    from src.modelo import JugadorIA, PIERDE_CONTRA, NUM_A_JUGADA, JUGADA_A_NUM
    MODELO_CARGADO = True
except ImportError as e:
    print(f"⚠️  Error al importar modelo: {e}")
    MODELO_CARGADO = False

# ============================================
# CONFIGURACIÓN
# ============================================

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:5175"])

# Ruta del CSV de partidas web
RUTA_PROYECTO = Path(__file__).parent
RUTA_CSV = RUTA_PROYECTO / "data" / "partidas_web.csv"

# Instancia global de la IA
ia_jugador: Optional['JugadorIA'] = None
total_rondas_jugadas = 0

# ============================================
# INICIALIZACIÓN
# ============================================

def inicializar_ia():
    """Inicializa la IA y carga el modelo entrenado."""
    global ia_jugador
    
    if not MODELO_CARGADO:
        print("❌ No se puede inicializar IA - modelo no cargado")
        return False
    
    try:
        ia_jugador = JugadorIA()
        print("✅ Modelo de IA cargado correctamente")
        return True
    except Exception as e:
        print(f"❌ Error al cargar modelo: {e}")
        return False


def inicializar_csv():
    """Crea el archivo CSV si no existe."""
    if not RUTA_CSV.exists():
        RUTA_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(RUTA_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Ronda', 'Jugador 1', 'Jugador 2', 'Ganador', 
                'Tiempo Jugador 1', 'Tiempo Jugador 2', 'Timestamp'
            ])
        print(f"✅ CSV creado: {RUTA_CSV}")
    else:
        # Contar rondas existentes
        global total_rondas_jugadas
        with open(RUTA_CSV, 'r', encoding='utf-8') as f:
            total_rondas_jugadas = sum(1 for _ in f) - 1  # -1 para header
        print(f"✅ CSV existente: {total_rondas_jugadas} rondas previas")


def guardar_ronda_csv(jugada_player: str, jugada_ia: str, ganador: str):
    """Guarda una ronda en el CSV."""
    global total_rondas_jugadas
    total_rondas_jugadas += 1
    
    with open(RUTA_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            total_rondas_jugadas,
            jugada_ia,      # Jugador 1 = IA
            jugada_player,  # Jugador 2 = Humano (se predice su jugada)
            ganador,
            0.0,  # Tiempo no aplica en web
            0.0,
            datetime.now().isoformat()
        ])


# ============================================
# ENDPOINTS
# ============================================

@app.route('/health', methods=['GET'])
def health():
    """Verifica el estado del servidor y modelo."""
    return jsonify({
        'status': 'ok',
        'modelLoaded': ia_jugador is not None,
        'totalRoundsPlayed': total_rondas_jugadas,
        'csvPath': str(RUTA_CSV)
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predice la próxima jugada del oponente y decide qué jugar.
    
    Body esperado:
    {
        "history": {
            "player": ["piedra", "tijera", ...],
            "ai": ["papel", "piedra", ...]
        }
    }
    
    Respuesta:
    {
        "predictedOpponentMove": "piedra",
        "aiMove": "papel",
        "reasoning": "Detectado patrón cíclico"
    }
    """
    if not ia_jugador:
        return jsonify({
            'error': 'Modelo de IA no cargado'
        }), 500
    
    try:
        data = request.get_json()
        
        if not data or 'history' not in data:
            return jsonify({
                'error': 'Falta campo "history" en el body'
            }), 400
        
        history = data['history']
        player_moves = history.get('player', [])
        ai_moves = history.get('ai', [])
        
        # Validar que tengan la misma longitud
        if len(player_moves) != len(ai_moves):
            return jsonify({
                'error': 'Historial inconsistente: player y ai deben tener la misma longitud'
            }), 400
        
        # Registrar historial en la IA
        # La IA predice la jugada del "Jugador 2" (humano), así que:
        # jugada_j1 = IA, jugada_j2 = Humano
        for i in range(len(player_moves)):
            ia_jugador.registrar_ronda(
                jugada_j1=ai_moves[i],      # Lo que jugó la IA
                jugada_j2=player_moves[i],  # Lo que jugó el humano
                tiempo_j1=0.0,
                tiempo_j2=0.0
            )
        
        # Obtener predicción de la IA
        prediccion_humano = ia_jugador.predecir_jugada_oponente()
        jugada_ia = ia_jugador.decidir_jugada()
        
        # Logs para debugging
        print(f"📊 Historial: {len(player_moves)} rondas")
        print(f"🤔 IA predice que humano jugará: {prediccion_humano}")
        print(f"🎲 IA decide jugar: {jugada_ia}")
        
        return jsonify({
            'predictedOpponentMove': prediccion_humano,
            'aiMove': jugada_ia,
            'reasoning': f'Basado en {len(player_moves)} rondas previas'
        })
        
    except Exception as e:
        print(f"❌ Error en /predict: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Error interno: {str(e)}'
        }), 500


@app.route('/register-round', methods=['POST'])
def register_round():
    """
    Registra una ronda jugada en el CSV.
    
    Body esperado:
    {
        "player": "piedra",
        "ai": "papel",
        "winner": "ai" | "player" | "tie"
    }
    
    Respuesta:
    {
        "success": true,
        "roundNumber": 123
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Body vacío'}), 400
        
        player = data.get('player')
        ai = data.get('ai')
        winner = data.get('winner')
        
        if not all([player, ai, winner]):
            return jsonify({
                'error': 'Faltan campos: player, ai, winner'
            }), 400
        
        # Validar jugadas
        jugadas_validas = ['piedra', 'papel', 'tijera']
        if player not in jugadas_validas or ai not in jugadas_validas:
            return jsonify({
                'error': f'Jugadas inválidas. Debe ser: {jugadas_validas}'
            }), 400
        
        # Mapear ganador a formato del CSV
        ganador_csv = {
            'ai': 'Jugador 1',
            'player': 'Jugador 2',
            'tie': 'Empate'
        }.get(winner, 'Empate')
        
        # Guardar en CSV
        guardar_ronda_csv(player, ai, ganador_csv)
        
        print(f"💾 Ronda guardada: Player={player}, IA={ai}, Ganador={ganador_csv}")
        
        return jsonify({
            'success': True,
            'roundNumber': total_rondas_jugadas
        })
        
    except Exception as e:
        print(f"❌ Error en /register-round: {e}")
        return jsonify({
            'error': f'Error interno: {str(e)}'
        }), 500


@app.route('/reset', methods=['POST'])
def reset():
    """Reinicia el historial de la IA (útil para testing)."""
    global ia_jugador
    
    if MODELO_CARGADO:
        ia_jugador = JugadorIA()
        print("🔄 IA reiniciada")
        return jsonify({'success': True, 'message': 'IA reiniciada'})
    else:
        return jsonify({'error': 'Modelo no disponible'}), 500


# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    print("=" * 60)
    print("   🤖 SERVIDOR DE IA - PIEDRA, PAPEL O TIJERA")
    print("=" * 60)
    
    # Inicializar CSV
    inicializar_csv()
    
    # Inicializar IA
    if inicializar_ia():
        print(f"\n🚀 Servidor corriendo en http://localhost:5001")
        print(f"📁 Partidas guardadas en: {RUTA_CSV}")
        print(f"🔗 CORS habilitado para: http://localhost:5173")
        print("\n✨ Endpoints disponibles:")
        print("   GET  /health           - Estado del servidor")
        print("   POST /predict          - Obtener predicción")
        print("   POST /register-round   - Guardar ronda")
        print("   POST /reset            - Reiniciar IA")
        print("\nPresiona Ctrl+C para detener\n")
        print("=" * 60)
        
        app.run(host='0.0.0.0', port=5001, debug=True)
    else:
        print("\n❌ No se pudo inicializar la IA. Verifica que el modelo esté entrenado.")
        print("   Ejecuta primero: python src/modelo.py")
