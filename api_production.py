"""
API Flask para servir el modelo de IA de Piedra, Papel o Tijera - VERSIÓN PRODUCCIÓN
====================================================================================
Compatible con Render.com y PostgreSQL para persistencia global.

Endpoints:
- GET  /health           - Estado del servidor y modelo
- POST /predict          - Obtener predicción de la IA
- POST /register-round   - Registrar una ronda jugada
- GET  /stats            - Estadísticas globales

Puerto: Variable de entorno PORT (Render.com lo asigna automáticamente)
CORS: Habilitado para cualquier origen en producción
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

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

# CORS: En producción permitimos cualquier origen
# En desarrollo podemos restringir
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*')
if ALLOWED_ORIGINS == '*':
    CORS(app)
else:
    CORS(app, origins=ALLOWED_ORIGINS.split(','))

# URL de la base de datos (Render.com la provee automáticamente)
DATABASE_URL = os.getenv('DATABASE_URL')

# Instancia global de la IA
ia_jugador: Optional['JugadorIA'] = None

# ============================================
# BASE DE DATOS
# ============================================

def get_db_connection():
    """Obtiene conexión a PostgreSQL."""
    if not DATABASE_URL:
        return None
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        print(f"❌ Error conectando a PostgreSQL: {e}")
        return None


def init_db():
    """Crea la tabla de partidas si no existe."""
    conn = get_db_connection()
    if not conn:
        print("⚠️  PostgreSQL no disponible, usando modo sin persistencia")
        return False
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS partidas (
                id SERIAL PRIMARY KEY,
                ronda INTEGER,
                jugador_ia VARCHAR(10),
                jugador_humano VARCHAR(10),
                ganador VARCHAR(20),
                tiempo_ia REAL DEFAULT 0.0,
                tiempo_humano REAL DEFAULT 0.0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Tabla 'partidas' lista en PostgreSQL")
        return True
    except Exception as e:
        print(f"❌ Error creando tabla: {e}")
        if conn:
            conn.close()
        return False


def save_round_to_db(player: str, ai: str, winner: str):
    """Guarda una ronda en PostgreSQL."""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Obtener número de ronda
        cursor.execute("SELECT COALESCE(MAX(ronda), 0) + 1 FROM partidas")
        ronda = cursor.fetchone()[0]
        
        # Mapear ganador
        ganador_db = {
            'ai': 'Jugador 1',
            'player': 'Jugador 2',
            'tie': 'Empate'
        }.get(winner, 'Empate')
        
        cursor.execute("""
            INSERT INTO partidas (ronda, jugador_ia, jugador_humano, ganador)
            VALUES (%s, %s, %s, %s)
        """, (ronda, ai, player, ganador_db))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"💾 Ronda {ronda} guardada en PostgreSQL")
        return True
    except Exception as e:
        print(f"❌ Error guardando ronda: {e}")
        if conn:
            conn.close()
        return False


def get_total_rounds():
    """Obtiene el total de rondas jugadas."""
    conn = get_db_connection()
    if not conn:
        return 0
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM partidas")
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return count
    except Exception as e:
        print(f"❌ Error obteniendo total: {e}")
        if conn:
            conn.close()
        return 0


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


# ============================================
# ENDPOINTS
# ============================================

@app.route('/health', methods=['GET'])
def health():
    """Verifica el estado del servidor y modelo."""
    total_rounds = get_total_rounds() if DATABASE_URL else 0
    
    return jsonify({
        'status': 'ok',
        'modelLoaded': ia_jugador is not None,
        'totalRoundsPlayed': total_rounds,
        'databaseConnected': DATABASE_URL is not None,
        'environment': 'production' if os.getenv('RENDER') else 'development'
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
        
        # Logs para debugging (solo en desarrollo)
        if not os.getenv('RENDER'):
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
        return jsonify({
            'error': f'Error interno: {str(e)}'
        }), 500


@app.route('/register-round', methods=['POST'])
def register_round():
    """
    Registra una ronda jugada en PostgreSQL.
    
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
        
        # Guardar en PostgreSQL
        success = save_round_to_db(player, ai, winner)
        
        if success:
            total = get_total_rounds()
            return jsonify({
                'success': True,
                'roundNumber': total
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No se pudo guardar en base de datos'
            }), 500
        
    except Exception as e:
        print(f"❌ Error en /register-round: {e}")
        return jsonify({
            'error': f'Error interno: {str(e)}'
        }), 500


@app.route('/stats', methods=['GET'])
def stats():
    """Obtiene estadísticas globales de todas las partidas."""
    conn = get_db_connection()
    if not conn:
        return jsonify({
            'error': 'Base de datos no disponible'
        }), 503
    
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Total de rondas
        cursor.execute("SELECT COUNT(*) as total FROM partidas")
        total = cursor.fetchone()['total']
        
        # Victorias por jugador
        cursor.execute("""
            SELECT ganador, COUNT(*) as victorias
            FROM partidas
            GROUP BY ganador
        """)
        victorias = {row['ganador']: row['victorias'] for row in cursor.fetchall()}
        
        # Jugadas más comunes del humano
        cursor.execute("""
            SELECT jugador_humano as jugada, COUNT(*) as veces
            FROM partidas
            GROUP BY jugador_humano
            ORDER BY veces DESC
            LIMIT 3
        """)
        jugadas_comunes = [dict(row) for row in cursor.fetchall()]
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'totalRounds': total,
            'victories': victorias,
            'commonMoves': jugadas_comunes
        })
        
    except Exception as e:
        print(f"❌ Error en /stats: {e}")
        if conn:
            conn.close()
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
    print("   🤖 SERVIDOR DE IA - PRODUCCIÓN")
    print("=" * 60)
    
    # Inicializar base de datos
    if DATABASE_URL:
        print(f"🗄️  PostgreSQL: {DATABASE_URL[:50]}...")
        init_db()
    else:
        print("⚠️  Sin PostgreSQL - modo desarrollo")
    
    # Inicializar IA
    if inicializar_ia():
        port = int(os.getenv('PORT', 5001))
        print(f"\n🚀 Servidor corriendo en puerto {port}")
        print(f"🔗 CORS: {ALLOWED_ORIGINS}")
        print("\n✨ Endpoints disponibles:")
        print("   GET  /health           - Estado del servidor")
        print("   POST /predict          - Obtener predicción")
        print("   POST /register-round   - Guardar ronda")
        print("   GET  /stats            - Estadísticas globales")
        print("   POST /reset            - Reiniciar IA")
        print("\nPresiona Ctrl+C para detener\n")
        print("=" * 60)
        
        # En producción usar gunicorn, en desarrollo Flask
        if os.getenv('RENDER'):
            # Render.com usa gunicorn automáticamente
            app.run(host='0.0.0.0', port=port)
        else:
            app.run(host='0.0.0.0', port=port, debug=True)
    else:
        print("\n❌ No se pudo inicializar la IA. Verifica que el modelo esté entrenado.")
        print("   Ejecuta primero: python src/modelo.py")
