# 🤖 Servidor de IA - Piedra, Papel o Tijera

API REST para servir el modelo de machine learning entrenado.

## 🚀 Inicio Rápido

### Método 1: Script automático

```bash
./run_server.sh
```

### Método 2: Manual

```bash
# Activar virtual environment
source .venv/bin/activate

# Iniciar servidor
python api.py
```

El servidor estará disponible en: **http://localhost:5001**

---

## 📡 Endpoints Disponibles

### `GET /health`

Verifica el estado del servidor y modelo.

**Respuesta:**
```json
{
  "status": "ok",
  "modelLoaded": true,
  "totalRoundsPlayed": 1234,
  "csvPath": "/path/to/partidas_web.csv"
}
```

---

### `POST /predict`

Obtiene la predicción de la IA sobre la próxima jugada.

**Request:**
```json
{
  "history": {
    "player": ["piedra", "tijera", "papel"],
    "ai": ["papel", "piedra", "tijera"]
  }
}
```

**Response:**
```json
{
  "predictedOpponentMove": "piedra",
  "aiMove": "papel",
  "reasoning": "Basado en 3 rondas previas"
}
```

---

### `POST /register-round`

Registra una ronda jugada en el CSV persistente.

**Request:**
```json
{
  "player": "piedra",
  "ai": "papel",
  "winner": "ai"
}
```

**Response:**
```json
{
  "success": true,
  "roundNumber": 1235
}
```

---

### `POST /reset`

Reinicia el historial de la IA (útil para testing).

**Response:**
```json
{
  "success": true,
  "message": "IA reiniciada"
}
```

---

## 📂 Archivos Generados

### `data/partidas_web.csv`

Almacena **todas** las partidas jugadas desde la web. El formato es:

```csv
Ronda,Jugador 1,Jugador 2,Ganador,Tiempo Jugador 1,Tiempo Jugador 2,Timestamp
1,papel,piedra,Jugador 1,0.0,0.0,2026-02-28T10:30:45
```

- **Jugador 1**: IA
- **Jugador 2**: Humano (usuario web)
- **Timestamp**: Fecha/hora ISO de la jugada

---

## 🔧 Configuración

### Puerto

Por defecto: **5001**

Para cambiar el puerto, edita `api.py` línea final:

```python
app.run(host='0.0.0.0', port=5001, debug=True)
```

### CORS

Permitido por defecto para:
- `http://localhost:5173` (Vite dev)
- `http://localhost:5174` (Vite alternate)

Para agregar más orígenes, edita `api.py`:

```python
CORS(app, origins=["http://localhost:5173", "http://tu-dominio.com"])
```

---

## 🧪 Testing

### Con curl

```bash
# Health check
curl http://localhost:5001/health

# Predicción (primera jugada)
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"history": {"player": [], "ai": []}}'

# Registrar ronda
curl -X POST http://localhost:5001/register-round \
  -H "Content-Type: application/json" \
  -d '{"player": "piedra", "ai": "papel", "winner": "ai"}'
```

### Con Python

```python
import requests

# Health check
r = requests.get('http://localhost:5001/health')
print(r.json())

# Predicción
r = requests.post('http://localhost:5001/predict', json={
    'history': {
        'player': ['piedra'],
        'ai': ['papel']
    }
})
print(r.json())
```

---

## 🔍 Debugging

El servidor loguea todas las peticiones en consola:

```
📊 Historial: 3 rondas
🤔 IA predice que humano jugará: piedra
🎲 IA decide jugar: papel
```

Para más información, revisa los logs en tiempo real mientras el servidor corre.

---

## ⚠️ Problemas Comunes

### Puerto ocupado

Si el puerto 5001 está ocupado:

```bash
# Ver qué proceso usa el puerto
lsof -i :5001

# Cambiar puerto en api.py o matar el proceso
kill -9 <PID>
```

### Modelo no carga

```bash
# Verificar que el modelo existe
ls models/modelo_entrenado.pkl

# Si no existe, entrenar el modelo primero
python src/modelo.py
```

### CORS error desde navegador

Asegúrate de que el origen esté en la lista de CORS en `api.py`.

---

## 📊 Estadísticas

El CSV acumula todas las partidas. Para ver estadísticas:

```bash
# Total de rondas
wc -l data/partidas_web.csv

# Últimas 10 rondas
tail -10 data/partidas_web.csv

# Contar victorias de IA
grep "Jugador 1" data/partidas_web.csv | wc -l
```

---

## 🔄 Integración con Tamagotchi App

La app React se comunica automáticamente con este servidor cuando:

1. Servidor está corriendo en puerto 5001
2. App detecta servidor disponible (endpoint `/health`)
3. Usuario juega "Piedra, Papel o Tijera IA"

Si el servidor no está disponible, la app usa IA simple local automáticamente.

---

## 📝 Licencia

Mismo que el proyecto principal.
