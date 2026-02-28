#!/bin/bash
# Script de prueba para verificar que la API funciona correctamente

echo "🧪 Probando API de IA..."
echo ""

API_URL="http://localhost:5001"

# Test 1: Health check
echo "1️⃣  Test: GET /health"
response=$(curl -s "$API_URL/health")
if echo "$response" | grep -q "\"modelLoaded\": true"; then
    echo "   ✅ Servidor OK - Modelo cargado"
else
    echo "   ❌ Error: Servidor no responde correctamente"
    echo "   Respuesta: $response"
    exit 1
fi
echo ""

# Test 2: Primera predicción (sin historial)
echo "2️⃣  Test: POST /predict (sin historial)"
response=$(curl -s -X POST "$API_URL/predict" \
    -H "Content-Type: application/json" \
    -d '{"history": {"player": [], "ai": []}}')
if echo "$response" | grep -q "aiMove"; then
    echo "   ✅ Predicción recibida"
    echo "   $response" | python -m json.tool 2>/dev/null | grep -E "(aiMove|predictedOpponentMove)" || echo "   $response"
else
    echo "   ❌ Error en predicción"
    echo "   Respuesta: $response"
fi
echo ""

# Test 3: Predicción con historial
echo "3️⃣  Test: POST /predict (con historial)"
response=$(curl -s -X POST "$API_URL/predict" \
    -H "Content-Type: application/json" \
    -d '{"history": {"player": ["piedra", "papel"], "ai": ["tijera", "piedra"]}}')
if echo "$response" | grep -q "aiMove"; then
    echo "   ✅ Predicción con historial OK"
    echo "   $response" | python -m json.tool 2>/dev/null | grep -E "(aiMove|predictedOpponentMove)" || echo "   $response"
else
    echo "   ❌ Error en predicción con historial"
    echo "   Respuesta: $response"
fi
echo ""

# Test 4: Registrar ronda
echo "4️⃣  Test: POST /register-round"
response=$(curl -s -X POST "$API_URL/register-round" \
    -H "Content-Type: application/json" \
    -d '{"player": "piedra", "ai": "papel", "winner": "ai"}')
if echo "$response" | grep -q "\"success\": true"; then
    echo "   ✅ Ronda registrada correctamente"
    echo "   $response" | python -m json.tool 2>/dev/null || echo "   $response"
else
    echo "   ❌ Error al registrar ronda"
    echo "   Respuesta: $response"
fi
echo ""

# Test 5: Verificar CSV
echo "5️⃣  Test: Verificar CSV creado"
if [ -f "data/partidas_web.csv" ]; then
    lines=$(wc -l < data/partidas_web.csv)
    echo "   ✅ CSV existe con $lines líneas"
    echo "   Últimas 3 líneas:"
    tail -3 data/partidas_web.csv | sed 's/^/      /'
else
    echo "   ❌ CSV no encontrado"
fi
echo ""

echo "✅ Todos los tests completados!"
