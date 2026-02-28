#!/bin/bash
# Script para iniciar el servidor de IA
# Uso: ./run_server.sh

echo "🤖 Iniciando servidor de IA..."
echo ""

# Verificar que estamos en el directorio correcto
if [ ! -f "api.py" ]; then
    echo "❌ Error: api.py no encontrado"
    echo "   Ejecuta este script desde el directorio del proyecto"
    exit 1
fi

# Verificar que existe el venv
if [ ! -d ".venv" ]; then
    echo "❌ Error: Virtual environment no encontrado"
    echo "   Crea el venv primero con: python -m venv .venv"
    exit 1
fi

# Activar virtual environment
echo "📦 Activando virtual environment..."
source .venv/bin/activate

# Verificar que Flask está instalado
if ! python -c "import flask" 2>/dev/null; then
    echo "⚠️  Flask no encontrado, instalando dependencias..."
    pip install -r requirements.txt
fi

# Iniciar servidor
echo "🚀 Iniciando servidor Flask..."
echo ""
python api.py
