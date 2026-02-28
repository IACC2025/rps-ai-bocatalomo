# 🚀 Guía de Deployment - API de IA en Render.com

Esta guía explica cómo desplegar el servidor Flask de IA en Render.com para que funcione con tu app Tamagotchi en Vercel.

---

## 📋 Pre-requisitos

1. ✅ Cuenta en [Render.com](https://render.com) (gratis)
2. ✅ Cuenta en [GitHub](https://github.com)
3. ✅ Git instalado en tu computadora

---

## 🎯 Paso 1: Crear Repositorio en GitHub

### Opción A: Desde GitHub Web

1. Ve a https://github.com/new
2. Nombre del repo: `rps-ia-api` (o el que prefieras)
3. Descripción: "API Flask para modelo de IA de Piedra, Papel o Tijera"
4. Visibilidad: **Public** (Render.com gratis solo funciona con repos públicos)
5. ❌ NO marques "Add README" ni ".gitignore" ni "license"
6. Haz clic en "Create repository"

### Opción B: Desde Terminal

```bash
# Ir al directorio del proyecto
cd /Users/diego/PycharmProjects/rps-ai-bocatalomo

# Inicializar repositorio Git
git init

# Agregar todos los archivos
git add .

# Hacer commit inicial
git commit -m "feat: API Flask con modelo de IA entrenado"

# Crear repositorio en GitHub (usando GitHub CLI)
gh repo create rps-ia-api --public --source=. --remote=origin --push
```

Si no tienes GitHub CLI:
```bash
# Crear repo manualmente en GitHub y luego:
git remote add origin https://github.com/TU_USUARIO/rps-ia-api.git
git branch -M main
git push -u origin main
```

---

## 🌐 Paso 2: Desplegar en Render.com

### 2.1 Conectar con GitHub

1. Ve a https://dashboard.render.com
2. Haz clic en **"New +"** → **"Blueprint"**
3. Conecta tu cuenta de GitHub (si no lo has hecho)
4. Dale permiso a Render para acceder a tus repositorios

### 2.2 Seleccionar Repositorio

1. Busca `rps-ia-api` en la lista
2. Haz clic en **"Connect"**

### 2.3 Configurar Servicio

Render detectará automáticamente el archivo `render.yaml` y creará:

- ✅ **Web Service** para la API Flask
- ✅ **PostgreSQL Database** para persistencia

**Configuración automática:**
- Name: `rps-ia-api`
- Region: Frankfurt (o la que elegiste)
- Branch: `main`
- Build Command: `pip install -r requirements-prod.txt`
- Start Command: `gunicorn api_production:app`

### 2.4 Variables de Entorno

Render.com automáticamente asignará:

- `DATABASE_URL` - Conexión a PostgreSQL (automática)
- `PORT` - Puerto del servidor (automático)

**Agregar manualmente:**

1. En el dashboard, ve a tu servicio → **"Environment"**
2. Agrega esta variable:

| Key | Value |
|-----|-------|
| `ALLOWED_ORIGINS` | `https://tu-app.vercel.app,http://localhost:5173` |

**Nota:** Cambia `tu-app.vercel.app` por la URL de tu app en Vercel.

### 2.5 Desplegar

1. Haz clic en **"Create Blueprint"** o **"Manual Deploy"** → **"Deploy"**
2. Espera 5-10 minutos mientras Render:
   - Clona el repositorio
   - Instala dependencias
   - Crea la base de datos PostgreSQL
   - Inicia el servidor

### 2.6 Verificar Deployment

Una vez completado, verás:

✅ Estado: **"Live"**  
✅ URL: `https://rps-ia-api.onrender.com` (o similar)

**Probar la API:**
```bash
curl https://rps-ia-api.onrender.com/health
```

Deberías ver:
```json
{
  "status": "ok",
  "modelLoaded": true,
  "totalRoundsPlayed": 0,
  "databaseConnected": true,
  "environment": "production"
}
```

---

## ⚛️ Paso 3: Conectar con Vercel (App React)

### 3.1 Actualizar Variables de Entorno en Vercel

1. Ve a tu proyecto en Vercel
2. **Settings** → **Environment Variables**
3. Agrega:

| Name | Value | Environment |
|------|-------|-------------|
| `VITE_AI_API_URL` | `https://rps-ia-api.onrender.com` | Production, Preview, Development |

### 3.2 Re-desplegar

```bash
# Si ya tienes la app en Vercel
vercel --prod

# O desde el dashboard de Vercel
# Settings → Deployments → Redeploy
```

---

## 🎮 Paso 4: Probar en Producción

1. Abre tu app en Vercel: `https://tu-app.vercel.app`
2. Navega al juego "🤖 Piedra, Papel o Tijera IA"
3. Deberías ver "🤖 IA conectada" (indicador verde)
4. Juega unas rondas
5. Verifica que se guarden en la base de datos:

```bash
curl https://rps-ia-api.onrender.com/stats
```

---

## 📊 Monitoreo y Logs

### Ver Logs en Tiempo Real

1. En Render dashboard → Tu servicio → **"Logs"**
2. Verás peticiones, predicciones y errores

### Métricas

1. **"Metrics"** → Uso de CPU, memoria, peticiones

### Base de Datos

1. **"Dashboard"** → **"rps-ia-db"** → **"Connect"**
2. Usa las credenciales para conectarte con un cliente PostgreSQL

---

## 🔄 Actualizaciones Automáticas

Render.com se actualiza automáticamente cuando haces push a GitHub:

```bash
# Hacer cambios al código
git add .
git commit -m "feat: Mejorar predicciones de IA"
git push origin main

# Render.com detecta el push y re-despliega automáticamente
```

---

## 💰 Costos

**Render.com Free Tier:**
- ✅ Web Services: Gratis (con limitaciones)
  - Se duerme después de 15 min de inactividad
  - Tarda ~30s en despertar al recibir petición
- ✅ PostgreSQL: 90 días gratis, luego $7/mes
- ✅ 750 horas/mes de servicio activo

**Recomendación:** Para desarrollo/demo es perfecto gratis. Para producción seria considera el plan de pago ($7/mes).

---

## ⚠️ Solución de Problemas

### Error: "Build failed"

**Solución:**
1. Verifica que `requirements-prod.txt` esté en el repo
2. Revisa los logs de build en Render
3. Asegúrate que `src/modelo.py` y `models/modelo_entrenado.pkl` estén en el repo

### Error: "Database connection failed"

**Solución:**
1. Verifica que la variable `DATABASE_URL` esté configurada
2. Espera a que PostgreSQL termine de inicializarse
3. Revisa logs: `psycopg2.OperationalError`

### La IA responde lento (>10s)

**Causa:** El servicio se durmió (free tier)  
**Solución:**
- Actualiza a plan de pago ($7/mes)
- O acepta el delay inicial (es normal en free tier)

### CORS Error desde Vercel

**Solución:**
1. Verifica que `ALLOWED_ORIGINS` incluya la URL de Vercel
2. Agrega: `https://tu-app.vercel.app`
3. Re-despliega el servicio

---

## 🔒 Seguridad

### Variables Sensibles

**NUNCA** incluyas en el código:
- ❌ Claves API
- ❌ Contraseñas de base de datos
- ❌ Tokens

Usa **Environment Variables** en Render.com.

### CORS

En producción, limita CORS a tu dominio:
```python
ALLOWED_ORIGINS = "https://tu-app.vercel.app"
```

---

## 📈 Próximos Pasos

Una vez desplegado:

1. ✅ Compartir URL de Vercel con amigos/usuarios
2. ✅ Monitorear estadísticas: `GET /stats`
3. ✅ Ver cómo la IA mejora con más partidas jugadas
4. ✅ Considerar reentrenar el modelo con datos de producción

---

## 📞 Soporte

- **Render Docs:** https://render.com/docs
- **Community:** https://community.render.com
- **Status:** https://status.render.com

---

¡Listo! Tu modelo de IA ahora está en producción y accesible desde cualquier lugar 🚀
