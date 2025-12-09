# Documentacion de Recogida de Datos

**Alumno:** Diego Sánchez Cano 

## Formato del CSV

Tu archivo `data/partidas.csv` debe tener **minimo** estas columnas:

| Columna | Descripcion | Ejemplo |
|---------|-------------|---------|
| `numero_ronda` | Numero de la ronda (1, 2, 3...) | 1 |
| `jugada_j1` | Jugada del jugador 1 | piedra |
| `jugada_j2` | Jugada del jugador 2 (oponente) | papel |

### Ejemplo de CSV minimo:

```csv
numero_ronda,jugada_j1,jugada_j2
1,piedra,papel
2,tijera,piedra
3,papel,papel
4,piedra,tijera
...
```

---

## Como recogiste los datos?

Marca con [x] el metodo usado y describe brevemente:

### Metodo de recogida:

- [x] **Programa propio**: Cree un programa para jugar y guardar datos
- [ ] **Manual**: Jugue partidas y apunte los resultados a mano
- [ ] **Aplicacion/Web externa**: Use una app y exporte los datos
- [ ] **Otro**: _________________

### Descripcion del proceso:

```
Para recoger los datos hize un pequeño programa en python que recopilaba
los datos de las partidas a medida que se iban jugando. 



```
---

## Datos adicionales capturados

Si capturaste datos extra ademas de los basicos, marcalos aqui:

- [X] `tiempo_reaccion_ms` - Tiempo que tardo el jugador en responder
- [ ] `timestamp` - Fecha/hora de cada jugada
- [ ] `sesion` - ID de sesion de juego
- [ ] `resultado` - victoria/derrota/empate
- [X] Otro: Número de ronda

### Descripcion de datos adicionales:

```
Al inicio utilizaba el tiempo de reacción en milisegundos en alguna 
feature, ej: si elige tarde tiende a jugar papel, etc.
 
Luego me di cuenta que esas features que tenían en cuenta el tiempo 
de reacción solo hacían ruido y no aportaban precisión, así que las 
descarté 

```

---

## Estadisticas del dataset

- **Total de rondas:** 789
- **Numero de sesiones/partidas:** 52 sesiones de 15 rondas + 1 sesion de 8 rondas
- **Contra cuantas personas diferentes:** más de 10 personas diferentes

### Tipo de IA:

- [ ] **IA Especifica**: Entrenada para ganar a UNA persona concreta
  - Nombre/identificador del oponente: _________________
- [x] **IA General**: Entrenada para ganar a cualquier oponente
