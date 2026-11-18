<!-- PDF (docs/laboratorio-10.pdf), from the repository root, after running xai/explicabilidad.ipynb:
pandoc docs/laboratorio-10.md -s --embed-resources --resource-path=docs:. --css docs/informe.css --metadata pagetitle="Serve Lab: Laboratorio 10, Explainable AI" -o /tmp/laboratorio-10.html
"/Applications/Brave Browser.app/Contents/MacOS/Brave Browser" --headless --no-pdf-header-footer --print-to-pdf=docs/laboratorio-10.pdf /tmp/laboratorio-10.html
-->

<style>
@page { margin: 1.5cm 1.7cm 1.6cm; }
body, table, code, figcaption { font-size: 9pt; line-height: 1.3; }
h1 { font-size: 12.5pt; color: #000; margin: 0 0 0.15em; }
h2 { font-size: 10pt; margin: 0.7em 0 0.25em; }
p { margin: 0.3em 0; }
ul, ol { margin: 0.2em 0; }
table { font-size: 8pt; margin: 0.25em 0 0.4em; }
th, td { padding: 0.1em 0.3em; }
figure { margin: 0.3em 0; }
figcaption { font-size: 8pt; margin-top: 0.1em; }
.head p { margin: 0; color: #3a4650; }
.pair { display: flex; gap: 0.5cm; align-items: flex-start; }
.pair figure { flex: 1; }
</style>

<div class="head">

# Serve Lab: qué mira M1e para decir que el brazo baja antes de tiempo

<p>CC3092 Deep Learning y Sistemas Inteligentes · Laboratorio 10: Explainable AI · Universidad del Valle de Guatemala</p>
<p>Ernesto David Ascencio Ramírez, 23009 · Hugo Daniel Barillas Ajín, 23556 · 20 de noviembre de 2026</p>

</div>

**El modelo explicado.** M1e es el modelo que usa la app y el que defendimos en la presentación del proyecto final.

- **Qué decide:** si en un saque de tenis el brazo que lanza la pelota baja antes de tiempo. Si es así, la app recomienda un ejercicio que vale Q87.50 (15 minutos de entrenador).
- **Qué ve:** no ve el video. Ve 4 alturas sobre el hombro, en largos de torso, durante 64 cuadros (de 1.6 s antes del contacto a 0.5 s después): el codo y la muñeca del brazo que lanza y los del brazo de la raqueta.
- **Cómo decide:** es un ensamble de 5 semillas de una CNN 1-D de 6,497 pesos. Recomienda el ejercicio si la probabilidad media es P ≥ 0.526.
- **Prueba:** 18 saques. Marca los 13 de JC con la falla, pero también los 3 de JC sin ella, y ninguno de los 2 del profesional pro01. Las métricas no dicen por qué marca esos 3 saques limpios; la explicación sí.

## 1. Investigación: métodos de XAI para series de tiempo

| Método | Idea | Requiere | Costo por saque | Limitaciones |
|---|---|---|---|---|
| Oclusión por ventanas | Reemplaza una variable durante un tramo y mide cuánto cambia P | Solo predicciones (caja negra) y una regla de reemplazo | Una predicción por variable × tramo (32) | Depende del tramo y del reemplazo; ignora interacciones; crea entradas fuera de distribución |
| Integrated Gradients (+ TSR) | Integra el gradiente de P desde una línea base hasta la entrada. TSR reescala primero por tiempo y luego por variable | Gradientes y una línea base. TSR necesita una pasada de IG por cuadro | 64 pasadas de ida y vuelta (con TSR, × 64) | Muy sensible a la línea base; ruidoso en series largas |
| TimeSHAP | KernelSHAP sobre eventos, variables y celdas de la secuencia | Caja negra (pensado para RNN) y un conjunto de fondo | Miles de evaluaciones | Aproximación por muestreo; depende del fondo |
| Dynamask | Optimiza una máscara suave tiempo × variable que conserve la predicción | Gradientes y una optimización por instancia | Cientos de iteraciones | Hiperparámetros de área y suavidad |
| WinIT | Importancia de una ventana por el cambio de P al quitarla, con efectos retardados | Un generador entrenado para rellenar la ventana | Entrenar el generador más una evaluación por ventana | Hay que entrenar y validar otro modelo |

La atención (TFT) no aplica, porque M1e es una CNN sin atención.

**Interpretable por diseño frente a explicación a posteriori.**

- **Por diseño**, el modelo mismo es la explicación. Nuestro baseline interpretable es B0, una regla con un umbral: dice "falla" si la muñeca que lanza queda debajo del hombro más de 0.133 s antes del contacto.
  - En prueba vale Q28.43 por análisis, contra Q30.62 de M1e.
  - Pero llama falla a los 12 saques limpios de YouTube del 9 de noviembre, y M1e a ninguno.
- **A posteriori**, se explica un modelo ya entrenado. Esa explicación puede no ser fiel al modelo, así que medimos su fidelidad (figura 4).

**Propiedades formales.**

- **IG, completitud:** las atribuciones suman exactamente P(x) − P(línea base). Así se reparte toda la diferencia con la referencia, pero eso no prueba causalidad.
- **SHAP, aditividad:** las atribuciones suman f(x) − E[f] sobre el fondo.
- **LIME, fidelidad local:** el sustituto lineal es fiel solo cerca de x.

**Parámetros de diseño que cambian la explicación:** en IG, la línea base y el número de pasos; en la oclusión, el tamaño del tramo y el reemplazo; en SHAP, el fondo; en LIME, el kernel y el vecindario; en Grad-CAM, la capa objetivo.

## 2. Selección del método

| Criterio | **Integrated Gradients (elegido)** | Oclusión por tramos (descartada) | TimeSHAP (descartado) |
|---|---|---|---|
| Compatibilidad | Red PyTorch diferenciable, entrada continua | Cualquier modelo; ya está en la app | Pensado para RNN; con una CNN pierde su poda de eventos |
| Tipo de explicación | Por altura × cuadro, con signo y suma exacta | Por altura × tramo de 0.27 s, con signo | Por evento y variable, relativa al fondo |
| Costo (el análisis tarda 3 a 6 s) | 24 ms por saque en CPU | 16 ms por saque | Miles de evaluaciones: segundos (estimado) |
| Comunicación | Se resume en "qué parte y cuándo" | Igual | Hay que explicar el fondo |
| Limitaciones | Depende de la línea base | Sin garantía formal; resolución fija | Fondo de solo 37 saques de entrenamiento |

**Por qué IG:** la completitud separa exactamente cuánto de P viene de cada brazo, resuelve por cuadro y usa la misma referencia que la app, un saque limpio. La oclusión queda como chequeo cruzado.

## 3. Implementación

El notebook `xai/explicabilidad.ipynb` aplica IG (captum 0.9) al M1e ya entrenado, sin reentrenarlo, y regenera todas las figuras y tablas de este reporte.

| Parámetro | Valor | Justificación |
|---|---|---|
| Salida | P media de las 5 semillas | Es la que la app compara con el umbral 0.526 |
| Entrada | 4 alturas × 64 cuadros | Los 4 canales de máscara valen 1 siempre, así que no se atribuyen |
| Línea base | Saque limpio de referencia: la media de los saques sin la falla de entrenamiento (P = 0.472) | Pregunta lo mismo que un entrenador: ¿en qué se aleja este saque de uno limpio? Con ceros (P = 0.661) el mapa cambia por completo (acuerdo de rangos 0.16) |
| Pasos | 64, Gauss-Legendre | Con 32 o 128 sale el mismo mapa (acuerdo 1.00); error de completitud ≤ 0.0004 |
| Fondo y semilla | No hay fondo. Semilla 0 | IG es determinista; la semilla solo fija el orden al azar de la figura 4, y dos corridas dan los mismos CSV |
| Instancias | Elegidas por una regla en el código | Acierto de mayor P, falsa alarma de mayor P, saque más cercano al umbral, falsa alarma más cercana al umbral y profesional de menor P |

<figure><img src="xai/results/local.png" style="max-height: 9.4cm"><figcaption>Figura 1. Explicaciones locales: atribución de IG por altura y cuadro. Naranja empuja hacia "baja antes de tiempo", azul hacia un saque limpio; la línea punteada es el contacto.</figcaption></figure>

| Caso | P | Brazo que lanza | Brazo de la raqueta | Lectura |
|---|---|---|---|---|
| Acierto seguro (jc-s41, con falla) | 0.627 | +0.088 | +0.067 | Muñeca que lanza más baja que en un saque limpio desde 0.8 s antes, sobre todo de 0.35 a 0.2 s |
| Falsa alarma (jc-s36) | 0.579 | +0.075 | +0.032 | Mismo patrón que un acierto: el modelo ve una falla |
| Cerca del umbral (jc-s30, con falla) | 0.554 | +0.058 | +0.025 | La muñeca que lanza empuja, pero en contra cerca del contacto |
| Falsa alarma cerca del umbral (jc-s42) | 0.558 | +0.025 | +0.061 | Lo marca sobre todo el brazo de la raqueta |
| Profesional (pro01_s01) | 0.302 | −0.115 | −0.055 | Todo empuja hacia un saque limpio, sobre todo la muñeca de la raqueta de 0.9 a 0.5 s antes |

<figure><img src="xai/results/global.png" style="max-height: 5.8cm"><figcaption>Figura 2. Explicación global: atribución media de los 18 saques de prueba por grupo.</figcaption></figure>

## 4. Interpretación en el contexto del proyecto

**¿Señales del dominio, atajos, fugas o proxies?** Repartimos la atribución por zona de tiempo y por brazo, en prueba y en 119 saques nuevos que el modelo nunca vio. Las columnas de brazos son medianas de la suma de atribuciones.

| Grupo | Saques | Marcados | Brazo que lanza | Brazo de la raqueta | Postura / lanzamiento / después | Ejercicio con la regla |
|---|---|---|---|---|---|---|
| JC con la falla (prueba) | 13 | 100 % | +0.078 | +0.057 | 11 / 85 / 4 % | 85 % |
| JC sin la falla (prueba) | 3 | 100 % | +0.048 | +0.055 | 5 / 89 / 6 % | 33 % |
| JC después de la sesión (nuevo, sin falla) | 44 | 66 % | +0.012 | +0.048 | 9 / 86 / 6 % | 30 % |
| EA (nuevo, sin falla) | 54 | 2 % | −0.082 | −0.020 | 11 / 80 / 9 % | 0 % |
| Profesionales (prueba y nuevos) | 22 | 0 % | −0.10 a −0.11 | −0.05 a −0.01 | 5–12 / 82–92 / 3–6 % | 0 % |
| YouTube con la falla (nuevo) | 1 | 100 % | +0.150 | −0.092 | 4 / 93 / 3 % | 100 % |

- **No hay fuga en el tiempo.** Solo 3 a 9 % de la atribución cae después del contacto, y lo que pasa después no puede causar la falla.
- **La postura pesa poco.** Los primeros 0.53 s, con el jugador parado, reciben 4 a 12 %: no reconoce a la persona por cómo se para.
- **Usa la señal del dominio.** Lo que más empuja hacia la falla es la muñeca que lanza entre 0.53 y 0.2 s antes del contacto. Es lo que mira un entrenador y la zona de la regla B0.
- **Hay un proxy: el brazo de la raqueta de JC.**
  - Entre 1.07 y 0.53 s antes del contacto empuja hacia la falla en todos los saques de JC, con o sin la falla (+0.05), y en contra en los de los demás.
  - Todas las fallas de entrenamiento son de JC, así que la preparación de su raqueta funciona como su firma.
  - Dentro de JC, el brazo que lanza sí separa la falla: +0.078 con ella, +0.048 sin ella y +0.012 después de la retroalimentación.
  - El único saque con la falla de otro jugador se detecta por el brazo que lanza.

**Errores.** Los 3 errores de prueba son falsas alarmas en JC, de dos tipos:

- **jc-s36:** el brazo que lanza se ve como en una falla (+0.075). Es un caso límite que debería revisar un segundo evaluador ciego, porque las etiquetas fueron de un solo evaluador y no fueron ciegas.
- **jc-s42:** lo marca la firma de la raqueta (+0.061 contra +0.025).

Lo habría corregido tener fallas de otros amateurs, para que la falla no coincida con una sola persona, y más saques limpios de JC.

**Decisiones que cambian gracias a la explicación.**

1. **Regla de revisión "En duda".** Si en un saque marcado el brazo de la raqueta pesa más que el brazo que lanza, la app muestra "En duda" en lugar del ejercicio.
   - En prueba, el recall baja de 1.00 a 0.85 y las falsas alarmas de 0.60 a 0.20. El valor pasa de Q34.30 a Q35.08 por análisis, con las tasas del ensamble (el informe promedió por semilla: Q30.62).
   - En los 44 saques de JC después de la sesión, que no se usaron para elegir la regla, las falsas alarmas bajan de 66 % a 30 %.
   - Se eligió mirando la prueba, así que entra a la app solo después de validarla con datos nuevos.
2. **No recortar el brazo de la raqueta.** Si se reemplaza por el saque limpio, las falsas alarmas en los saques nuevos de JC bajan a 20 %, pero se pierde la mitad de las fallas (recall 0.54). La raqueta lleva señal real, porque la falla se define respecto de ella.
3. **Recolectar** amateurs distintos de JC, con y sin la falla, y sumar un segundo evaluador ciego.

**Cómo se muestra al jugador.** Un mapa de calor no le dice nada a un amateur. La app le muestra la curva de su mano frente a la de un saque limpio, con el momento que decidió sombreado y una frase (figura 3). El entrenador ve el mapa en el detalle técnico.

**¿Un modelo interpretable si el sistema fuera de alto riesgo?** Sí: B0 vale Q28.43 contra Q30.62, con intervalos que se solapan (Q8.48–47.99 y Q20.37–41.11), y se audita por completo. Pero un ejercicio innecesario cuesta 15 minutos y B0 falla con jugadores nuevos, así que aquí conviene M1e con su explicación y la regla.

<div class="pair">
<figure><img src="xai/results/audiencia.png" style="max-height: 4.6cm"><figcaption>Figura 3. Lo que vería el jugador (jc-s41). El tramo sombreado es el de mayor atribución.</figcaption></figure>
<figure><img src="xai/results/fidelidad.png" style="max-height: 4.6cm"><figcaption>Figura 4. Fidelidad: P media de los 16 saques marcados al reemplazar por el saque limpio las k celdas que ordena cada método.</figcaption></figure>
</div>

## 5. Discusión y análisis

**¿Por qué IG y no la oclusión? ¿Cambiaríamos la decisión?**

- Coinciden: el acuerdo de rangos por tramo es 0.90, y las dos bajan P mucho más rápido que el azar (figura 4). Con 32 de 256 celdas reemplazadas siguen marcados 100 % de los saques con el azar, 19 % con IG y 0 % con la oclusión.
- La oclusión gana con pocas celdas porque la prueba usa su misma operación; desde 64 celdas, IG baja P más.
- Para el análisis nos quedamos con IG: la completitud separó la parte de cada brazo, y así apareció el proxy. Para la frase de la app, la oclusión basta.

**¿Qué reveló la explicación que las métricas no?** Las métricas daban recall 1.00 y falsas alarmas 0.60. La explicación dice por qué: el modelo mira el momento correcto, pero suma una firma de JC. Confirma la sospecha de `eval/bias.py` (los modelos distinguen a JC de los profesionales) y la matiza, porque dentro de JC el brazo que lanza sí separa la falla.

**Costo.** IG tarda 24 ms por saque en CPU, frente a 3 a 6 s del análisis completo. Es viable en cada predicción, como pide la regla "En duda".

**Riesgos éticos y de uso indebido.**

- Una explicación convincente de una respuesta equivocada (jc-s36) puede dar falsa seguridad; por eso la app muestra también la confianza.
- IG explica al modelo, no la biomecánica del jugador.
- No debe usarse para calificar, seleccionar o rankear jugadores, ni para decisiones médicas, con menores o con la cámara de frente.
- Los puntos del cuerpo son datos biométricos y no salen de la computadora. EA no dio consentimiento de imagen, así que aparece solo en agregados.

**Conclusiones.**

1. M1e decide con la señal correcta, la muñeca que lanza antes del contacto, sin fugas en el tiempo.
2. Parte de sus falsas alarmas viene de un proxy que ninguna métrica mostraba: la preparación de la raqueta de JC, el único jugador con fallas en el entrenamiento.
3. La explicación cambia tres decisiones: una regla "En duda" (pendiente de validar), no recortar la entrada y priorizar fallas de otros amateurs.

**Repositorio:** <https://github.com/erneram/tennis-ai> (rama `proyecto-final`, carpeta `xai/` y sección "Explainability (Lab 10)" del README).
