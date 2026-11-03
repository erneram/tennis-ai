# Serve Lab: informe escrito

CC3092 Deep Learning y Sistemas Inteligentes, Proyecto 3: Casos de Uso de LLM. Ernesto y Hugo (completar nombres y carnés). Entrega: viernes 6 de noviembre de 2026.

Estructura del enunciado (`docs/Proyecto_3_Casos_de_Uso_LLM-1.pdf`, sección 2). La sección 2.3 está escrita. Las demás tienen lo que pide el enunciado y dónde está el material en el repositorio.

## 2.1 Definición del problema y usuario objetivo

Por escribir. Debe responder:

- Quién es el usuario y qué tarea le resuelve el producto. Cómo la resolvía antes.
- Por qué hace falta un LLM y qué alternativas más simples se consideraron.
- Criterios de éxito y qué se mide.

Material: [proposal.md](proposal.md), la sección «El problema» del póster ([poster/poster.pdf](poster/poster.pdf)) y la regla del producto: «una corrección equivocada es peor que una omitida».

## 2.2 Datos y pipeline del sistema

Por escribir. Debe incluir:

- Fuentes, volumen, formato y calidad de los datos.
- Preprocesamiento.
- Diagrama de arquitectura.
- Tokens por solicitud.

Material:

- **Datos:** [data/README.md](../data/README.md). Son 39 videos: 28 de JC (20 analizables) y 11 de profesionales.
- **Pipeline paso a paso:** README, «How one video is analyzed».
- **Diagrama:** sección «Arquitectura» del póster.
- **Tokens:** unos 29 000 por análisis de JC (`experiments/iterations.csv`).

## 2.3 Selección del modelo y técnica de adaptación

### Modelos evaluados

El producto usa **Qwen3-VL-4B-Instruct**, cuantizado a 4 bits para MLX y ejecutado en la Mac del stand (Mac M5 Pro, 24 GB de memoria unificada).

Es un modelo de visión y lenguaje abierto, denso (no MoE) y en versión *instruct*: está ajustado para seguir instrucciones. Eso hace falta porque cada respuesta tiene que ser un JSON con un formato fijo, y un modelo base necesitaría ejemplos o ajuste fino para lograrlo. Tiene licencia Apache-2.0, pesa unos 3 GB y lee varias imágenes en una sola solicitud.

Modelos considerados (licencias revisadas en [licenses.md](licenses.md)):

| Modelo | Licencia | Decisión |
| --- | --- | --- |
| Qwen3-VL-4B-Instruct, 4 bits | Apache-2.0 | Modelo principal. |
| Qwen3-VL-2B-Instruct, 4 bits | Apache-2.0 | Línea base: la mitad del tamaño. |
| Qwen3-VL-8B-Instruct, 4 bits | Apache-2.0 | Probado y descartado (tabla siguiente). |
| Qwen3.5-4B | Apache-2.0 | Candidato, no probado. |
| Gemma 4 E4B | Apache-2.0 | No había versión MLX de 4 bits. |
| Qwen2.5-VL-3B | Qwen Research License | Descartado: solo uso no comercial. |
| Gemma 3 | Gemma Terms of Use | Descartado: política de usos prohibidos y descarga restringida. |
| Llama 3.2 Vision | Llama Community License | Descartado: los modelos multimodales tienen restricciones en la UE. |
| GPT, Claude, Gemini (vía API) | Comerciales | No se usan en el producto (ver abajo). Claude se usó solo como juez para etiquetar el conjunto de evaluación. |

### Comparación de tamaños

Mismo prompt (v17) y misma semilla (0) en los tres tamaños, sobre los mismos 31 videos analizables:

| Modelo | Macro-F1 | Caídas de JC detectadas (de 20) | Falsas alarmas en profesionales (de 11) | Segundos por video de JC | Memoria máxima |
| --- | --- | --- | --- | --- | --- |
| Qwen3-VL-2B | 0.052 | 1 | 9 | 18 | 3.7 GB |
| **Qwen3-VL-4B** | **0.574** | **7** | **0** | 30 | 5.1 GB |
| Qwen3-VL-8B | 0.135 | 0 | 1 | 48 | 7.9 GB |

- **2B:** marca problemas a casi todos, incluidos 9 de los 11 profesionales.
- **8B:** tarda 1.6 veces más y usa 1.5 veces la memoria, y no fue mejor. Con los prompts v6 y v7 también siguió la redacción de la pregunta: marcó a 9 y luego a 11 de los 11 profesionales. El 4B con v6 marcó a 3. Un modelo más grande no corrigió ese sesgo; lo corrigieron los cambios de prompt (v8 en adelante).
- **4B:** es el único que encontró caídas del brazo sin dar falsas alarmas, con una espera aceptable para la feria.

Corridas en `eval/serve/runs/`, cada una con su fila en `experiments/iterations.csv`:

- 2B: `20261102-032818-full` y `20261102-033105-full`.
- 4B: `20261102-023129-full` y `20261102-023612-full`.
- 8B: `20261102-031416-full` y `20261102-032147-full`.
- 8B con v6 y v7: `20261101-pro-v6-8b`, `20261101-pro-v7-8b`, `20261101-jc-crop-v6-8b` y `20261101-jc-crop-v7-8b`.

Límites de esta comparación:

- Los prompts se escribieron con el 4B, así que el 8B podría mejorar con prompts propios.
- Se hizo con una sola semilla. Las versiones de prompt se compararon con 4 semillas porque el puntaje cambia mucho entre semillas (sección 2.5). La diferencia entre tamaños es grande, pero no está medida con esa misma precisión.

### Por qué un modelo abierto en la Mac y no uno de pago por internet

El enunciado permite modelos cerrados vía API (GPT, Claude, Gemini) o abiertos. Elegimos uno abierto que corre en la computadora por cinco razones:

1. **Privacidad.** Los videos muestran a personas: JC, los profesionales y los visitantes que suban su saque en la feria. Con un modelo local, ningún cuadro sale de la computadora. La app solo escucha en `127.0.0.1` y no guarda los videos de los visitantes. Con una API, cada cuadro se enviaría a un tercero.
2. **Funciona sin internet.** Después de la primera descarga, todo corre sin conexión. El enunciado recomienda tener un plan de respaldo para la feria porque «la conexión a internet o una API externa pueden fallar» (recomendación 4). Aquí no hay API que pueda fallar.
3. **Costo cero por análisis.** Un análisis procesa unos 29 000 tokens, con 32 cuadros para las fases, cuadros clave y varias preguntas. Con una API se pagarían esos tokens en cada saque, y en la feria se analizan muchos. En la Mac el costo es $0. La energía no se midió; se estima en 0.3 a 0.5 Wh por análisis (unos 30 s a 40–60 W).
4. **Reproducibilidad.** El modelo es un archivo fijo: no cambia sin aviso, y la semilla y la temperatura están bajo nuestro control. Por eso se pueden repetir las 4 semillas de cada versión y comparar versiones con justicia. Un modelo por API puede actualizarse entre una corrida y otra.
5. **Cabe en el equipo.** Necesita unos 5 GB de los 24 GB de la Mac y analiza un saque en unos 36 s, dentro de la espera razonable para un visitante.

Lo que se pierde: un modelo cerrado grande probablemente distinguiría mejor ángulos finos del cuerpo, que es la principal limitación del 4B. Esto no se midió: ningún modelo cerrado se evaluó como sistema. Además, el producto depende de una Mac con Apple Silicon.

### Dónde está el modelo

Los pesos del modelo **no están en el repositorio**: pesan unos 3 GB y GitHub no está hecho para archivos de ese tamaño. Tampoco hace falta subirlos, porque se descargan de Hugging Face (`mlx-community/Qwen3-VL-4B-Instruct-4bit`) la primera vez que se ejecuta la app y quedan en la caché local. El repositorio guarda todo lo demás: el nombre exacto del modelo en `configs/default.yaml`, los prompts, el código, las corridas y las métricas.

El modelo tampoco está desplegado en un servidor: la app corre en la Mac y se abre en el navegador de esa misma computadora. El enunciado pide el producto «funcionando en vivo en al menos un dispositivo», y ese dispositivo es la Mac del stand.

### Técnica de adaptación

Prompting *zero-shot*: los prompts no incluyen ejemplos resueltos. El modelo se usa tal como se descarga, sin cambiar sus pesos. Es transferencia de aprendizaje a gran escala: un modelo preentrenado con muchísimas imágenes y textos resuelve una tarea nueva solo con instrucciones.

Se combina con tres técnicas:

- **Salida estructurada.** Cada respuesta es un JSON restringido por un esquema (llguidance, dentro de MLX-VLM). El modelo solo puede elegir cuadros, fases, categorías y ejercicios que existen, así que no puede inventar opciones.
- **Varias llamadas pequeñas en vez de una grande.** Cada llamada responde una sola pregunta sobre pocos cuadros. El código toma las decisiones que el modelo hacía mal: ubica el impulso y el aterrizaje a partir del impacto, y elige el punto a mejorar como la fase más temprana con un problema confirmado.
- **Preguntas que se verifican.** Por cada categoría se hacen dos preguntas opuestas, y el chequeo solo cuenta si coinciden. Para el brazo de lanzamiento se hace una pregunta neutral con opciones, porque con sí / no el modelo daba la razón a la pregunta.

No hubo ajuste fino (ni completo ni LoRA). Hay 31 videos analizables de un solo amateur y tres profesionales: son muy pocos para entrenar sin sobreajustar, y las etiquetas sirven solo como clave de evaluación. El ajuste fino con cuadros etiquetados de más jugadores queda como trabajo futuro.

### Diseño de los prompts finales (v22)

El prompt de cada versión es un archivo, `strokes/serve/prompts/v22.yaml`. Su encabezado explica qué cambió respecto a la versión anterior, por qué, y qué resultó. Las llamadas de un análisis son:

| Llamada | Qué recibe | Qué devuelve |
| --- | --- | --- |
| `player` | Primer cuadro, el del medio y el último | Recuadro del jugador, para recortar el video |
| `window` | 32 cuadros del video | Si se puede analizar (o el motivo para grabar de nuevo) y dónde empieza y termina el saque |
| `phases` | 32 cuadros dentro del saque, de 512 px, numerados | Un cuadro por cada una de las seis fases. Si parecen adivinadas, se repite con 1 de cada 2, 3 y 4 cuadros |
| `refine` | 12 cuadros cercanos al impacto | El cuadro exacto del impacto |
| `offsets` | (código, sin modelo) | Impulso 0.1 s antes del impacto, aterrizaje 0.6 s después |
| `checks` | Un cuadro clave de 1 024 px por pregunta | Para cada categoría, un campo `seen` con lo que ve, y luego sí / no / no está claro. Para el brazo de lanzamiento, una opción sobre la parte superior del cuerpo |
| `observation` | El cuadro del punto a mejorar | Hasta 3 frases en español, en tú, solo sobre ese punto |
| `summary` | Los intentos de la sesión | Resumen y enfoque de la siguiente práctica, sin inventar resultados |

Parámetros de generación (`configs/default.yaml`):

| Parámetro | Valor |
| --- | --- |
| `temperature` | 0.2 |
| `top_p` | 0.9 |
| `max_tokens` | 600 (el modelo escribe primero lo que ve y después juzga) |
| `seed` | 0 |

Con temperatura 0 las respuestas eran iguales para cualquier semilla, pero el puntaje fue menor (0.115 con v17 frente a 0.574).

## 2.4 Evaluación y medición de respuestas

Por escribir. Material:

- **Conjunto de evaluación:** 31 casos analizables. Las referencias son de Claude como juez, en `eval/serve/claude_labels.csv`.
- **Métricas y su significado:** README, «What the numbers mean».
- **Líneas base:** el prompt simple y el modelo 2B.

**Pendiente obligatorio:** validar las referencias de Claude contra las etiquetas del equipo en una muestra. Hay que llenar `eval/serve/labels_ernesto.csv` y `labels_hugo.csv` y calcular κ con `uv run python -m eval.run_eval --agreement`.

## 2.5 Resultados de iteraciones

Por escribir. Material:

- **Tabla por versión:** README, «Summary of the work» y «Results».
- **Registro completo:** `experiments/iterations.csv`.
- **Por qué de cada versión:** el encabezado de cada `strokes/serve/prompts/v*.yaml`.
- **Media por semilla:** `uv run python -m experiments.seeds`.

## 2.6 Producto final y experiencia de usuario

Por escribir. Material:

- **Capturas:** [images/app/result.jpg](images/app/result.jpg).
- **Flujo de uso e incertidumbre:** README, sección de la app. La app muestra la confianza, «Pudimos revisar 5 de 6 puntos» y «no se ve claro».
- **Manejo de errores:** pide una nueva grabación cuando el video no se puede analizar.

## 2.7 Limitaciones, riesgos y consideraciones éticas

Por escribir. Puntos que deben estar:

- Las referencias son de un solo juez (Claude) y faltan las del equipo.
- La mitad de prueba se vio al elegir versiones, así que los puntajes son optimistas.
- La tarea es casi binaria: todo JC tiene la misma falla y ningún profesional tiene falla. El modelo podría distinguir a JC de los profesionales por otras cosas.
- Hay un solo amateur, y todos los videos son de lado.
- El 4B no distingue ángulos finos.
- **Privacidad:** el modelo es local, solo se publican imágenes de JC (con consentimiento) y los videos de los visitantes no se guardan.
- **Prompt injection:** las respuestas están restringidas por esquema, y `app/api.py` valida los intentos que vuelven de la página antes de pasarlos al prompt del resumen.

## 2.8 Conclusiones

Por escribir. Resultado final: v22, con macro-F1 0.842 ± 0.084 en 4 semillas, 0 videos sin respuesta y 3 falsas alarmas en 44 saques profesionales.

## 2.9 Enlace al repositorio

https://github.com/erneram/tennis-ai/tree/proyecto-3 (público). El README explica cómo instalar, ejecutar y reproducir las métricas.
