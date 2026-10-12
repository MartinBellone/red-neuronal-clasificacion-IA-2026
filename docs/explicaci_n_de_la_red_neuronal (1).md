# Explicación de la Red Neuronal para Clasificación de Formatos

A lo largo del proyecto, la arquitectura del modelo atravesó distintas etapas de evolución. El objetivo siempre fue maximizar la precisión en la identificación de formatos (PNG, JPG, GIF, WEBP, QOI) a partir de un fragmento de $512$ bytes crudos (que no necesariamente es el comienzo del archivo). 

A continuación se detalla la línea de tiempo del desarrollo y las decisiones arquitectónicas tomadas en cada fase:

## Línea de Tiempo del Desarrollo

### Fase 1: Red Densa y Feature Engineering (El Modelo Base)
* **Enfoque:** Extracción manual de características matemáticas. Se procesaron los bytes crudos para calcular entropía, Transformadas de Fourier y estadísticas descriptivas, generando un vector de $3078$ características.
* **Arquitectura:** Red Neuronal Densa (Multi-Layer Perceptron) optimizada con `keras-tuner`. Uso de capas densas con activación *Swish*, `BatchNormalization` y `Dropout`.
* **Resultado:** Alcanzó un **~85% de exactitud**. Demostró ser un modelo sólido, pero se decidió explorar arquitecturas espaciales para intentar superar este techo.

### Fase 2: Exploración Espacial (Conv1D Pura)
* **Enfoque:** Eliminar el preprocesamiento manual y alimentar a la red directamente con los $512$ bytes crudos normalizados.
* **Arquitectura:** Red Convolucional 1D (Sequential). Se buscaba que los filtros convolucionales encontraran las firmas de los archivos ("magic numbers") por su cuenta.
* **Resultado:** El rendimiento cayó al **~71%**. Se diagnosticó que el uso de capas como `GlobalMaxPooling1D` destruía la información posicional (dónde estaba ubicado cada patrón), volviendo al modelo "ciego" a la estructura del archivo.

### Fase 3: El Modelo Híbrido (Dos Cabezas)
* **Enfoque:** Combinar lo mejor de ambos mundos usando la API Funcional de Keras.
* **Arquitectura:** Una rama procesaba los $512$ bytes crudos (Conv1D) y otra procesaba simultáneamente las $3078$ características enriquecidas (Dense). Ambas ramas se concatenaban antes de la predicción final.
* **Resultado:** Se logró un **~83.4%**. Aunque mejoró sustancialmente respecto a la convolucional pura, no logró superar a la red Densa original. Hubo redundancia de información y un ligero sobreajuste por el exceso de parámetros.

### Fase 4: Experimentos con Convoluciones Dilatadas
* **Enfoque:** Forzar a la rama convolucional a tener un mayor "campo visual" sin sumar parámetros.
* **Arquitectura:** Se agregó `dilation_rate` a las capas `Conv1D` para que los filtros leyeran bytes de forma salteada, junto con `SpatialDropout1D`.
* **Resultado:** Caída drástica al **~68.9%**. Se concluyó matemáticamente que la dilatación rompe la contigüidad estricta necesaria para leer los "magic numbers" (ej. $89 \ 50 \ 4E \ 47$ del PNG).

### Fase 5: Retorno a las Bases 
* **Decisión Final:** Se descartaron las convoluciones y se retornó a la **Red Densa de la Fase 1**.
* **Conclusión:** El proceso iterativo demostró que para este dominio particular, la ingeniería de atributos (el preprocesamiento matemático manual) es inmensamente superior a la extracción automática de características. Exponer la estructura subyacente mediante entropía y frecuencias resultó ser la clave del éxito.

### Fase 6: Optimización de Datos 
Enfoque: Solucionar la "maldición de la dimensionalidad". Ingresar $3078$ variables para clasificar $3600$ archivos generaba demasiado ruido estadístico.
* **Solución:** Poda de Características: Se utilizó un RandomForestClassifier para calcular la importancia de las $3078$ variables matemáticas, filtrando y conservando únicamente el Top 800.Estandarización: Se aplicó StandardScaler sobre las $800$ características restantes para homogeneizar las varianzas, permitiendo que el optimizador AdamW descendiera por el gradiente de forma mucho más limpia.
* **Resultado:** Se logró un pico histórico del 88.20% de exactitud en el test público ciego. La red Densa, ahora alimentada con datos limpios y enfocados, logró un 91% de acierto en la clase JPG y eliminó casi por completo los errores cruzados entre formatos no relacionados.

### Fase 7: Intento de un modelo de ensamble con multiarquitectura
No funciono dado que luego de varias pruebas, aun habiendolo optimizado con el tuner, aumentaba el acurracy en train y test pero al publicarlo dio menor a la marca obtenida en la 4ta entrega/

### Fase 8: Vuelta a las bases otra vez
Gracias al comunicado de la catedra hicimos la red mas simple, con menos capas y neuronas obteniendo el mismo resultado pico de antes de 93.6% pero mejorando un poco el f1 macro.

### Fase 9: Busqueda de mejoras 
Prueba con Tangente Hiperbólica (tanh): Reemplazo de la función swish en la red minimalista. El accuracy local cayó al 87.60%. Conclusión: swish es matemáticamente superior para la distribución de este dataset.

Filtro de Correlación de Pearson: Eliminación manual de variables con más del 90% de similitud para reducir ruido. El rendimiento bajó al 88.60%. Conclusión: El Random Forest ya hace una selección óptima y forzar la eliminación de variables colineales destruye matices necesarios para desempatar clases.

Label Smoothing y Seed Blending: Implementación de predicciones "conservadoras" (suavizado del 10%) promediando 5 semillas de inicialización. Alcanzó un 89.60% local, pero sin superar la red base determinista.

Bypass de Feature Engineering (Datos Crudos): Pase directo de los 512 bytes crudos a la red densa para intentar imitar a los líderes. Colapso al 75.00%. Conclusión: Una red densa no puede entender la contigüidad espacial de los magic bytes por sí sola.

Filtro Cero (Features sin podar): Entrenamiento pasando las 3078 variables generadas por enriquecer_features directo a la red, eliminando el Random Forest. Dio 88.40% local. Conclusión: La red se asfixia con el ruido matemático; la poda a 800 variables es indispensable.

Red Convolucional 1D (CNN): Intento de detectar firmas secuenciales exactas directamente sobre los 512 bytes crudos. Desplome crítico al 69.80%. Conclusión: 4000 archivos de muestra son insuficientes para que una CNN aprenda patrones espaciales desde cero.

Ensamble Heterogéneo (Soft Voting): Combinación estratégica de tres modelos de familias distintas (Red Neuronal Minimalista, XGBoost y SVC) entrenados sobre las 800 variables óptimas. Resolvió gran parte de la confusión entre PNG y WEBP, logrando 93.00% local y disparando el puntaje oficial al 94.80% en el Leaderboard, asegurando el cuarto puesto.

### Fase 10: Análisis de los bytes y features a nivel de bits (Entrega 9)
**Enfoque:** En lugar de seguir probando arquitecturas, se analizó directamente el contenido de los fragmentos para entender qué distingue a cada formato. Este análisis dio tres hallazgos:

* **Casi ningún fragmento empieza en el header:** solo ~1% de las muestras de cada clase arranca con su firma (magic number). Por eso el feature `header_32` (los primeros 32 bytes) era ruido posicional y se eliminó. Esto también explica por qué falló el aumento de datos de la Fase 7: se generaron muestras tomando los primeros 512 bytes de cada archivo, es decir, justo el header, que en los datos reales casi no aparece.
* **JPG y GIF tienen patrones propios del formato:**
    * **JPG:** en los datos comprimidos, todo byte `0xFF` va seguido de `0x00` (*byte stuffing*). Esto ocurre en el 93% de los casos en JPG y en ~0% en el resto de las clases.
    * **GIF:** los datos LZW se guardan en sub-bloques de 255 bytes, así que aparece un `0xFF` cada 256 bytes en el 98% de los GIF.

    Se agregaron como features explícitas, junto con la presencia de las cadenas `IDAT`/`IEND` (PNG) y `VP8` (WEBP). Por sí solas no mejoraron el resultado (92.20%), porque el modelo ya resolvía bien JPG y GIF.
* **PNG y WEBP parecen ruido a nivel de bytes:** el 87% de los PNG y el 98% de los WEBP tienen un histograma de bytes indistinguible del de bytes aleatorios. Ahí estaba casi todo el error del modelo.

**Solución: n-gramas de bits leídos LSB-first.** El compresor de PNG (Deflate) escribe sus códigos Huffman bit a bit, empezando por el bit **menos significativo** de cada byte y sin respetar los límites entre bytes. Por eso, mirando bytes enteros no se ve ninguna estructura. Se reinterpretó cada fragmento como un flujo de $4096$ bits en ese orden, y se calculó el histograma de todas las ventanas de 12 bits consecutivos ($2^{12} = 4096$ features nuevas). Así quedan expuestos los patrones de los códigos Huffman de PNG, mientras que el codificador aritmético de WEBP (VP8) sigue pareciendo aleatorio.

Comparación rápida con un clasificador ExtraTrees separando solo PNG de WEBP:

| Features | Accuracy PNG vs WEBP |
|---|---|
| Histograma de bytes | 67.6% |
| N-gramas de 8 bits (MSB-first) | 83.3% |
| N-gramas de 8 bits (LSB-first) | 89.2% |
| N-gramas de 12 bits (LSB-first) | 92.8% |

Que el orden LSB-first funcione mejor que el MSB-first confirma que la señal viene de la forma en que Deflate empaqueta los bits.

**Pipeline final:** `enriquecer_features` sin `header_32` + n-gramas LSB de 12 bits + 8 features de formato ($7182$ en total). La poda con Random Forest conserva el Top 800, y las 8 features de formato se fuerzan a entrar siempre ($804$ en total). Después se aplica StandardScaler y el mismo ensamble de la Fase 9 (Red Minimalista + XGBoost + SVC con soft voting, 10 folds). Para la entrega final también se sumaron al entrenamiento las 500 muestras etiquetadas de `test_publico.csv` (4500 en total).

**Resultado:**

| | Test público (local) | Leaderboard |
|---|---|---|
| Entrega 8 | 93.00% | 94.80% |
| **Entrega 9** | **97.40%** (F1 macro 97.47%) | **96.80%** (F1 macro 0.9668) |

La validación cruzada sobre las 4500 muestras también dio 97.40%, y el leaderboard quedó a menos de un punto del valor local, lo que confirma que no hay sobreajuste.

**Variantes probadas después de la entrega 9 (sin mejora):**
* **Tamaño de ventana:** con ExtraTrees como referencia, 10 bits dio 89.2%, 12 bits 94.0%, 14 bits 94.4% y 16 bits 93.0%. Combinar 10+12+14 dio 93.4%. 12 y 14 quedan empatados dentro del ruido, así que se mantuvo 12.
* **N-gramas MSB-first de 12 bits (además de los LSB):** JPEG empaqueta sus códigos Huffman empezando por el bit más significativo, y con estas features desaparecieron todos los errores de JPG. Pero en el pipeline completo el resultado fue 97.00% contra 97.40% (2 muestras de diferencia, ruido), porque JPG ya casi no fallaba.
* De los 15 errores que quedan en test público, 13 son PNG ↔ WEBP: fragmentos que ni los patrones de bits logran distinguir. Se consideró que el modelo está cerca de su techo y no se gastaron más entregas.

**Conclusión:** el salto no vino de la arquitectura sino de entender cómo está codificado cada formato. Una vez que la información correcta está en las features, el mismo ensamble que antes llegaba a 93% pasa a 97%.

---
**Arquitectura Final Entregada (Entrega 9):** Ensamble heterogéneo con soft voting (Red Neuronal Densa Minimalista 64-32 con swish + XGBoost + SVC), entrenado con K-Fold de 10 particiones sobre $804$ características: features estadísticas, n-gramas de bits LSB-first y features de formato.