# Explicación de la Red Neuronal para Clasificación de Formatos

A lo largo del proyecto, la arquitectura del modelo atravesó distintas etapas de evolución. El objetivo siempre fue maximizar la precisión en la identificación de formatos (PNG, JPG, GIF, WEBP, QOI) a partir de sus primeros $512$ bytes crudos. 

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

### Fase 5: Retorno a las Bases (Arquitectura Definitiva)
* **Decisión Final:** Se descartaron las convoluciones y se retornó a la **Red Densa de la Fase 1**.
* **Conclusión:** El proceso iterativo demostró que para este dominio particular, la ingeniería de atributos (el preprocesamiento matemático manual) es inmensamente superior a la extracción automática de características. Exponer la estructura subyacente mediante entropía y frecuencias resultó ser la clave del éxito.

Fase 6: Optimización de Datos (La Arquitectura Definitiva)
Enfoque: Solucionar la "maldición de la dimensionalidad". Ingresar $3078$ variables para clasificar $3600$ archivos generaba demasiado ruido estadístico.Solución:Poda de Características: Se utilizó un RandomForestClassifier para calcular la importancia de las $3078$ variables matemáticas, filtrando y conservando únicamente el Top 800.Estandarización: Se aplicó StandardScaler sobre las $800$ características restantes para homogeneizar las varianzas, permitiendo que el optimizador AdamW descendiera por el gradiente de forma mucho más limpia.
Resultado: Se logró un pico histórico del 88.20% de exactitud en el test público ciego. La red Densa, ahora alimentada con datos limpios y enfocados, logró un 91% de acierto en la clase JPG y eliminó casi por completo los errores cruzados entre formatos no relacionados.

---
**Arquitectura Final Entregada:** Modelo Secuencial (Dense), entrada de $3078$ características, optimizado con hiperparámetros de Keras Tuner.