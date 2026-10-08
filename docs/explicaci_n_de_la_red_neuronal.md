# Explicación oral de la red neuronal de clasificación

## Línea de tiempo del proyecto

### 1. Punto de partida: entender el problema
La competencia pide reconocer el formato de una imagen a partir de un fragmento de 512 bytes. Las clases son `gif`, `jpg`, `png`, `qoi` y `webp`.
El dataset tiene 4000 filas balanceadas. Cada fila contiene 512 valores y una etiqueta textual en la última columna.

### 2. Primera carga del dataset
Inicialmente la notebook estaba preparada para MNIST. Reemplazamos la carga por una lectura de `dataset.csv`.
Separamos los 512 valores de la etiqueta y convertimos las clases usando un orden fijo: `{'gif': 0, 'jpg': 1, 'png': 2, 'qoi': 3, 'webp': 4}`.

### 3. Preprocesamiento inicial
Dividimos el dataset en entrenamiento y validación usando `train_test_split` con `stratify`, y normalizamos los bytes dividiendo por 255. La primera representación usaba solamente los 512 valores originales.

### 4. Primera red feedforward y diagnóstico
Construimos una red con capas `Dense`, ReLU y salida softmax. La matriz de confusión mostró que `gif` y `qoi` eran fáciles de reconocer, pero había confusiones severas entre `jpg`, `png` y `webp`. Al estar limitados a usar solo redes feedforward (sin capas convolucionales), la red carecía de la capacidad de comprender secuencias espaciales.

### 5. Primer enriquecimiento de features
Para darle más contexto a la red, calculamos:
- Histograma global de los valores 0-255.
- Media, desvío estándar, mínimo y máximo.
- Histogramas locales dividiendo la muestra en 8 bloques de 64 bytes.

### 6. Control del entrenamiento y refinamiento
Agregamos `EarlyStopping` y `ReduceLROnPlateau`. Probamos diferentes arquitecturas y agregamos histogramas de transiciones cuantizadas, alcanzando un techo aproximado del 68% en el test final ciego. La barrera persistía: PNG y WEBP se seguían confundiendo porque ambos se veían como "ruido aleatorio" de alta compresión.

### 7. Extracción de Features Avanzadas y eliminación de ruido (El gran salto)
Nos dimos cuenta de que las capas *Dense* sufren de falta de invarianza espacial (un mismo patrón desplazado confunde a la red). Por ende, tomamos una decisión radical: **eliminar los 512 bytes originales de la entrada** y dejar que la red trabaje solo con estadística pura. Agregamos:
- **Entropía de Shannon:** Para medir la densidad matemática de la compresión.
- **Diferencias absolutas consecutivas:** Los "Filtros PNG" restan bytes consecutivos. Hacer un histograma de esta diferencia creó una firma evidente para cazar PNGs.
- **Transformada Rápida de Fourier (FFT):** Para darle a la red "ojos" que detecten el espectro de frecuencias espaciales del fragmento.
La entrada quedó limpia y consolidada en **3078 características**.

### 8. Optimizador y Arquitectura de Nivel Competencia
Para evitar que la red asfixiara neuronas y sobreajustara en las imágenes confusas, aplicamos tres técnicas avanzadas:
- Cambio de activación de `ReLU` a `Swish` (para mantener gradientes en valores negativos).
- `Label Smoothing` de 0.1 en la función de pérdida.
- Optimizador `AdamW` (learning_rate=0.001, weight_decay=0.004) para una regularización inteligente de los pesos.

### 9. Pesos dinámicos y Mega-Ensamble K-Fold
La red era "perezosa" y ganaba accuracy fácil acertando GIFs. Aplicamos un diccionario de `class_weight` que multiplicaba x3 el error al equivocarse en PNG y WEBP, obligando al modelo a aprender la diferencia difícil.
Finalmente, pasamos de usar un solo modelo a un **Ensamble de 10 divisiones (10-Fold)**. Entrenamos 10 redes diferentes (cada una con el 90% de los datos) y promediamos sus probabilidades de predicción, eliminando la varianza de inicialización.

### 10. Resultado actual
Tras estas optimizaciones de fuerza bruta matemática, rompimos el techo de cristal con los siguientes resultados:
```text
K-Fold (Validación cruzada): 83.60%
Dataset (Entrenamiento):     85.00%
Test público:                82.40%
Test final (ciego):          74.60%
```

---

## Resumen para la exposición

Una forma breve de explicarlo oralmente es:

> "Iniciamos separando las etiquetas y normalizando los bytes. Al notar en la matriz de confusión que el modelo Feedforward chocaba al intentar distinguir PNG de WEBP (debido a la alta entropía de ambos), decidimos eliminar los 512 bytes crudos de la entrada, ya que las capas Dense no manejan bien los desplazamientos espaciales. En su lugar, construimos 3078 características puramente estadísticas: histogramas globales y por bloques, entropía de Shannon, un histograma de diferencias absolutas (para detectar los filtros de escaneo de PNG) y magnitudes de la Transformada de Fourier. 
> 
> Usamos una arquitectura `3078 -> 256 -> 128 -> 5` usando la activación `Swish`, BatchNormalization y Dropout. Para estabilizar el aprendizaje complejo, utilizamos el optimizador `AdamW` y aplicamos `Label Smoothing`. Además, introdujimos pesos de clase que penalizan el triple los errores en PNG y WEBP.
> 
> Finalmente, para la evaluación usamos un Mega-Ensamble K-Fold de 10 divisiones. Entrenamos 10 modelos distintos y los hicimos votar promediando sus probabilidades de predicción sobre el set de test. Esto neutralizó los sesgos individuales y nos disparó de un estancamiento del 68% hasta un **74.6% de precisión en el test final ciego**."


