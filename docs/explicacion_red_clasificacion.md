# Explicacion oral de la red neuronal de clasificacion

## Linea de tiempo del proyecto

### 1. Punto de partida: entender el problema

La competencia pide reconocer el formato de una imagen a partir de un fragmento de 512 bytes. Las clases son `gif`, `jpg`, `png`, `qoi` y `webp`.

El dataset tiene 4000 filas balanceadas. Cada fila contiene 512 valores y una etiqueta textual en la ultima columna.

### 2. Primera carga del dataset

Inicialmente la notebook estaba preparada para MNIST, con imagenes de 28x28 y etiquetas numericas. Eso no correspondia con este problema, asi que reemplazamos la carga por una lectura de `dataset.csv`.

Separamos los 512 valores de la etiqueta y convertimos las clases usando un orden fijo:

```python
{'gif': 0, 'jpg': 1, 'png': 2, 'qoi': 3, 'webp': 4}
```

Tambien agregamos `strip()` porque las etiquetas del archivo podian contener espacios.

### 3. Preprocesamiento inicial

Dividimos el dataset en entrenamiento y validacion usando `train_test_split` con `stratify`, y normalizamos los bytes dividiendo por 255.

La primera representacion usaba solamente los 512 valores originales.

### 4. Primera red feedforward

Construimos una red totalmente conectada con capas `Dense`, ReLU y una salida softmax de cinco clases. Probamos distintas cantidades de neuronas, `Dropout`, cantidad de epocas y learning rate.

La precision de entrenamiento era mayor que la precision sobre datos externos, lo que mostro que habia sobreajuste.

### 5. Diagnostico con matrices de confusion

La matriz mostro que `gif` y `qoi` eran relativamente faciles de reconocer, pero habia confusiones entre `jpg`, `png` y `webp`.

A partir de ese diagnostico decidimos que el problema no era solamente la cantidad de capas: a la red le faltaba una representacion mas rica de los bytes.

### 6. Primer enriquecimiento de features

Sin agregar informacion externa ni usar las etiquetas para construir las entradas, calculamos a partir de los mismos 512 bytes:

- Histograma global de los valores 0-255.
- Media, desvio estandar, minimo y maximo.
- Histogramas locales dividiendo la muestra en 8 bloques de 64 bytes.

La entrada paso de 512 a 2820 caracteristicas. La red seguia siendo feedforward, porque solo usaba capas `Dense`.

### 7. Control del entrenamiento

Agregamos dos callbacks:

- `EarlyStopping`, monitoreando `val_accuracy`, para recuperar la mejor epoca.
- `ReduceLROnPlateau`, monitoreando `val_loss`, para reducir el learning rate cuando el aprendizaje se estanca.

Tambien incorporamos `BatchNormalization` y `Dropout` para estabilizar el entrenamiento y reducir el sobreajuste.

### 8. Comparacion de arquitecturas

Probamos una red mas grande, de `256 -> 128`, y una mas pequeña, de `128 -> 64`.

La red grande aprendia mucho el dataset de entrenamiento, pero no siempre generalizaba mejor. La red de `128 -> 64` mostro un mejor equilibrio entre entrenamiento y datos externos.

### 9. Ultima mejora de features

Como `png` y `webp` seguian confundiendose, agregamos un histograma de transiciones entre bytes consecutivos. Para limitar la cantidad de combinaciones, agrupamos los valores de byte en 16 rangos.

Esta mejora agrego 256 caracteristicas:

```text
2820 features anteriores
+ 256 transiciones cuantizadas
= 3076 caracteristicas
```

La arquitectura final quedo:

```text
3076 -> 128 -> 64 -> 5
```

### 10. Resultado actual

La ultima version alcanzo aproximadamente:

```text
Train:         77.24%
Test publico:  70%
Test final:    68%
```

La matriz final muestra que `gif` y `qoi` se reconocen muy bien. La principal dificultad continua siendo la confusion entre `png` y `webp`.

### 11. Entrega final

Para la entrega cargamos `clasificacion_test_features.csv`, aplicamos exactamente el mismo preprocesamiento y generamos las predicciones textuales:

```python
clases = ['gif', 'jpg', 'png', 'qoi', 'webp']
probas = modelo_entrenado.predict(test_features)
predicciones = [clases[i] for i in np.argmax(probas, axis=1)]
```

Finalmente guardamos un CSV con una sola columna, `y_pred`, y 500 filas.

## 1. Objetivo

El objetivo es clasificar un fragmento de 512 bytes segun el formato de imagen al que pertenece. Las clases son:

- `gif`
- `jpg`
- `png`
- `qoi`
- `webp`

El dataset de entrenamiento tiene 4000 muestras balanceadas, con 800 ejemplos por clase. Cada fila contiene 512 valores de pixel/byte y una etiqueta textual en la ultima columna.

## 2. Carga del dataset

La funcion `cargar_dataset()` lee `dataset.csv` sin encabezado. Separamos:

- Las primeras 512 columnas como entradas `X`.
- La ultima columna como etiqueta `y`.

Las etiquetas textuales se convierten a numeros usando un orden fijo:

```python
{'gif': 0, 'jpg': 1, 'png': 2, 'qoi': 3, 'webp': 4}
```

El uso de `strip()` es importante porque algunas etiquetas del CSV tienen espacios al comienzo o al final.

Luego dividimos los datos en entrenamiento y validacion:

```python
train_test_split(
    datos,
    etiquetas,
    test_size=0.2,
    random_state=1,
    stratify=etiquetas
)
```

Esto deja 3200 muestras para entrenar y 800 para validar. `stratify` mantiene la misma proporcion de clases en ambos conjuntos.

## 3. Normalizacion

Los bytes tienen valores entre 0 y 255. Los dividimos por 255 para llevarlos al rango 0-1:

```python
train_norm = train.astype('float32') / 255.0
test_norm = test.astype('float32') / 255.0
```

La normalizacion ayuda a que el optimizador trabaje con valores comparables y evita que las actualizaciones de pesos sean inestables.

## 4. Features adicionales

La entrada original tenia 512 valores. Como la matriz de confusion mostraba dificultades entre `jpg`, `png` y `webp`, agregamos informacion estadistica de los bytes.

Para cada muestra calculamos:

- Un histograma de 256 posiciones, que indica la frecuencia de cada valor de byte.
- La media de los bytes.
- El desvio estandar.
- El minimo.
- El maximo.

La entrada final queda formada por:

```text
512 valores originales
+ 256 valores del histograma
+ 4 estadisticas
+ 2048 valores de histogramas locales
+ 256 valores de transiciones cuantizadas
= 3076 caracteristicas
```

Los histogramas locales se obtienen dividiendo cada muestra en 8 bloques de 64 bytes. Para cada bloque calculamos un histograma de 256 posiciones. Esto conserva informacion sobre la posicion de los patrones, que se pierde cuando usamos solamente un histograma global.

Tambien calculamos transiciones entre bytes consecutivos. Para reducir la cantidad de combinaciones, agrupamos los valores de byte en 16 rangos y contamos las parejas de rangos que aparecen juntas. Este histograma agrega informacion sobre relaciones locales entre bytes y ayuda especialmente a distinguir `png` de `webp`.

Estas features no cambian el tipo de red: seguimos usando una red feedforward totalmente conectada.

## 5. Arquitectura

La red usa capas `Dense`, por lo que es una red feedforward. No usamos redes recurrentes ni convolucionales.

```python
keras.Input(shape=(3076,)),
Dense(128, activation='relu'),
BatchNormalization(),
Dropout(0.2),
Dense(64, activation='relu'),
BatchNormalization(),
Dropout(0.2),
Dense(5, activation='softmax')
```

### Entrada

`keras.Input(shape=(3076,))` indica que cada muestra tiene 3076 caracteristicas.

### Capas densas

Las capas `Dense` combinan las caracteristicas para aprender patrones que permitan separar las cinco clases.

La cantidad de neuronas se redujo de `256 -> 128` a `128 -> 64` para disminuir la capacidad del modelo y probar si generalizaba mejor fuera del dataset de entrenamiento. La reduccion mantuvo la arquitectura feedforward y fue una modificacion controlada, sin cambiar las features ni el optimizador.

### Funcion de activacion ReLU

ReLU agrega no linealidad y permite que la red aprenda relaciones mas complejas que una combinacion lineal.

### BatchNormalization

Normaliza las activaciones internas durante el entrenamiento. Esto ayuda a estabilizar y acelerar el aprendizaje.

### Dropout

Apaga aleatoriamente el 20% de las conexiones durante el entrenamiento. Su objetivo es reducir el sobreajuste.

### Salida softmax

La ultima capa tiene cinco neuronas, una por clase. `softmax` convierte sus valores en probabilidades cuya suma es 1.

## 6. Compilacion

Usamos:

```python
loss='categorical_crossentropy'
optimizer=keras.optimizers.Adam(learning_rate=0.001)
metrics=['accuracy']
```

- `categorical_crossentropy`: es adecuada para clasificacion multiclase con etiquetas one-hot.
- `Adam`: ajusta los pesos usando una combinacion de momento y tasas de aprendizaje adaptativas.
- `learning_rate=0.001`: controla el tamaño de las actualizaciones.
- `accuracy`: porcentaje de predicciones correctas.

## 7. Entrenamiento

El entrenamiento usa:

```python
epochs=150
batch_size=32
validation_data=(testX, testY)
```

- Una epoca es un recorrido completo por los datos de entrenamiento.
- `batch_size=32` actualiza los pesos cada 32 muestras.
- `testX` y `testY` permiten medir la generalizacion despues de cada epoca.

## 8. EarlyStopping

Agregamos:

```python
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    mode='max',
    patience=10,
    restore_best_weights=True
)
```

La red observa `val_accuracy`. Si no mejora durante 10 epocas, detiene el entrenamiento y recupera los pesos de la mejor epoca.

Esto evita seguir entrenando cuando el modelo comienza a sobreajustar.

## 9. Reduccion del learning rate

Tambien usamos:

```python
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    min_lr=0.00001
)
```

Si la perdida de validacion se estanca durante cuatro epocas, el learning rate se reduce a la mitad. Esto permite que el modelo haga ajustes mas pequenos cerca de una solucion.

## 10. Evaluacion

Medimos el resultado en tres lugares:

1. `trainX`: datos que la red vio durante el entrenamiento.
2. `testX`: 20% separado desde `dataset.csv` para validacion local.
3. `test_publico.csv`: conjunto etiquetado entregado por la catedra para una validacion adicional.

La version con features globales obtuvo aproximadamente 63% en train y 59% en el conjunto publico. Luego agregamos histogramas por bloques y llegamos aproximadamente a 72% en train y 67% en `test_publico`. Finalmente agregamos transiciones cuantizadas entre bytes consecutivos. La version actual obtuvo aproximadamente 77.24% en train, 70% en `test_publico` y 68% en el test final.

Para este ultimo experimento se uso `test_size=0.1`: aproximadamente 3600 muestras para entrenar y 400 para validacion local. Por eso la validacion local debe interpretarse con cautela y las referencias principales son `test_publico` y el test final.

## 11. Matriz de confusion

La matriz de confusion muestra las predicciones por clase:

- Las filas representan la clase real.
- Las columnas representan la clase predicha.
- La diagonal contiene los aciertos.
- Fuera de la diagonal aparecen las confusiones.

La mayor dificultad aparece entre `png` y `webp`. En la matriz actual, `png` tiene aproximadamente 68% de aciertos y `webp` 50%; una parte importante de los errores de ambas clases se intercambia entre ellas. `gif` y `qoi` se reconocen mucho mejor, mientras que `jpg` queda en una posicion intermedia.

La matriz se calcula con `ConfusionMatrixDisplay` y se puede normalizar por fila para comparar porcentajes entre clases.

## 12. Prediccion final

Para la competencia se carga `clasificacion_test_features.csv`, que no tiene etiquetas. Se aplica exactamente el mismo preprocesamiento:

1. Carga de los 512 valores.
2. Division por 255.
3. Generacion de las features adicionales.
4. Prediccion con el modelo entrenado.
5. Conversion de los indices numericos a etiquetas textuales.

```python
clases = ['gif', 'jpg', 'png', 'qoi', 'webp']
probas = modelo_entrenado.predict(test_features)
predicciones = [
    clases[i] for i in np.argmax(probas, axis=1)
]
```

Finalmente guardamos el archivo requerido:

```python
pd.DataFrame({
    'y_pred': predicciones
}).to_csv('entrega_clasificacion.csv', index=False)
```

El archivo debe tener una sola columna llamada `y_pred` y 500 filas.

## 13. Resumen para la exposicion

Una forma breve de explicarlo oralmente es:

> Primero cargue el dataset separando los 512 bytes de la etiqueta. Converti las etiquetas a valores numericos y use one-hot encoding. Dividi los datos de forma estratificada y normalice los bytes al rango 0-1. Como la red confundia principalmente png y webp, agregue histogramas globales, estadisticas, histogramas locales por bloques y transiciones cuantizadas entre bytes consecutivos. De esa forma la entrada paso de 512 a 3076 caracteristicas. Luego use una red feedforward `3076 -> 128 -> 64 -> 5` con ReLU, BatchNormalization, Dropout y softmax. Para controlar el entrenamiento use EarlyStopping y una reduccion automatica del learning rate. La mejor version alcanzo aproximadamente 77.24% en train, 70% en test_publico y 68% en el test final. Finalmente analice el resultado con accuracy, matrices de confusion y genere el CSV final con las etiquetas textuales.
