# Explicacion oral de la red neuronal de clasificacion

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
= 2820 caracteristicas
```

Los histogramas locales se obtienen dividiendo cada muestra en 8 bloques de 64 bytes. Para cada bloque calculamos un histograma de 256 posiciones. Esto conserva informacion sobre la posicion de los patrones, que se pierde cuando usamos solamente un histograma global.

Estas features no cambian el tipo de red: seguimos usando una red feedforward totalmente conectada.

## 5. Arquitectura

La red usa capas `Dense`, por lo que es una red feedforward. No usamos redes recurrentes ni convolucionales.

```python
keras.Input(shape=(2820,)),
Dense(256, activation='relu'),
BatchNormalization(),
Dropout(0.2),
Dense(128, activation='relu'),
BatchNormalization(),
Dropout(0.2),
Dense(5, activation='softmax')
```

### Entrada

`keras.Input(shape=(2820,))` indica que cada muestra tiene 2820 caracteristicas.

### Capas densas

Las capas `Dense` combinan las caracteristicas para aprender patrones que permitan separar las cinco clases.

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

La version con features globales obtuvo aproximadamente 63% en train y 59% en el conjunto publico. Luego agregamos histogramas por bloques y el resultado subio aproximadamente a 72% en train y 67% en `test_publico`. La diferencia de 5 puntos porcentuales indica una generalizacion razonable.

## 11. Matriz de confusion

La matriz de confusion muestra las predicciones por clase:

- Las filas representan la clase real.
- Las columnas representan la clase predicha.
- La diagonal contiene los aciertos.
- Fuera de la diagonal aparecen las confusiones.

La mayor dificultad aparece entre `jpg`, `png` y `webp`. Esto indica que esas clases comparten patrones estadisticos en los fragmentos de 512 bytes. En el conjunto base la separacion de `jpg` mejoro, aunque en `test_publico` las matrices siguen siendo mas difusas porque contiene muestras diferentes y menos numerosas.

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

> Primero cargue el dataset separando los 512 bytes de la etiqueta. Converti las etiquetas a valores numericos y use one-hot encoding. Dividi los datos en entrenamiento y validacion de forma estratificada y normalice los bytes al rango 0-1. Como la red confundia principalmente jpg, png y webp, agregue histogramas globales, estadisticas y luego histogramas locales dividiendo cada muestra en 8 bloques. De esa forma la entrada paso de 512 a 2820 caracteristicas. Luego use una red feedforward con capas densas, ReLU, BatchNormalization, Dropout y una salida softmax de cinco clases. Para controlar el entrenamiento use EarlyStopping y una reduccion automatica del learning rate. El resultado mejoro hasta aproximadamente 72% en train y 67% en test_publico. Finalmente analice el resultado con accuracy, matrices de confusion y genere el CSV final con las etiquetas textuales.
