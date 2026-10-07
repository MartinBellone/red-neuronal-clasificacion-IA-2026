# Competencia Redes Neuronales 2026 — FI-UNMdP

Fuente: https://ia2026ingenieriaunmdp.duckdns.org/instrucciones

## Contexto

Hay dos competencias independientes: **regresión** y **clasificación**. En cada una:

1. Se entrena una red neuronal con datos etiquetados (localmente).
2. Se predice un conjunto de test sin etiquetas.
3. Se suben solo las predicciones (CSV) + el notebook `.ipynb` que las generó.

Grupos de hasta 2 integrantes.

---

## Competencia 1: Regresión

| Ítem | Valor |
|---|---|
| Tarea | Predecir un valor numérico continuo a partir de 14 variables |
| Train | `regresion_train.csv` — 800 filas, columnas `feat_A` … `feat_N` + `y_target` |
| Test | `regresion_test_features.csv` — 200 filas, mismas 14 columnas, sin `y_target` |
| Métrica | RMSE (menor es mejor) |
| Desempate | MAE |
| Entregas | 2 por día |

Carga sugerida:

```python
import pandas as pd

train = pd.read_csv("regresion_train.csv")
X = train.drop(columns="y_target").to_numpy("float32")
y = train["y_target"].to_numpy("float32")

X_test = pd.read_csv("regresion_test_features.csv").to_numpy("float32")  # 200 x 14
```

---

## Competencia 2: Clasificación

| Ítem | Valor |
|---|---|
| Tarea | Identificar el formato de imagen a partir de un fragmento de 512 bytes |
| Clases | `gif`, `jpg`, `png`, `qoi`, `webp` |
| Input | 512 enteros (0–255) por fila |
| Métrica | Accuracy (mayor es mejor) |
| Desempate | F1 macro |
| Entregas | 2 por día |

**Importante:** el fragmento NO es necesariamente el comienzo del archivo, así que no alcanza con detectar la firma (magic bytes) del formato.

### Archivos

| Archivo | Contenido |
|---|---|
| `dataset.csv` | 4000 filas, sin encabezado: 512 bytes + etiqueta en la última columna. 800 filas por clase (balanceado). |
| `test_publico.csv` | 500 filas etiquetadas, mismo formato. Para validación local. **No** es el conjunto puntuado. |
| `fulldataset.pickle` | Las mismas muestras de `dataset.csv`: lista de pares `(np.ndarray uint8 de 512, etiqueta)`. |
| `clasificacion_test_features.csv` | 500 filas × 512 columnas, sin encabezado ni etiqueta. Este es el que se predice. |

Carga sugerida:

```python
import pandas as pd

train = pd.read_csv("dataset.csv", header=None, skipinitialspace=True)
X = train.iloc[:, :512].to_numpy("float32") / 255.0
y = train[512].to_numpy()  # "gif", "jpg", "png", "qoi" o "webp"

X_test = pd.read_csv("clasificacion_test_features.csv", header=None,
                     skipinitialspace=True).to_numpy("float32") / 255.0  # 500 x 512

# Alternativa:
# import pickle; muestras = pickle.load(open("fulldataset.pickle", "rb"))
```

Conversión de softmax a etiquetas (respetar el mismo orden de clases usado al codificar `y`):

```python
clases = ["gif", "jpg", "png", "qoi", "webp"]
probas = model.predict(X_test)  # 500 x 5
y_pred = [clases[i] for i in probas.argmax(axis=1)]
```

---

## Formato de entrega

Cada entrega = **2 archivos**:

1. **CSV de predicciones**
   - Una única columna, encabezado exacto: `y_pred`
   - Sin índice ni features
   - Regresión: 200 filas, números con punto decimal
   - Clasificación: 500 filas, etiquetas en texto y minúsculas (`png`, no `2`)
   - La fila *i* corresponde a la fila *i* de las test features. **No reordenar ni filtrar.**
2. **Notebook `.ipynb`**
   - El que entrenó el modelo y generó ese CSV
   - Guardado después de ejecutarlo, con las salidas visibles
   - Debe ser `.ipynb` original (no `.py`, `.html`, PDF). Máx. 20 MB
   - Solo lo ve el docente

Guardado seguro:

```python
pd.DataFrame({"y_pred": y_pred}).to_csv("entrega.csv", index=False)
```

La web valida el formato al subir; si falla, no se descuenta del cupo.

---

## Puntuación

| Competencia | Leaderboard (visible) | Final (oculto) |
|---|---|---|
| Regresión | filas 1–100 | filas 101–200 |
| Clasificación | filas 1–250 | filas 251–500 |

- Durante la competencia, cada entrega se puntúa con la primera mitad.
- Al cierre, se toma la **mejor entrega del leaderboard** de cada competencia y se puntúa una sola vez con la segunda mitad.
- Empate persistente: gana la entrega más antigua.
- Hay que predecir **todas** las filas: la segunda mitad define el resultado.
- Advertencia: sobreajustar al leaderboard es riesgoso (pocas filas). Confiar en la validación propia.

---

## Reglas

- 2 entregas por día por competencia, por grupo. Se renueva a las 00:00 (hora Argentina).
- Solo cuentan entregas que pasaron la validación de formato.
- Cada alumno en un solo grupo.
- Las predicciones deben salir de un modelo entrenado por el grupo; el notebook debe ser el que las produjo.
- Prohibido etiquetar a mano el test o conseguir las etiquetas por otros medios.
- Se pueden compartir ideas en el Conversatorio, pero no archivos de predicciones.

---

## Errores comunes

| Síntoma | Causa / solución |
|---|---|
| Columna extra antes de `y_pred` | Se guardó el índice → `to_csv(..., index=False)` |
| "Se esperan 200 / 500 predicciones" | Se predijo otro conjunto (validación o `test_publico.csv`) |
| Notebook inválido | No es un `.ipynb` real; descargarlo desde Colab/Jupyter |
| Falta encabezado | La primera línea debe ser `y_pred` |
| Separador `;` / coma decimal | Excel en español; generar con pandas y no abrir/guardar con Excel |
| Etiqueta inválida | Usar texto (`png`), no números ni probabilidades |
| Puntaje malo con buena validación | Se mezclaron (shuffle) las test features o se normalizaron distinto que el train |

---

## Validación local antes de subir

```python
import pandas as pd

def chequear(path, filas, etiquetas=None):
    df = pd.read_csv(path)
    assert list(df.columns) == ["y_pred"], f"Columnas: {list(df.columns)}"
    assert len(df) == filas, f"Hay {len(df)} filas y se esperan {filas}"
    if etiquetas:
        malas = set(df.y_pred) - set(etiquetas)
        assert not malas, f"Etiquetas inválidas: {malas}"
    else:
        assert df.y_pred.notna().all(), "Hay valores vacíos"
    print("Formato OK")

chequear("entrega_regresion.csv", 200)
chequear("entrega_clasificacion.csv", 500, ["gif", "jpg", "png", "qoi", "webp"])
```
