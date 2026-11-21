# Tarea 2 — Introducción a visión computacional

| | |
|---|---|
| **Alumno** | Nonato Bazán Alberto |
| **Número de cuenta** | 423022225 |
| **Semestre** | Séptimo |
| **Materia** | Sistemas Inteligentes |

## Descripción

Se entrenan redes neuronales densas para clasificar prendas del conjunto **Fashion-MNIST**
(70,000 imágenes de 28×28 en escala de grises, 10 clases), cargado con `tensorflow_datasets`.
Todas las redes empiezan con una capa `Flatten`, que convierte cada imagen en un vector de 784 valores.
Se comparan cuatro configuraciones, se usa **accuracy** como métrica y se grafican en 2D las
representaciones que aprende cada red.

## Configuraciones

| # | Arquitectura (después de `Flatten`) | Parámetros | Accuracy train | Accuracy test |
|---|---|---:|---:|---:|
| 1 | `Dense(10, softmax)` (regresión logística) | 7,850 | 0.8645 | 0.8459 |
| 2 | `Dense(128, relu)` → `Dense(10)` | 101,770 | 0.9153 | **0.8790** |
| 3 | `Dense(256, relu)` → `Dropout(0.3)` → `Dense(128, relu)` → `Dropout(0.3)` → `Dense(10)` | 235,146 | 0.8816 | 0.8779 |
| 4 | `Dense(128, relu)` → `Dense(2)` → `Dense(10)` (cuello de botella 2D) | 100,768 | 0.8873 | 0.8562 |

Todas se entrenaron 10 épocas con Adam, `sparse_categorical_crossentropy` y lotes de 32.

## Conclusiones

- Basta una capa oculta (config. 2) para subir unos 3 puntos de accuracy sobre el modelo lineal.
- La config. 2 tiene la mejor accuracy en test, pero también la mayor diferencia entre train y test
  (sobreajuste). Con Dropout (config. 3) se obtiene casi la misma accuracy en test sin sobreajuste.
- Si se obliga a la red a pasar por solo 2 neuronas (config. 4), pierde poco más de 2 puntos, y a cambio
  su representación interna se puede graficar directamente en 2D. Las clases de calzado, la bolsa y el
  pantalón quedan bien separadas, mientras que camiseta, camisa, suéter y abrigo se enciman.
  Esas también son las clases que más se confunden en la matriz de confusión.

## Archivos

- `fashion_mnist.py` — código completo: carga de datos, modelos, entrenamiento y gráficas.
- `requirements.txt` — dependencias.
- `figuras/`
  - `ejemplos.png` — un ejemplo de cada clase.
  - `curvas_entrenamiento.png` — accuracy y pérdida por época de cada configuración.
  - `comparacion_accuracy.png` — accuracy en test de cada configuración.
  - `espacios_2d.png` — proyección 2D de los píxeles crudos y de la representación de cada modelo.
  - `cuello_2d.png` — salida de la capa de 2 neuronas del modelo 4.
  - `matriz_confusion.png` — matriz de confusión del mejor modelo.
  - `resultados.csv` — tabla de resultados.

## Ejecución

```bash
pip install -r requirements.txt
python fashion_mnist.py            # 10 épocas por configuración (~6 min en CPU)
python fashion_mnist.py --epochs 5
```
