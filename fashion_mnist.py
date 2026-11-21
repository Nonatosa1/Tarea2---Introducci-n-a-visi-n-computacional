"""
Tarea 2 - Introducción a visión computacional
Sistemas Inteligentes

Entrena varias redes neuronales densas (con capa Flatten) sobre Fashion-MNIST,
compara su accuracy y grafica en 2D la representación que aprende cada red.

Uso:
    python fashion_mnist.py            # 10 épocas por configuración
    python fashion_mnist.py --epochs 5
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")  # guarda las figuras sin necesidad de abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

CLASES = [
    "Camiseta", "Pantalón", "Suéter", "Vestido", "Abrigo",
    "Sandalia", "Camisa", "Tenis", "Bolsa", "Botín",
]
OUT_DIR = "figuras"
SEED = 42


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def normalizar(imagen, etiqueta):
    """Convierte los píxeles de [0, 255] a [0, 1]."""
    return tf.cast(imagen, tf.float32) / 255.0, etiqueta


def cargar_datos():
    train_ds, test_ds = tfds.load(
        "fashion_mnist",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        batch_size=32,
    )
    train_ds = train_ds.map(normalizar).cache().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(normalizar).cache().prefetch(tf.data.AUTOTUNE)
    return train_ds, test_ds


def dataset_a_numpy(ds):
    xs, ys = [], []
    for x, y in ds:
        xs.append(x.numpy())
        ys.append(y.numpy())
    return np.concatenate(xs), np.concatenate(ys)


# ---------------------------------------------------------------------------
# Modelos: todos empiezan con Flatten (28x28x1 -> vector de 784)
# ---------------------------------------------------------------------------
def modelo_lineal():
    """Config. 1: Flatten + softmax (equivale a regresión logística)."""
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation="softmax"),
    ], name="1_lineal")


def modelo_una_oculta():
    """Config. 2: una capa oculta de 128 neuronas ReLU."""
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ], name="2_oculta_128")


def modelo_profundo():
    """Config. 3: dos capas ocultas (256, 128) con Dropout."""
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation="softmax"),
    ], name="3_profundo_dropout")


def modelo_cuello_2d():
    """Config. 4: cuello de botella de 2 neuronas para graficar directamente en 2D."""
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(2, name="espacio_2d"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ], name="4_cuello_2d")


CONFIGURACIONES = [modelo_lineal, modelo_una_oculta, modelo_profundo, modelo_cuello_2d]


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------
def scatter_2d(ax, puntos, etiquetas, titulo):
    cmap = plt.get_cmap("tab10")
    for c in range(10):
        m = etiquetas == c
        ax.scatter(puntos[m, 0], puntos[m, 1], s=3, alpha=0.6,
                   color=cmap(c), label=CLASES[c])
    ax.set_title(titulo, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def graficar_ejemplos(x, y):
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    for c, ax in enumerate(axes.flat):
        idx = np.where(y == c)[0][0]
        ax.imshow(x[idx].squeeze(), cmap="gray")
        ax.set_title(CLASES[c])
        ax.axis("off")
    fig.suptitle("Fashion-MNIST: un ejemplo por clase")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ejemplos.png"), dpi=120)
    plt.close(fig)


def graficar_curvas(historiales):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for nombre, h in historiales.items():
        linea, = ax1.plot(h["accuracy"], label=f"{nombre} (train)")
        ax1.plot(h["val_accuracy"], "--", color=linea.get_color(),
                 label=f"{nombre} (test)")
        ax2.plot(h["loss"], color=linea.get_color(), label=f"{nombre} (train)")
        ax2.plot(h["val_loss"], "--", color=linea.get_color(), label=f"{nombre} (test)")
    ax1.set_title("Accuracy por época")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Accuracy")
    ax2.set_title("Pérdida por época")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Sparse categorical cross-entropy")
    ax1.legend(fontsize=7)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "curvas_entrenamiento.png"), dpi=120)
    plt.close(fig)


def graficar_comparacion(resultados):
    nombres = [r["modelo"] for r in resultados]
    accs = [r["test_accuracy"] for r in resultados]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    barras = ax.bar(nombres, accs, color="steelblue")
    ax.bar_label(barras, labels=[f"{a:.4f}" for a in accs])
    ax.set_ylim(min(accs) - 0.05, 1.0)
    ax.set_ylabel("Accuracy en test")
    ax.set_title("Comparación de configuraciones")
    plt.setp(ax.get_xticklabels(), rotation=15)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "comparacion_accuracy.png"), dpi=120)
    plt.close(fig)


def graficar_espacios_2d(x_test, y_test, modelos, n=5000):
    """Proyección 2D de los píxeles crudos y de la penúltima capa de cada modelo."""
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(x_test), size=n, replace=False)
    x, y = x_test[idx], y_test[idx]

    paneles = [("Píxeles crudos (Flatten) + PCA",
                PCA(2, random_state=SEED).fit_transform(x.reshape(len(x), -1)))]

    for modelo in modelos:
        if isinstance(modelo.layers[-2], tf.keras.layers.Flatten):
            # El modelo lineal no tiene capa oculta: se proyecta su salida softmax
            rep = modelo.predict(x, verbose=0)
            titulo = f"{modelo.name}: salida softmax (10) + PCA"
            rep = PCA(2, random_state=SEED).fit_transform(rep)
            paneles.append((titulo, rep))
            continue
        # Salida de la penúltima capa (antes del softmax)
        extractor = tf.keras.Model(modelo.inputs, modelo.layers[-2].output)
        rep = extractor.predict(x, verbose=0)
        if rep.shape[1] == 2:
            titulo = f"{modelo.name}: capa de 2 neuronas (sin PCA)"
        else:
            titulo = f"{modelo.name}: capa oculta ({rep.shape[1]}) + PCA"
            rep = PCA(2, random_state=SEED).fit_transform(rep)
        paneles.append((titulo, rep))

    cols = 3
    filas = int(np.ceil(len(paneles) / cols))
    fig, axes = plt.subplots(filas, cols, figsize=(5 * cols, 4.5 * filas))
    for ax, (titulo, puntos) in zip(axes.flat, paneles):
        scatter_2d(ax, puntos, y, titulo)
    for ax in list(axes.flat)[len(paneles):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=10, markerscale=4, fontsize=9)
    fig.suptitle("Representaciones en 2D del conjunto de prueba")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(os.path.join(OUT_DIR, "espacios_2d.png"), dpi=120)
    plt.close(fig)

    # Figura individual del cuello de botella (la gráfica 2D "pura")
    modelo_2d = next(m for m in modelos if m.name == "4_cuello_2d")
    extractor = tf.keras.Model(modelo_2d.inputs, modelo_2d.get_layer("espacio_2d").output)
    puntos = extractor.predict(x, verbose=0)
    fig, ax = plt.subplots(figsize=(8, 7))
    scatter_2d(ax, puntos, y, "Salida de la capa de 2 neuronas (modelo 4)")
    ax.set_xticks(np.round(np.linspace(*ax.get_xlim(), 5), 1))
    ax.set_yticks(np.round(np.linspace(*ax.get_ylim(), 5), 1))
    ax.set_xlabel("Neurona 1")
    ax.set_ylabel("Neurona 2")
    ax.legend(markerscale=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "cuello_2d.png"), dpi=120)
    plt.close(fig)


def graficar_confusion(modelo, x_test, y_test):
    pred = modelo.predict(x_test, verbose=0).argmax(axis=1)
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10), CLASES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASES)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión - {modelo.name}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "matriz_confusion.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    train_ds, test_ds = cargar_datos()
    x_test, y_test = dataset_a_numpy(test_ds)
    graficar_ejemplos(x_test, y_test)

    resultados, historiales, modelos = [], {}, []
    for construir in CONFIGURACIONES:
        modelo = construir()
        modelo.compile(optimizer="adam",
                       loss="sparse_categorical_crossentropy",
                       metrics=["accuracy"])
        print(f"\n=== {modelo.name} ({modelo.count_params():,} parámetros) ===")
        modelo.summary()
        h = modelo.fit(train_ds, validation_data=test_ds,
                       epochs=args.epochs, verbose=2)
        _, acc = modelo.evaluate(test_ds, verbose=0)
        print(f"Accuracy en test: {acc:.4f}")

        historiales[modelo.name] = h.history
        modelos.append(modelo)
        resultados.append({
            "modelo": modelo.name,
            "parametros": modelo.count_params(),
            "train_accuracy": round(h.history["accuracy"][-1], 4),
            "test_accuracy": round(acc, 4),
        })

    with open(os.path.join(OUT_DIR, "resultados.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=resultados[0].keys())
        w.writeheader()
        w.writerows(resultados)

    graficar_curvas(historiales)
    graficar_comparacion(resultados)
    graficar_espacios_2d(x_test, y_test, modelos)
    mejor = max(zip(resultados, modelos), key=lambda rm: rm[0]["test_accuracy"])[1]
    graficar_confusion(mejor, x_test, y_test)

    print("\nResumen:")
    for r in resultados:
        print(f"  {r['modelo']:<22} params={r['parametros']:>8,}  "
              f"train={r['train_accuracy']:.4f}  test={r['test_accuracy']:.4f}")
    print(f"\nFiguras y resultados guardados en ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
