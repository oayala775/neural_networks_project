import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
import random
from tqdm import tqdm

# ----------------------------------------------------------
# Configuraciones generales
# ----------------------------------------------------------
DATASET_PATH = r"D:\beto5\Descargas\Tareas\9no\Neuronales\a\PetImages"  # Ruta original
DESTINO = r"D:\beto5\Descargas\Tareas\9no\Neuronales\a\PetImages_preprocesado"  # Guardado
IMG_SIZE = (150, 150)  # Tamaño fijo
os.makedirs(DESTINO, exist_ok=True)

# ----------------------------------------------------------
# a) Cargar y visualizar ejemplos de cada clase
# ----------------------------------------------------------
def mostrar_ejemplos(ruta_dataset):
    clases = os.listdir(ruta_dataset)
    print("Clases encontradas:", clases)

    for clase in clases:
        ruta_clase = os.path.join(ruta_dataset, clase)
        imagenes = os.listdir(ruta_clase)
        seleccion = random.sample(imagenes, 3)

        plt.figure(figsize=(10, 4))
        plt.suptitle(f"Ejemplos de la clase: {clase}", fontsize=14)

        for i, img_nombre in enumerate(seleccion):
            img_path = os.path.join(ruta_clase, img_nombre)
            try:
                img = Image.open(img_path)
                plt.subplot(1, 3, i + 1)
                plt.imshow(img)
                plt.axis("off")
            except Exception:
                plt.subplot(1, 3, i + 1)
                plt.text(0.5, 0.5, "Imagen dañada", ha="center", va="center")
                plt.axis("off")
        plt.show()

# ----------------------------------------------------------
# b) Contar cuántas imágenes hay por clase
# ----------------------------------------------------------
def contar_imagenes(ruta_dataset):
    conteo = {}
    for clase in os.listdir(ruta_dataset):
        ruta_clase = os.path.join(ruta_dataset, clase)
        conteo[clase] = len(os.listdir(ruta_clase))
    print("Conteo de imágenes por clase:", conteo)
    return conteo

# ----------------------------------------------------------
# c) Detectar y eliminar imágenes corruptas
# ----------------------------------------------------------
def eliminar_corruptas(ruta_dataset):
    total_corruptas = 0
    for clase in os.listdir(ruta_dataset):
        ruta_clase = os.path.join(ruta_dataset, clase)
        for img_nombre in os.listdir(ruta_clase):
            img_path = os.path.join(ruta_clase, img_nombre)
            try:
                img = Image.open(img_path)
                img.verify()  # Verifica sin cargarla completamente
            except (UnidentifiedImageError, IOError, SyntaxError):
                print("Imagen corrupta eliminada:", img_path)
                os.remove(img_path)
                total_corruptas += 1
    print(f"Total de imágenes corruptas eliminadas: {total_corruptas}")

# ----------------------------------------------------------
# d) Redimensionar, normalizar y guardar en disco
# ----------------------------------------------------------
def procesar_y_guardar(ruta_dataset, img_size, destino):
    clases = sorted(os.listdir(ruta_dataset))
    print("Clases detectadas:", clases)

    for etiqueta, clase in enumerate(clases):
        ruta_clase = os.path.join(ruta_dataset, clase)
        archivos = os.listdir(ruta_clase)

        for img_nombre in tqdm(archivos, desc=f"Procesando {clase}"):
            img_path = os.path.join(ruta_clase, img_nombre)
            nombre_salida = f"{clase}_{os.path.splitext(img_nombre)[0]}.npy"
            destino_path = os.path.join(destino, nombre_salida)

            try:
                # Abrir, redimensionar y normalizar
                img = Image.open(img_path).convert("RGB")
                img = img.resize(img_size)
                img_array = np.array(img, dtype=np.float32) / 255.0

                # Guardar imagen procesada en disco
                np.save(destino_path, img_array)

            except (UnidentifiedImageError, IOError, SyntaxError):
                # Si no se puede leer, se omite
                continue

    print("\n✅ Procesamiento completado. Imágenes guardadas en:")
    print(destino)

# ----------------------------------------------------------
# EJECUCIÓN SECUENCIAL
# ----------------------------------------------------------

# a) Mostrar imágenes
mostrar_ejemplos(DATASET_PATH)

# b) Contar imágenes
contar_imagenes(DATASET_PATH)

# c) Buscar y eliminar corruptas
eliminar_corruptas(DATASET_PATH)

# d + e) Redimensionar, normalizar y guardar
procesar_y_guardar(DATASET_PATH, IMG_SIZE, DESTINO)
