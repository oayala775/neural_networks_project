import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
import random
from tqdm import tqdm
import shutil

# ----------------------------------------------------------
# Configuraciones generales
# ----------------------------------------------------------
DATASET_PATH = r"datasets/PetImages"  
DESTINO = r"datasets/PetImages_Processed"  
IMG_SIZE = (150, 150)  
os.makedirs(DESTINO, exist_ok=True)

class PreProcesser():
    def __init__(self, dataset_path: str, destino: str, img_size: tuple[int, int]) -> None:
        self.DATASET_PATH = dataset_path
        self.DESTINO = destino
        self.IMG_SIZE = img_size
        os.makedirs(DESTINO, exist_ok=True)

# ----------------------------------------------------------
# a) Cargar y visualizar ejemplos de cada clase
# ----------------------------------------------------------
    def mostrar_ejemplos(self):
        clases = os.listdir(self.DATASET_PATH)
        clases.pop()
        print("Clases encontradas:", clases)

        for clase in clases:
            ruta_clase = os.path.join(self.DATASET_PATH, clase)
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
    def contar_imagenes(self):
        conteo = {}
        clases = os.listdir(self.DATASET_PATH)
        clases.pop()
        for clase in clases:
            ruta_clase = os.path.join(self.DATASET_PATH, clase)
            conteo[clase] = len(os.listdir(ruta_clase))-1
        print("Conteo de imágenes por clase:", conteo)
        return conteo

    def count_images_inside_folder(self,folder_path):
        conteo = {}
        for filename in os.listdir(folder_path):
            tag = filename.rsplit('_')[0]
            if tag in conteo.keys():
                conteo[tag] += 1
            else:
                conteo[tag] = 1
        print(f"Conteo de imágenes por clase: {conteo}\nEn el folder {folder_path}")
        return conteo

    # ----------------------------------------------------------
    # c) Detectar y eliminar imágenes corruptas
    # ----------------------------------------------------------
    def eliminar_corruptas(self):
        total_corruptas = 0
        clases = os.listdir(self.DATASET_PATH)
        clases.pop()
        for clase in clases:
            ruta_clase = os.path.join(self.DATASET_PATH, clase)
            for img_nombre in os.listdir(ruta_clase):
                img_path = os.path.join(ruta_clase, img_nombre)
                try:
                    img = Image.open(img_path)
                    img.verify()
                except (UnidentifiedImageError, IOError, SyntaxError):
                    print("Imagen corrupta eliminada:", img_path)
                    os.remove(img_path)
                    total_corruptas += 1
        print(f"Total de imágenes corruptas eliminadas: {total_corruptas}")

    # ----------------------------------------------------------
    # d) Redimensionar, normalizar y guardar en disco
    # ----------------------------------------------------------
    def procesar_y_guardar(self):
        clases = os.listdir(self.DATASET_PATH)
        clases.pop()
        print("Clases detectadas:", clases)

        for etiqueta, clase in enumerate(clases):
            ruta_clase = os.path.join(self.DATASET_PATH, clase)
            archivos = os.listdir(ruta_clase)

            for img_nombre in tqdm(archivos, desc=f"Procesando {clase}"):
                img_path = os.path.join(ruta_clase, img_nombre)
                nombre_salida = f"{clase}_{os.path.splitext(img_nombre)[0]}"
                destino_path = os.path.join(self.DESTINO, nombre_salida)

                try:
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize(self.IMG_SIZE)
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    np.save(destino_path, img_array)
                except (UnidentifiedImageError, IOError, SyntaxError):
                    continue

        print("\n✅ Procesamiento completado. Imágenes guardadas en:")
        print(self.DESTINO)

    # ----------------------------------------------------------
    # e) División del dataset (70% / 15% / 15%)
    # ----------------------------------------------------------
    def dividir_dataset(self, proporciones=(0.7, 0.15, 0.15)):
        print("\n🔄 Dividiendo dataset en entrenamiento, validación y prueba...")

        # Crear subcarpetas
        rutas = {
            "train": os.path.join(self.DESTINO, "train"),
            "val": os.path.join(self.DESTINO, "val"),
            "test": os.path.join(self.DESTINO, "test")
        }
        for r in rutas.values():
            os.makedirs(r, exist_ok=True)

        # Obtener todos los archivos procesados
        archivos = [f for f in os.listdir(self.DESTINO) if f.endswith(".npy")]
        random.shuffle(archivos)

        total = len(archivos)
        n_train = int(total * proporciones[0])
        n_val = int(total * proporciones[1])

        # División de archivos
        subconjuntos = {
            "train": archivos[:n_train],
            "val": archivos[n_train:n_train + n_val],
            "test": archivos[n_train + n_val:]
        }

        # Mover o copiar archivos
        for tipo, lista in subconjuntos.items():
            for archivo in tqdm(lista, desc=f"Copiando {tipo}"):
                origen = os.path.join(self.DESTINO, archivo)
                destino_archivo = os.path.join(rutas[tipo], archivo)
                shutil.move(origen, destino_archivo)

        print("\n✅ División completada.")
        print(f"Entrenamiento: {len(subconjuntos['train'])} imágenes")
        print(f"Validación: {len(subconjuntos['val'])} imágenes")
        print(f"Prueba: {len(subconjuntos['test'])} imágenes")

    def preprocess(self):
        self.mostrar_ejemplos()
        self.contar_imagenes()
        self.eliminar_corruptas()
        self.procesar_y_guardar()
        self.dividir_dataset()

# ----------------------------------------------------------
# EJECUCIÓN SECUENCIAL
# ----------------------------------------------------------
preprocesser = PreProcesser(DATASET_PATH, DESTINO, IMG_SIZE)

# a) Mostrar imágenes
preprocesser.preprocess()

preprocesser.count_images_inside_folder('datasets/PetImages_Processed/test')
preprocesser.count_images_inside_folder('datasets/PetImages_Processed/train')
preprocesser.count_images_inside_folder('datasets/PetImages_Processed/val')