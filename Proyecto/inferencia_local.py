# -*- coding: utf-8 -*-
"""script_local.ipynb


Original file is located at
    https://colab.research.google.com/drive/1wdVf47St2P8XQBJ9KPRWt6c5vGW8oLCP
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
import os

def realizar_inferencia(ruta_imagen, ruta_modelo):
    """
    Carga el modelo y realiza la predicción sobre la imagen dada.
    """
    # 1. Verificar que los archivos existen
    if not os.path.exists(ruta_imagen):
        print(f"Error: La imagen '{ruta_imagen}' no existe.")
        return
    if not os.path.exists(ruta_modelo):
        print(f"Error: El modelo '{ruta_modelo}' no existe.")
        return

    print(f"Cargando modelo: {ruta_modelo}...")

    # 2. Cargar el modelo (Soporta .pt y .torchscript gracias a Ultralytics)
    model = YOLO(ruta_modelo, task='classify')

    print(f"Procesando imagen: {ruta_imagen}...")

    # 3. Ejecutar inferencia
    # conf=0.25 establece un umbral mínimo de confianza (opcional)
    results = model(ruta_imagen)

    # 4. Procesar resultados
    result = results[0]

    # Obtener la clase con mayor probabilidad
    top1_index = result.probs.top1
    top1_conf = result.probs.top1conf.item()
    class_name = result.names[top1_index]

    print(f"\n--- RESULTADO ---")
    print(f"Predicción: {class_name}")
    print(f"Confianza: {top1_conf:.2f} ({top1_conf*100:.1f}%)")

    # 5. Graficar resultados
    img_bgr = cv2.imread(ruta_imagen)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.axis('off')

    # Título con el resultado
    titulo = f"Clase: {class_name} | Conf: {top1_conf:.2%}"
    plt.title(titulo, fontsize=16, color='green', fontweight='bold')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Configuración de argumentos para ejecutar desde consola
    parser = argparse.ArgumentParser(description="Script de Inferencia Local para Clasificación de Plantas")

    parser.add_argument(
        "--imagen",
        type=str,
        required=True,
        help="Ruta del archivo de imagen a clasificar"
    )

    parser.add_argument(
        "--modelo",
        type=str,
        default="best.torchscript",
        help="Ruta del archivo del modelo ( best.pt o best.torchscript)"
    )

    args = parser.parse_args()

    realizar_inferencia(args.imagen, args.modelo)