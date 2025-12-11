# Plant Classification – Medicinal Plants Classifier

**Autores:** Juan Jeronimo Castaño Rivera  y Campos Herney Tulcan Cuasapud
**Universidad Nacional de Colombia – Procesamiento Digital de Imágenes (PDI)**

Este proyecto implementa un clasificador de plantas medicinales utilizando **YOLOv8 en modo de clasificación**, junto con una **aplicación interactiva** desplegada en **HuggingFace Spaces** y un **script de inferencia local**.

El modelo fue entrenado con un dataset personalizado y exportado a **TorchScript**, permitiendo compatibilidad tanto en entornos web como en ejecución local.

---

## Dataset

El conjunto de datos fue gestionado y anotado en **Roboflow**.

 **Dataset en Roboflow:**  
https://app.roboflow.com/procesamientoimagenes/green_machinev2-nxegl-fzy6k/1

**Clases incluidas (11):**

- Aloe vera (Sábila)  
- Calendula officinalis  
- Chamaemelum nobile (Manzanilla)  
- Dysphania ambrosioides (Paico)  
- Eryngium foetidum (Cimarrón)  
- Erythroxylum coca  
- Mentha spicata (Hierbabuena)  
- Peumus boldus (Boldo)  
- Plantas No Medicinales  
- Ruta graveolens (Ruda)  
- Valeriana officinalis  

---

## Modelo

Modelo base utilizado:

```
yolov8s-cls.pt
```

Entrenado durante **50 épocas**, generando:

- **results.png** (curvas de entrenamiento)
- **Matriz de confusión**
- **best.pt** (mejor modelo)
- **best.torchscript** (modelo exportado)

---

## Demo en HuggingFace Spaces

El proyecto incluye una aplicación web interactiva para clasificar imágenes.

 **HuggingFace Space:**  
https://huggingface.co/spaces/juacastanori/plantspace

**Funciones de la app:**

- Subir una imagen  
- Ver predicciones Top‑K  
- Retorno en JSON  
- Interfaz con **Gradio**

---

## Ejecución local – Inferencia

Puedes ejecutar el modelo localmente mediante un script que acepta cualquier imagen como entrada.

### Requisitos

```bash
pip install ultralytics opencv-python matplotlib torch pillow
```

### Uso del script

```bash
python inferencia_local.py --imagen "planta.jpg" --modelo "best.torchscript"
```

**Parámetros:**

- `--imagen`: ruta a la imagen a clasificar  
- `--modelo`: permite `best.pt` o `best.torchscript`

### Ejemplo de salida

```
Predicción: Mentha spicata (Hierbabuena)
Confianza: 0.94 (94.0%)
```

Incluye visualización con **Matplotlib**.

---

## Arquitectura General del Proyecto

```
Clasificación de Plantas Medicinales
├── entrenamiento/
│   ├── dataset descargado
│   ├── data.yaml
│   ├── entrenamiento YOLOv8
│   ├── métricas y gráficas
│   └── exportación TorchScript
│
├── huggingface/
│   ├── app.py
│   ├── labels.json
│   ├── requirements.txt
│   ├── best.torchscript
│   └── deployment en HF Space
│
├── local/
│   ├── inferencia_local.py
│   └── pruebas
│
└── README.md
```

---

## 🚀 Cómo replicar el proyecto (entrenamiento)

### 1. Descargar dataset desde Roboflow con API Key  
### 2. Crear archivo `data.yaml` con rutas locales  
### 3. Entrenar modelo:

```python
from ultralytics import YOLO

model = YOLO('yolov8s-cls.pt')
model.train(
    task='classify',
    data=dataset_dir,
    epochs=50,
    imgsz=640,
    batch=32
)
```

### 4. Exportar a TorchScript

```python
model.export(format="torchscript", imgsz=640)
```

### 5. Subir modelo exportado a HuggingFace Spaces


