# -*- coding: utf-8 -*-
"""
API Web de producción construida con FastAPI para la digitalización de
fechas de vencimiento a partir de fotos de productos o cajas.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import shutil
import tempfile
from typing import Dict, Any, Optional
import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

from deployment.inference import OCRPipeline

app = FastAPI(
    title="API de Digitalización de Vencimientos",
    description="API para detectar y transcribir fechas de vencimiento en fotos utilizando YOLO OBB y EasyOCR.",
    version="1.0.0"
)

# Inicializar variables de configuración
pipeline = None
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "deploy_config.yaml")


@app.on_event("startup")
def startup_event():
    """Inicializa el pipeline de inferencia al arrancar el servidor web."""
    global pipeline
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"No se encontró el archivo de configuración en {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    inf_cfg = config["inference"]

    # Verificar existencia de archivos de modelo antes de inicializar
    if not os.path.exists(inf_cfg["yolo_model_path"]):
        print(f"⚠️ Advertencia: No se encontró el modelo YOLO en {inf_cfg['yolo_model_path']}")
    if not os.path.exists(inf_cfg["ocr_model_path"]):
        print(f"⚠️ Advertencia: No se encontró el modelo EasyOCR en {inf_cfg['ocr_model_path']}")

    pipeline = OCRPipeline(
        yolo_model_path=inf_cfg["yolo_model_path"],
        ocr_model_path=inf_cfg["ocr_model_path"],
        conf_threshold=inf_cfg.get("conf_threshold", 0.25),
        iou_threshold=inf_cfg.get("iou_threshold", 0.7),
        gpu=inf_cfg.get("gpu", False)
    )
    print("API de Digitalización de Vencimientos cargada y lista para recibir peticiones.")


class PredictionResponse(BaseModel):
    """Estructura del JSON de respuesta de la API."""
    success: bool
    filename: str
    ocr_texts: list
    normalized_dates: list
    expiration_date: Optional[str] = None
    error: Optional[str] = None


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Recibe una imagen a través de FormData, ejecuta el pipeline unificado de
    detección de fecha con YOLO OBB y transcripción OCR, y retorna los resultados.
    """
    global pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline de inferencia no inicializado.")

    # Validar extensión básica de la imagen
    allowed_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    _, ext = os.path.splitext(file.filename)
    if ext.lower() not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Formato de archivo no soportado. Extensiones permitidas: {allowed_extensions}"
        )

    # Crear un directorio temporal seguro dentro del workspace para persistir la carga
    temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)

    temp_file_path = os.path.join(temp_dir, f"upload_{os.urandom(8).hex()}{ext}")

    try:
        # Escribir bytes del archivo cargado a disco
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Ejecutar pipeline unificado de inferencia (sin persistir outputs gráficos intermedios)
        results = pipeline.execute_pipeline(
            image_path=temp_file_path,
            save_crops=True,
            persist_images=False
        )

        return {
            "success": True,
            "filename": file.filename,
            "ocr_texts": results.get("ocr_results", []),
            "normalized_dates": results.get("normalized_dates", []),
            "expiration_date": results.get("fecha_vencimiento", None)
        }

    except Exception as e:
        print(f"Error procesando la imagen {file.filename}: {e}")
        return {
            "success": False,
            "filename": file.filename,
            "ocr_texts": [],
            "normalized_dates": [],
            "expiration_date": None,
            "error": str(e)
        }

    finally:
        # Eliminar archivo temporal
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                print(f"No se pudo eliminar el archivo temporal {temp_file_path}: {e}")


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Endpoint simple de chequeo de estado de salud del servicio."""
    status = "healthy" if pipeline is not None else "initializing"
    return {"status": status}
