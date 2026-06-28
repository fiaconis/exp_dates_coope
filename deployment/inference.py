# -*- coding: utf-8 -*-
"""
Módulo de inferencia en producción. Define la clase OCRPipeline que integra
YOLOOBBInference, EasyOCRInference y DateNormalizer en un único flujo robusto.
"""

from datetime import datetime
import os
from typing import Optional, Dict, Any, List
import cv2

from src.models import YOLOOBBInference, EasyOCRInference
from src.evaluation import DateNormalizer


class OCRPipeline:
    """
    Pipeline unificado de inferencia. Ejecuta la detección orientada (YOLO OBB)
    en 4 rotaciones de la imagen original, realiza recortes y rotaciones a nivel de crop,
    transcribe el texto con EasyOCR y normaliza las fechas para determinar la fecha
    de vencimiento definitiva (la más futura).
    """

    def __init__(self, yolo_model_path: str, ocr_model_path: str,
                 conf_threshold: float = 0.25, iou_threshold: float = 0.7,
                 gpu: bool = False, model_storage_dir: Optional[str] = None):
        """
        Inicializa los modelos y el normalizador de fechas del pipeline.

        Args:
            yolo_model_path (str): Ruta al checkpoint de YOLO (.pt).
            ocr_model_path (str): Ruta al checkpoint de EasyOCR (.pth).
            conf_threshold (float): Umbral de confianza para YOLO.
            iou_threshold (float): Umbral IoU para YOLO.
            gpu (bool): Usar aceleración GPU.
            model_storage_dir (str, optional): Directorio de descarga/caché de EasyOCR.
        """
        self.detector = YOLOOBBInference(
            model_path=yolo_model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device="cuda" if gpu else "cpu"
        )
        self.ocr_processor = EasyOCRInference(
            model_path=ocr_model_path,
            gpu=gpu,
            model_storage_dir=model_storage_dir
        )
        self.date_normalizer = DateNormalizer()

        print("Pipeline unificado inicializado:")
        print(f"  - Detector YOLO: {yolo_model_path}")
        print(f"  - OCR EasyOCR: {ocr_model_path}")

    def obtener_fecha_mas_futura(self, lista_fechas: List[str]) -> Optional[str]:
        """
        Filtra una lista de fechas en formato 'YYYY-MM-DD' y devuelve la más futura.

        Args:
            lista_fechas (list): Lista de cadenas de fecha.

        Returns:
            str, optional: La fecha más futura en formato 'YYYY-MM-DD' o None.
        """
        fechas_validas = []
        for fecha_str in lista_fechas:
            if fecha_str and fecha_str.strip():
                try:
                    fecha_obj = datetime.strptime(fecha_str.strip(), '%Y-%m-%d')
                    fechas_validas.append((fecha_obj, fecha_str.strip()))
                except (ValueError, TypeError):
                    continue

        if fechas_validas:
            # Seleccionar la tupla con el valor máximo de datetime
            fecha_max = max(fechas_validas, key=lambda x: x[0])
            return fecha_max[1]

        return None

    def execute_pipeline(self,
                         image_path: str,
                         output_folder: Optional[str] = None,
                         crops_root: Optional[str] = None,
                         class_filter: Optional[List[int]] = None,
                         save_crops: bool = True,
                         persist_images: bool = False) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo de detección e inferencia OCR.

        Args:
            image_path (str): Ruta al archivo de imagen.
            output_folder (str, optional): Carpeta para guardar imágenes anotadas con cajas.
            crops_root (str, optional): Carpeta para guardar los recortes (crops) de fechas.
            class_filter (list, optional): Clases de detección YOLO a filtrar (por defecto [0]).
            save_crops (bool): Si es True, realiza recortes y transcripción OCR.
            persist_images (bool): Guardar físicamente imágenes de salida en disco.

        Returns:
            dict: Resultados del pipeline (fecha final, textos OCR, fechas normalizadas).
        """
        if output_folder is None:
            output_folder = './test_outputs'
        if crops_root is None:
            crops_root = './test_crops'
        if class_filter is None:
            class_filter = [0]  # Clase 0 es exp_date

        results = {
            'image_path': image_path,
            'detection': None,
            'ocr_results': [],
            'normalized_dates': [],
            'fecha_vencimiento': None
        }

        # 1. Detección orientada en 4 rotaciones de la imagen original
        res, crops = self.detector.process_single_image_2(
            image_path=image_path,
            output_path=output_folder,
            class_filter=class_filter,
            save_crops=save_crops,
            crops_root=crops_root,
            persist_imgs=persist_images
        )
        results['detection'] = res

        # 2. Inferencia de OCR sobre las sub-imágenes recortadas (y sus 4 rotaciones locales)
        for i in range(len(crops)):
            if not crops[i]:
                continue

            for crop in crops[i]:
                try:
                    # Si el crop es más alto que largo, hacemos solo dos rotaciones (90 y 270 grados)
                    if crop.shape[0] > crop.shape[1]:
                        crops_rots_list = [
                            cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE),
                            cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        ]
                    else:
                        # Si es horizontal o cuadrado, hacemos las 4 rotaciones
                        crops_rots_list = [
                            crop,
                            cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE),
                            cv2.rotate(crop, cv2.ROTATE_180),
                            cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        ]
                    for crop_rot in crops_rots_list:
                        pred_text = self.ocr_processor.predict(crop_rot)
                        if pred_text.strip():
                            results['ocr_results'].append(pred_text)
                except Exception:
                    pass

        # 3. Normalizar textos a formato estándar de fecha
        norm_list = []
        for ocr_res in results['ocr_results']:
            norm_date = self.date_normalizer.normalize(ocr_res)
            if norm_date:
                norm_list.append(norm_date)
        results['normalized_dates'] = norm_list

        # 4. Obtener la fecha de vencimiento más futura
        fecha_mas_futura = self.obtener_fecha_mas_futura(results['normalized_dates'])
        results['fecha_vencimiento'] = fecha_mas_futura

        return results
