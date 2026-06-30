# -*- coding: utf-8 -*-
"""
Módulo que contiene las clases de envoltura (wrappers) para la inferencia
con los modelos YOLO OBB (Oriented Bounding Boxes) y EasyOCR personalizado.
"""

import os
from typing import Optional, Dict, Tuple, List
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from ultralytics import YOLO
import easyocr


class YOLOOBBInference:
    """
    Envoltura para la ejecución del modelo YOLO orientado a cajas delimitadoras
    alineadas por ángulo (OBB - Oriented Bounding Boxes).
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.25,
                 iou_threshold: float = 0.7, device: str = "cpu"):
        """
        Inicializa el detector YOLO OBB.

        Args:
            model_path (str): Ruta al checkpoint .pt del modelo entrenado.
            conf_threshold (float): Umbral de confianza mínimo.
            iou_threshold (float): Umbral de IoU para la supresión de no máximos (NMS).
            device (str): Dispositivo de ejecución ('cpu', 'cuda', etc.).
        """
        self.device = device
        if str(self.device).lower() == "cpu":
            # Forzar CPU ocultando variables de CUDA para bibliotecas subyacentes
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        self.model = YOLO(model_path)
        try:
            self.model.to(self.device)
        except Exception:
            pass

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def preprocess_image(self, image_path: str) -> List[np.ndarray]:
        """
        Carga una imagen, la redimensiona a 600x600 y genera 4 variantes rotadas
        (0, 90, 180, 270 grados) para realizar la detección robusta multi-orientación.

        Args:
            image_path (str): Ruta a la imagen.

        Returns:
            list: Lista de 4 arreglos numpy (imágenes en formato BGR, redimensionadas a 600x600).
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen en la ruta: {image_path}")

        # Redimensionar la imagen a 600x600 para la inferencia YOLO OBB
        image = cv2.resize(image, (600, 600))

        # Rotaciones de 90 grados en sentido de las agujas del reloj
        rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotated_180 = cv2.rotate(image, cv2.ROTATE_180)
        rotated_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return [image, rotated_90, rotated_180, rotated_270]

    def run_inference(self, image: np.ndarray, class_filter: Optional[List[int]] = None):
        """
        Ejecuta la inferencia de YOLO sobre una imagen específica.

        Args:
            image (np.ndarray): Imagen en formato BGR.
            class_filter (list, optional): Clases a filtrar.

        Returns:
            list: Resultados devueltos por la API de ultralytics.
        """
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            classes=class_filter,
            device=self.device,
        )
        return results

    def draw_obb_detections(self, image: np.ndarray, results) -> np.ndarray:
        """
        Dibuja los bounding boxes orientados detectados sobre la imagen.

        Args:
            image (np.ndarray): Imagen BGR original.
            results: Objeto de resultados de YOLO.

        Returns:
            np.ndarray: Imagen anotada con rectángulos orientados y clases.
        """
        image_with_detections = image.copy()

        for result in results:
            # Detecciones orientadas (OBB)
            if hasattr(result, "obb") and result.obb is not None:
                try:
                    obb_boxes = result.obb.xyxyxyxy.cpu().numpy()
                    confidences = result.obb.conf.cpu().numpy()
                    class_ids = result.obb.cls.cpu().numpy()
                except Exception:
                    continue

                unique_classes = np.unique(class_ids) if len(class_ids) else [0]
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_classes)))

                for obb, conf, cls_id in zip(obb_boxes, confidences, class_ids):
                    obb_points = obb.reshape(4, 2).astype(np.int32)
                    color_rgb = (colors[int(np.where(unique_classes == cls_id)[0][0])][:3] * 255).astype(int)
                    # Convertir RGB de matplotlib a BGR para OpenCV
                    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))

                    cv2.polylines(image_with_detections, [obb_points], True, color_bgr, 2)
                    label = f"{self.model.names[int(cls_id)]} {conf:.2f}"
                    text_x = int(obb_points[0][0])
                    text_y = int(max(0, obb_points[0][1] - 10))
                    cv2.putText(
                        image_with_detections, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2, cv2.LINE_AA
                    )

            # Fallback: Detecciones de caja regular (Axis-Aligned)
            elif hasattr(result, "boxes") and result.boxes is not None:
                try:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()
                except Exception:
                    continue

                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = box.astype(int)
                    color_bgr = (255, 255, 0)
                    cv2.rectangle(image_with_detections, (x1, y1), (x2, y2), color_bgr, 2)
                    label = f"{self.model.names[int(cls_id)]} {conf:.2f}"
                    cv2.putText(
                        image_with_detections, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2, cv2.LINE_AA
                    )

        return image_with_detections

    def process_single_image_2(self,
                               image_path: str,
                               output_path: Optional[str] = None,
                               class_filter: Optional[List[int]] = None,
                               save_crops: bool = False,
                               crops_root: Optional[str] = None,
                               persist_imgs: bool = False):
        """
        Procesa una sola imagen, aplicando detección de OBB a sus 4 rotaciones.

        Args:
            image_path (str): Ruta al archivo de imagen.
            output_path (str, optional): Carpeta para imágenes resultantes anotadas.
            class_filter (list, optional): Lista de IDs de clase a detectar.
            save_crops (bool): Si es True, recorta las regiones de interés.
            crops_root (str, optional): Carpeta raíz de guardado de crops.
            persist_imgs (bool): Si se guardan físicamente en disco.

        Returns:
            tuple: (results_list, crops_list)
        """
        if class_filter is None:
            class_filter = [0]

        image_list = self.preprocess_image(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        if save_crops and crops_root is None:
            if output_path:
                crops_root = (output_path if os.path.isdir(output_path)
                              else os.path.join(os.path.dirname(output_path) or '.', 'crops'))
            else:
                crops_root = os.path.join('.', 'crops')

        results = []
        crop_list = []

        rotations = ['0', '90', '180', '270']
        for img, rot in zip(image_list, rotations):
            preds = self.run_inference(img, class_filter=class_filter)
            result_img = self.draw_obb_detections(img, preds)

            if output_path and persist_imgs:
                # Si output_path no tiene extensión de archivo, asumimos que es una carpeta
                _, ext = os.path.splitext(output_path)
                if not ext:
                    os.makedirs(output_path, exist_ok=True)

                out_file = (os.path.join(output_path, f"result_{base_name}_rot{rot}.jpg")
                            if (os.path.isdir(output_path) or output_path.endswith(os.sep))
                            else output_path)
                os.makedirs(os.path.dirname(out_file), exist_ok=True)
                cv2.imwrite(out_file, result_img)

            if save_crops:
                counters, temp_crops = self.save_crops_by_class(
                    img, [preds], image_path, crops_root, persist_imgs
                )
            else:
                temp_crops = []

            results.append([preds])
            crop_list.append(temp_crops)

        return results, crop_list

    def save_crops_by_class(self, image: np.ndarray, results, image_path: str,
                            output_root: str, persist_imgs: bool = False):
        """
        Alinea y recorta la región delimitada por la OBB (caja orientada) usando
        transformaciones de perspectiva.

        Args:
            image (np.ndarray): Imagen original.
            results (list): Resultados de la inferencia.
            image_path (str): Ruta de imagen base para nombrar archivos.
            output_root (str): Carpeta destino.
            persist_imgs (bool): Si se persisten las imágenes de recorte.

        Returns:
            tuple: (counters, list con imágenes recortadas en formato numpy)
        """
        def order_points(pts):
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]  # Superior izquierdo
            rect[2] = pts[np.argmax(s)]  # Inferior derecho
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]  # Superior derecho
            rect[3] = pts[np.argmax(diff)]  # Inferior izquierdo
            return rect

        def dist(a, b):
            return np.linalg.norm(a - b)

        class_map = {0: "exp_date", 1: "lote_date", 2: "mfg_date", 3: "unknown_date"}
        if persist_imgs:
            os.makedirs(output_root, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        counters = {v: 0 for v in class_map.values()}
        crops_sublist = []

        for result in results[0]:
            if hasattr(result, "obb") and result.obb is not None:
                try:
                    obb_boxes = result.obb.xyxyxyxy.cpu().numpy()
                    class_ids = result.obb.cls.cpu().numpy()
                except Exception:
                    continue

                for obb, cls_id in zip(obb_boxes, class_ids):
                    cls = int(cls_id)
                    folder = class_map.get(cls, "unknown_date")

                    pts = np.array(obb, dtype=np.float32).reshape(4, 2)
                    rect = order_points(pts)

                    width_a = dist(rect[2], rect[3])
                    width_b = dist(rect[1], rect[0])
                    max_w = int(max(width_a, width_b))

                    height_a = dist(rect[1], rect[2])
                    height_b = dist(rect[0], rect[3])
                    max_h = int(max(height_a, height_b))

                    if max_w <= 0 or max_h <= 0:
                        continue

                    # Coordenadas destino para el warp de perspectiva
                    dst = np.array([
                        [0, 0],
                        [max_w - 1, 0],
                        [max_w - 1, max_h - 1],
                        [0, max_h - 1]
                    ], dtype=np.float32)

                    try:
                        transform_matrix = cv2.getPerspectiveTransform(rect, dst)
                        crop = cv2.warpPerspective(image, transform_matrix, (max_w, max_h))
                    except Exception:
                        # Fallback a recorte axis-aligned en caso de error matemático
                        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
                        if w <= 0 or h <= 0:
                            continue
                        crop = image[y:y+h, x:x+w]

                    if crop.size == 0:
                        continue

                    if persist_imgs:
                        out_dir = os.path.join(output_root, folder)
                        os.makedirs(out_dir, exist_ok=True)
                        counters[folder] += 1
                        out_file = os.path.join(out_dir, f"{base_name}_{folder}_{counters[folder]:03d}.jpg")

                    # Guardar y mantener cultivos tanto horizontales como verticales
                    if persist_imgs:
                        cv2.imwrite(out_file, crop)
                    crops_sublist.append(crop)

        return counters, crops_sublist


class EasyOCRInference:
    """
    Clase para instanciar el motor OCR personalizado, soportando el reemplazo
    del cabezal de predicción con pesos reentrenados y alfabeto personalizado.
    """

    def __init__(self, model_path: str, gpu: bool = False, model_storage_dir: Optional[str] = None):
        """
        Inicializa EasyOCR con checkpoints personalizados.

        Args:
            model_path (str): Ruta al checkpoint .pth reentrenado.
            gpu (bool): Usar aceleración por GPU.
            model_storage_dir (str, optional): Directorio base de modelos de EasyOCR.
        """
        self.model_path = model_path
        self.gpu = gpu
        self.device = "cuda" if gpu and torch.cuda.is_available() else "cpu"
        if not model_storage_dir and model_path:
            model_storage_dir = os.path.dirname(os.path.abspath(model_path))
        self.model_storage_dir = model_storage_dir
        self._load()

    def _load(self):
        """Carga el modelo y aplica los pesos personalizados del clasificador."""
        checkpoint = torch.load(self.model_path, map_location="cpu")
        self.config = checkpoint.get("config", {})
        self.alphabet = checkpoint.get("alphabet", {})

        # Construir tabla de mapeo de índice a caracter
        if isinstance(self.alphabet, dict):
            try:
                self.idx_to_char = {int(k): v for k, v in self.alphabet.items()}
            except Exception:
                self.idx_to_char = self.alphabet
        elif isinstance(self.alphabet, list):
            self.idx_to_char = {i + 1: c for i, c in enumerate(self.alphabet)}
        else:
            self.idx_to_char = {}

        languages = self.config.get("language", ["en"])
        self.reader = easyocr.Reader(
            languages,
            gpu=(self.device != "cpu"),
            verbose=False,
            download_enabled=False,
            model_storage_directory=self.model_storage_dir,
        )
        self.recognizer = getattr(self.reader, "recognizer", self.reader)
        self.recognizer.to(self.device)
        self.recognizer.eval()

        pred = checkpoint.get("prediction_weights", {})
        if "weight" in pred:
            w = pred["weight"]
            b = pred.get("bias", None)
            self._replace_prediction_layer(w, b)
        else:
            # Respaldar carga del estado completo (state_dict) del modelo
            state = checkpoint.get("model_state_dict", {})
            if isinstance(state, dict):
                def _strip(k):
                    return k.replace("module.", "").replace("model.", "").replace("recognizer.", "")
                state_fixed = {_strip(k): v for k, v in state.items()}
                try:
                    self.recognizer.load_state_dict(state_fixed, strict=False)
                except Exception:
                    pass

        self.recognizer.to(self.device)
        self.recognizer.eval()

    def _replace_prediction_layer(self, w: torch.Tensor, b: torch.Tensor = None):
        """Sustituye la capa final (Linear) del reconocedor con la entrenada."""
        if not isinstance(w, torch.Tensor):
            w = torch.tensor(w)
        if b is not None and not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        in_feats = w.shape[1]
        out_feats = w.shape[0]

        new_pred = nn.Linear(in_feats, out_feats, bias=(b is not None))
        new_pred.weight.data.copy_(w)
        if b is not None:
            new_pred.bias.data.copy_(b)

        # Vincular
        if hasattr(self.recognizer, "Prediction"):
            setattr(self.recognizer, "Prediction", new_pred)
        else:
            # Fallback dinámico buscando la primera capa Lineal
            replaced = False
            for name, module in self.recognizer.named_modules():
                if isinstance(module, nn.Linear):
                    parent = self.recognizer
                    parts = name.split(".")
                    *ps, last = parts
                    try:
                        for p in ps:
                            parent = getattr(parent, p)
                        setattr(parent, last, new_pred)
                        replaced = True
                        break
                    except Exception:
                        continue
            if not replaced:
                setattr(self.recognizer, "Prediction", new_pred)

        new_pred.to(self.device)

    def _preprocess(self, image_path, input_size=None) -> torch.Tensor:
        """Preprocesa la imagen de entrada convirtiéndola a tensor normalized."""
        if input_size is None:
            h = self.config.get("image_height", 64)
            w = self.config.get("image_width", 256)
            input_size = (h, w)

        if isinstance(image_path, (str, os.PathLike)):
            img = Image.open(str(image_path)).convert("L")
        elif isinstance(image_path, np.ndarray):
            if image_path.ndim == 3 and image_path.shape[2] == 3:
                img = Image.fromarray(cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY))
            else:
                img = Image.fromarray(image_path)
        else:
            img = image_path.convert("L")

        tf = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        t = tf(img).unsqueeze(0).to(self.device)
        return t

    def _decode(self, logits: torch.Tensor) -> str:
        """Decodifica los logits de la inferencia utilizando descodificación greedy CTC."""
        if logits.dim() == 3 and logits.shape[0] < logits.shape[1]:
            logits = logits.permute(1, 0, 2)

        logp = logits.log_softmax(dim=-1)
        preds = logp.argmax(dim=-1).squeeze(0).cpu().numpy()

        result = []
        prev = None
        blank = 0

        for idx in preds:
            arr = np.asarray(idx)
            if arr.size == 1:
                i = int(arr.item())
            else:
                i = int(arr.flatten()[0])

            if i != blank and i != prev:
                ch = self.idx_to_char.get(i, "")
                if ch:
                    result.append(ch)
            prev = i

        return "".join(result)

    def predict(self, image_path: str, input_size=None) -> str:
        """
        Ejecuta la transcripción de texto en un recorte de imagen.

        Args:
            image_path (str o np.ndarray): Ruta de la imagen o recorte en numpy array.
            input_size (tuple, optional): Tamaño de redimensionamiento de entrada.

        Returns:
            str: Texto predicho.
        """
        t = self._preprocess(image_path, input_size=input_size)
        self.recognizer.eval()
        with torch.no_grad():
            try:
                logits = self.recognizer(t, None)
            except TypeError:
                logits = self.recognizer(t)
        return self._decode(logits)
