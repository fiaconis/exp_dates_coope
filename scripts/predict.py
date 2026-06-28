# -*- coding: utf-8 -*-
"""
Script CLI utilitario para correr inferencias y predicciones en producción.
Recibe una ruta de imagen y ejecuta el pipeline de detección YOLO + EasyOCR,
escribiendo el resultado en JSON por stdout y guardando las anotaciones visuales.
"""

import os
import sys
import json
import argparse
import yaml

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Asegurar que el directorio raíz de vencimientos_clean_try esté en el PATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deployment.inference import OCRPipeline


def main():
    parser = argparse.ArgumentParser(description="CLI de Inferencia del Pipeline de Vencimientos.")
    default_cfg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "deploy_config.yaml")
    parser.add_argument("--image", type=str, required=True, help="Ruta al archivo de imagen de entrada")
    parser.add_argument("--output_dir", type=str, default="./predictions", help="Carpeta donde guardar imágenes anotadas y crops")
    parser.add_argument("--deploy_cfg", type=str, default=default_cfg, help="Ruta al deploy_config.yaml")
    parser.add_argument("--gpu", action="store_true", help="Forzar el uso de GPU si está disponible")
    args = parser.parse_args()

    # Validar existencia de la imagen
    if not os.path.exists(args.image):
        print(json.dumps({"success": False, "error": f"La imagen {args.image} no existe."}), file=sys.stderr)
        sys.exit(1)

    # Cargar archivo de configuración de despliegue
    if not os.path.exists(args.deploy_cfg):
        print(json.dumps({"success": False, "error": f"Configuración no encontrada en {args.deploy_cfg}"}), file=sys.stderr)
        sys.exit(1)

    with open(args.deploy_cfg, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    inf_cfg = config["inference"]

    # Inicialización del pipeline
    try:
        pipeline = OCRPipeline(
            yolo_model_path=inf_cfg["yolo_model_path"],
            ocr_model_path=inf_cfg["ocr_model_path"],
            conf_threshold=inf_cfg.get("conf_threshold", 0.25),
            iou_threshold=inf_cfg.get("iou_threshold", 0.7),
            gpu=args.gpu or inf_cfg.get("gpu", False)
        )
    except Exception as e:
        print(json.dumps({"success": False, "error": f"Error inicializando pipeline: {str(e)}"}), file=sys.stderr)
        sys.exit(1)

    # Configuración de salidas
    output_folder = args.output_dir
    crops_folder = os.path.join(output_folder, "crops")

    # Ejecutar pipeline
    try:
        results = pipeline.execute_pipeline(
            image_path=args.image,
            output_folder=output_folder,
            crops_root=crops_folder,
            save_crops=True,
            persist_images=True
        )

        # Imprimir JSON estructurado de salida para integración con otros sistemas
        output_json = {
            "success": True,
            "filename": os.path.basename(args.image),
            "ocr_texts": results.get("ocr_results", []),
            "normalized_dates": results.get("normalized_dates", []),
            "expiration_date": results.get("fecha_vencimiento", None),
            "output_annotated_dir": os.path.abspath(output_folder)
        }
        print(json.dumps(output_json, indent=4, ensure_ascii=False))

    except Exception as e:
        print(json.dumps({"success": False, "error": f"Fallo al ejecutar inferencia: {str(e)}"}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
