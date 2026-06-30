# -*- coding: utf-8 -*-
"""
Script CLI utilitario interactivo para correr inferencias continuas en producción.
Carga los modelos en memoria una sola vez y entra en un bucle interactivo de lectura.
Recibe rutas de imágenes por stdin, ejecuta la inferencia de manera inmediata, y escribe
el JSON resultante en stdout.
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
    parser = argparse.ArgumentParser(description="CLI Interactivo del Pipeline de Vencimientos.")
    default_cfg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "deploy_config.yaml")
    parser.add_argument("--output_dir", type=str, default="./predictions", help="Carpeta base donde guardar crops y resultados temporales")
    parser.add_argument("--deploy_cfg", type=str, default=default_cfg, help="Ruta al deploy_config.yaml")
    parser.add_argument("--gpu", action="store_true", help="Forzar el uso de GPU si está disponible")
    args = parser.parse_args()

    # Validar existencia de la configuración
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

    # Configuración de carpetas de salida
    output_folder = args.output_dir
    crops_folder = os.path.join(output_folder, "crops")

    print(json.dumps({"status": "ready", "message": "Modelos cargados. Ingrese rutas de imágenes línea por línea."}), flush=True)

    # Bucle interactivo leyendo de stdin
    try:
        for line in sys.stdin:
            image_path = line.strip()
            if not image_path:
                continue
            if image_path.lower() in ("exit", "quit"):
                break

            if not os.path.exists(image_path):
                print(json.dumps({"success": False, "error": f"La imagen {image_path} no existe."}), flush=True)
                continue

            try:
                results = pipeline.execute_pipeline(
                    image_path=image_path,
                    output_folder=output_folder,
                    crops_root=crops_folder,
                    save_crops=True,
                    persist_images=False  # No persistimos anotaciones visuales completas para máxima velocidad
                )

                output_json = {
                    "success": True,
                    "filename": os.path.basename(image_path),
                    "ocr_texts": results.get("ocr_results", []),
                    "normalized_dates": results.get("normalized_dates", []),
                    "expiration_date": results.get("fecha_vencimiento", None),
                }
                print(json.dumps(output_json, ensure_ascii=False), flush=True)

            except Exception as e:
                print(json.dumps({"success": False, "error": f"Fallo en inferencia: {str(e)}"}), flush=True)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
