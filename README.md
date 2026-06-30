# API y CLI de Inferencia para Digitalización de Fechas de Vencimiento

Este paquete contiene la versión de producción simplificada y optimizada del pipeline de digitalización de fechas de vencimiento. Está listo para ser desplegado en el entorno del cliente final, ofreciendo ejecución tanto por consola (CLI) como mediante un servicio web REST (FastAPI).

---

## 1. Explicación Detallada del Algoritmo (Pipeline E2E)

El pipeline de digitalización de fechas de vencimiento consta de 4 etapas que aseguran robustez frente a diferentes rotaciones, inclinaciones del envase y ruidos tipográficos del OCR.

### Etapa 1: Detección Orientada y Multi-Rotación (YOLO OBB)
Para mitigar problemas de orientación de la imagen (por ejemplo, si la cámara tomó la foto en sentido vertical, lateral o de cabeza):
1. **Redimensionamiento**: La imagen de entrada es redimensionada a $600 \times 600$ píxeles.
2. **Rotación en 4 Ejes**: Se generan dinámicamente 4 copias de la imagen en rotaciones de $0^\circ, 90^\circ, 180^\circ$ y $270^\circ$.
3. **Inferencia**: El detector YOLO OBB (Oriented Bounding Boxes) analiza las 4 imágenes. Las cajas orientadas tienen en cuenta el ángulo de inclinación, localizando con precisión la región de la fecha.
4. **Warp de Perspectiva**: Para cada caja detectada, se calculan sus 4 esquinas inclinadas y se realiza una transformación de perspectiva (`cv2.getPerspectiveTransform` y `cv2.warpPerspective`) para recortar e inclinar la imagen y alinearla horizontalmente de forma perfecta.

### Etapa 2: Rotaciones Locales y Lectura de Caracteres (EasyOCR)
A nivel de crop (recorte), el texto puede seguir estando invertido o vertical (por la naturaleza de la caja):
1. **Validación de Forma**: Si el recorte es de tipo vertical (más alto que largo), se evalúan 2 rotaciones locales ($90^\circ$ y $270^\circ$). Si es horizontal o cuadrado, se evalúan 4 rotaciones locales ($0^\circ, 90^\circ, 180^\circ, 270^\circ$).
2. **EasyOCR Fine-Tuned**: Cada variante rotada se introduce en el motor EasyOCR personalizado, el cual utiliza un clasificador entrenado específicamente para el alfabeto reducido y tipografías típicas de fechas de vencimiento.
3. **Transcripción**: Se genera una lista de textos candidatos.

### Etapa 3: Normalización de Fechas (`DateNormalizer`)
El texto arrojado por el OCR es a menudo imperfecto (p. ej., lee `vto 12-oct-26`, `v3nc.20.10.25`, o `12/28`). La clase `DateNormalizer` aplica reglas determinísticas y heurísticas de limpieza:
1. **Limpieza de Prefijos**: Remueve ruidos y términos de control (como `vence`, `vto:`, `lote:`, `fv`, etc.).
2. **Corrección de Typos**: Traduce fallos comunes del OCR en los meses (por ejemplo, `f3b` $\rightarrow$ `feb`, `s3p` $\rightarrow$ `sep`).
3. **Identificación de Formato**:
   * **Mes con Letras**: Mapea meses en español a su estándar de parseo inglés y resuelve formatos como `Día Mes Año` o `Mes Año`. Si solo viene el mes y año (ej: *DIC 2026*), asume el último día de dicho mes (*2026-12-31*).
   * **Numérico Compacto**: Remueve separadores y analiza la longitud de la cadena de dígitos (`DDMMYYYY`, `DDMMYY`, `MMYYYY` o `MMYY`).
4. **Validación de Rango**: Las fechas candidatas se convierten a formato estándar ISO `YYYY-MM-DD` y se filtran si están fuera de rangos lógicos (años menores a 1990 o mayores a 2050).

### Etapa 4: Selección de la Fecha Final
El normalizador compila todas las fechas encontradas en formato `YYYY-MM-DD`. El orquestador selecciona **la fecha más lejana en el futuro** como la fecha de vencimiento definitiva (evitando confundirse con la fecha de elaboración/fabricación).

---

## 2. Estructura del Proyecto Entregado

```text
vencimientos_delivery/
├── configs/
│   └── deploy_config.yaml      # Parámetros de umbral de YOLO y rutas a modelos
│
├── deployment/
│   ├── __init__.py
│   ├── inference.py            # Orquestador del pipeline completo (OCRPipeline)
│   └── app.py                  # API FastAPI para exponer el pipeline vía web
│
├── src/
│   ├── __init__.py
│   ├── models.py               # Wrappers e inferencia unitaria YOLO OBB y EasyOCR
│   └── evaluation.py           # Normalizador de fechas y cálculo de distancias de edición
│
├── models/
│   ├── yolo/
│   │   └── best.pt             # Pesos del modelo de detección de fecha YOLO OBB
│   └── ocr/
│       └── easyocr_fixed3.pth  # Pesos del modelo de reconocimiento OCR (fine-tuned)
│
├── scripts/
│   └── predict.py              # CLI ejecutable para procesar imágenes locales
│
├── requirements.txt            # Dependencias de librerías Python
└── README.md                   # Esta documentación técnica
```

---

## 3. Instalación de Dependencias

Se requiere Python 3.10+ (se recomienda el entorno virtual conda `coope_all` o equivalente).

1. Abre tu consola y navega al directorio del proyecto `vencimientos_delivery`.
2. Activa tu entorno virtual e instala las dependencias mediante pip:
```bash
pip install -r requirements.txt
```

---

## 4. Guía de Ejecución

El entregable ofrece tres formas de uso principales:

### Método A: Ejecución en Línea de Comandos (CLI)
Para procesar una imagen local, guardar los recortes y visualizar la imagen anotada, ejecuta `predict.py` pasando la ruta de la imagen:

```bash
python scripts/predict.py --image "/ruta/de/imagen.jpg" --output_dir "./predictions"
```

*   **`--image`**: Ruta a la imagen de entrada (obligatorio).
*   **`--output_dir`**: Directorio donde se guardará la imagen anotada y los cultivos de fecha (opcional, por defecto `./predictions`).
*   **`--gpu`**: Añade este flag si deseas forzar el uso de la GPU (CUDA).

#### Salida Estándar (stdout):
El script imprime directamente un JSON estructurado útil para ser integrado en otros procesos o lenguajes:
```json
{
    "success": true,
    "filename": "imagen.jpg",
    "ocr_texts": ["VTO:25.DIC.27", "25.DIC.27"],
    "normalized_dates": ["2027-12-25", "2027-12-25"],
    "expiration_date": "2027-12-25",
    "output_annotated_dir": "/absolute/path/vencimientos_delivery/predictions"
}
```

---

### Método B: Servicio Web API REST (FastAPI)
Para levantar el servidor web de FastAPI y procesar solicitudes de red desde otras aplicaciones o servidores:

```bash
uvicorn deployment.app:app --host 0.0.0.0 --port 8000
```

*   **`--host`**: IP a escuchar. `0.0.0.0` expone el servicio a cualquier red pública/privada de la máquina.
*   **`--port`**: Puerto web (por defecto `8000`).

#### Endpoints Principales:
1. **`GET /health`**: Verifica que el servicio esté inicializado y saludable. Devuelve `{"status": "healthy"}`.
2. **`POST /predict`**:
   * **Parámetro**: Envía un archivo de imagen en formato Multipart/Form-data bajo el nombre `file`.
   * **Respuesta (JSON)**:
     ```json
     {
       "success": true,
       "filename": "producto_caja_12.jpg",
       "ocr_texts": ["10/2026"],
       "normalized_dates": ["2026-10-31"],
       "expiration_date": "2026-10-31"
     }
     ```
   * **Documentación Interactiva (Swagger)**: Puedes interactuar con la API directamente desde tu navegador visitando [http://localhost:8000/docs](http://localhost:8000/docs).

#### Consulta desde Clientes Ligeros (Consumo Veloz)
Para evitar el retraso de carga de librerías y modelos en llamadas individuales (overhead de ~5s), puedes consultar la API activa usando herramientas ligeras nativas o scripts mínimos de Python sin dependencias pesadas:

*   **Bash / curl**:
    ```bash
    curl -X POST -F "file=@/ruta/de/imagen.jpg" http://localhost:8000/predict
    ```
*   **PowerShell (Windows)**:
    ```powershell
    Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Form @{file=[System.IO.File]::OpenRead("C:\ruta\de\imagen.jpg")}
    ```

---

### Método C: CLI Interactivo (Bucle de Inferencia Optimizado)
Si no deseas usar un servidor de red HTTP, puedes ejecutar el CLI interactivo. Este carga los modelos en memoria una sola vez en el arranque y luego procesa rutas de imágenes leídas secuencialmente desde la entrada estándar (`stdin`), entregando respuestas inmediatas en formato JSON por `stdout`.

```bash
python scripts/predict_interactive.py --output_dir "./predictions"
```

Al iniciar, imprimirá:
`{"status": "ready", "message": "Modelos cargados. Ingrese rutas de imágenes línea por línea."}`

A partir de ese momento, puedes escribir o redirigir (pipe) rutas de imágenes:
```
/ruta/a/la/imagen1.jpg
/ruta/a/la/imagen2.jpg
exit
```

Cada ruta enviada producirá de forma instantánea una salida JSON de una sola línea en `stdout`:
```json
{"success": true, "filename": "imagen1.jpg", "ocr_texts": ["25.DIC.27"], "normalized_dates": ["2027-12-25"], "expiration_date": "2027-12-25"}
```

---

## 5. Recomendaciones para Puesta en Producción

*   **Uso de GPU**: Aunque el pipeline soporta ejecución en CPU, se recomienda utilizar hardware con GPU compatible con **NVIDIA CUDA** para optimizar los tiempos de respuesta.
*   **Ajuste de Umbrales**: Si el detector pasa por alto fechas o genera falsos positivos, puedes ajustar los valores `conf_threshold` y `iou_threshold` en `configs/deploy_config.yaml`.
