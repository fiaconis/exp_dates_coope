# -*- coding: utf-8 -*-
"""
Módulo que contiene funciones para el cálculo de métricas de rendimiento (CER, WER)
y la clase DateNormalizer para la estandarización de fechas obtenidas por el OCR.
"""

import re
from datetime import datetime
from typing import Optional, List
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
import pandas as pd


def levenshtein_distance(seq_a: List, seq_b: List) -> int:
    """
    Calcula la distancia Levenshtein entre dos secuencias.

    Args:
        seq_a (List): Primera secuencia.
        seq_b (List): Segunda secuencia.

    Returns:
        int: Distancia de edición.
    """
    len_a, len_b = len(seq_a), len(seq_b)
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a

    prev_row = list(range(len_b + 1))
    for i in range(1, len_a + 1):
        cur_row = [i] + [0] * len_b
        for j in range(1, len_b + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            cur_row[j] = min(
                prev_row[j] + 1,        # Borrado
                cur_row[j - 1] + 1,    # Inserción
                prev_row[j - 1] + cost  # Sustitución
            )
        prev_row = cur_row

    return prev_row[len_b]


def calculate_cer(pred: str, truth: str) -> float:
    """
    Calcula el Character Error Rate (CER).

    Args:
        pred (str): Cadena predicha.
        truth (str): Cadena real (ground truth).

    Returns:
        float: CER (de 0.0 a 1.0, o superior si hay demasiadas inserciones).
    """
    if len(truth) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return levenshtein_distance(list(pred), list(truth)) / len(truth)


def calculate_wer(pred: str, truth: str) -> float:
    """
    Calcula el Word Error Rate (WER).

    Args:
        pred (str): Cadena predicha.
        truth (str): Cadena real (ground truth).

    Returns:
        float: WER.
    """
    pred_words = pred.split()
    truth_words = truth.split()
    if len(truth_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    return levenshtein_distance(pred_words, truth_words) / len(truth_words)


class DateNormalizer:
    """
    Clase para normalizar y estructurar fechas en texto crudo extraído por OCR.
    Especializada en resolver formatos de vencimiento con mes en letras (español/inglés)
    y formatos numéricos compactos o con errores tipográficos.
    """

    # Mapeo de meses de Español a Inglés para el fallback de parseo
    ES_TO_EN_MONTHS = {
        "ene": "jan", "feb": "feb", "mar": "mar", "abr": "apr",
        "may": "may", "jun": "jun", "jul": "jul", "ago": "aug",
        "sep": "sep", "oct": "oct", "nov": "nov", "dic": "dec",
        "enero": "january", "febrero": "february", "marzo": "march",
        "abril": "april", "mayo": "may", "junio": "june",
        "julio": "july", "agosto": "august", "septiembre": "september",
        "octubre": "october", "noviembre": "november", "diciembre": "december"
    }

    # Ordenar las claves por longitud descendente para evitar colisiones en búsquedas parciales
    _MONTH_KEYS_SORTED = sorted(ES_TO_EN_MONTHS.keys(), key=len, reverse=True)
    _MONTH_RE_PATTERN = re.compile(
        r'\b(' + '|'.join(map(re.escape, _MONTH_KEYS_SORTED)) + r')\b',
        re.IGNORECASE
    )

    # Términos de ruido y prefijos típicos a limpiar antes de procesar la fecha
    _PRIMARY_PREFIX_TERMS = [
        "venc/exp", "consumir antes de", "date/venc", "antes del",
        "vence", "vento", "mento", "lote", "años", "meses", "fab", "exp", "val", "fv", "vto",
        "venc", "ncno", "nce", "t0", "al", "tu", "em", "v", "cpexp"
    ]
    _ESCAPED_PRIMARY_TERMS_SORTED = sorted(
        [re.escape(term) for term in _PRIMARY_PREFIX_TERMS],
        key=len, reverse=True
    )
    _MAIN_PREFIX_REGEX = r"^(?:" + "|".join(_ESCAPED_PRIMARY_TERMS_SORTED) + r")\s*[:.]?\s*"

    # Corrección de errores comunes de OCR en abreviaturas de meses
    MONTH_FIXES = {
        r"e[nñ]e|fne|enf": "ene",
        r"f3b|eeb|fe6": "feb",
        r"m4r|m@r|mkr|mnr|narzo|nar2o|mar2o": "mar",
        r"a6r|a@r|4br|48r|a8r": "abr",
        r"m4y|nay|m@y|mav": "may",
        r"jvn|tun|\bjun[a-zA-Z]+\b": "jun",
        r"ju1|jui|jl": "jul",
        r"aco|ac0|ag0|4go|0c1|2go|ago:|acosto": "ago",
        r"s3p|s@p|sep7|se7|5ep|se |sepp|ser|\bsep[a-zA-Z]+\b": "sep",
        r"o[ccl]t|0ct|ct": "oct",
        r"n0v": "nov",
        r"d1c|dc|dtc|dte": "dic",
        r"t[uú]n": "jun"
    }

    MONTH_NAMES_ES = sorted(
        ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic",
         "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"],
        key=len, reverse=True
    )

    # Patrones regex de descarte y prefijo
    patrones_regex = [
        r'^fecha\s+(?:vto|venc)[\.:]?\s*',
        r'^vencimiento[\.:]?\s*',
        r'^caduca[\.:]?\s*',
        r'^venc[e\.:]?\s*',
        r'^ven[c\.:]?\s*',
        r'^vto[\.:]?\s*',
        r'^valor[\.:]?\s*',
        r'^val[\.:;]?\s*',
        r'^exp[\.:]?\s*',
        r'^f[\.:]?\s*v[\.:]?\s*',
        r'^date[\.:]?\s*',
        r'\bfecha\s+(?:vto|venc)[\.:]?\s*',
        r'\bvencimiento[\.:]?\s*',
        r'\bcaduca[\.:]?\s*',
        r'\bvenc[\.:]?\s*',
        r'\bvto[\.:]?\s*',
        r'\bvalor[\.:]?\s*',
        r'\bval[\.:;]?\s*',
        r'exp[\.:]?\s*',
        r'\bf[\.:]?\s*v[\.:]?\s*',
        r'\bdate[\.:]?\s*',
        r'^vt0[\.:]?\s*',
    ]

    PREFIXES = [
        _MAIN_PREFIX_REGEX,
        *patrones_regex,
        r'^[:.]\s*',
        r'[a-zA-Z]+(?=:)',
        r"\.$",
        r'ence[:. ]',
        r'en[: ]',
        r'1to[: .]',
        r'[ ]b[ ]',
        r'[ ]L',
        r'[ ]l',
        r'[e]nce',
        r'L:'
    ]

    PREFIXES_ORDENADOS = sorted(PREFIXES, key=len, reverse=True)

    def __init__(self, year_cutoff: int = 2050, verbose: bool = False):
        """
        Inicializa el normalizador de fechas.

        Args:
            year_cutoff (int): Año máximo permitido para la validación de rango.
            verbose (bool): Activar logs detallados durante el análisis.
        """
        self.year_cutoff = year_cutoff
        self.verbose = verbose

    def _normalize_year(self, dt: datetime) -> Optional[datetime]:
        """Ajusta años de 2 dígitos y valida rangos lógicos."""
        year = dt.year
        if year < 100:
            year = year + 2000 if year < self.year_cutoff % 100 else year + 1900

        if year < 1990 or year > self.year_cutoff:
            return None

        return dt.replace(year=year)

    def _parse_and_normalize(self, text: str, df: bool = True) -> Optional[str]:
        """Intenta parsear la fecha y realiza fallback de traducción ES->EN."""
        text = text.strip()
        if not text:
            return None

        default_date = datetime(2000, 1, 1)

        # Primer intento en el locale del sistema
        try:
            parsed_dt = parse(text, dayfirst=df, default=default_date)
        except Exception:
            # Reintento traduciendo meses al inglés en caso de error de idioma
            def replace_month(match):
                return self.ES_TO_EN_MONTHS.get(match.group(0).lower(), match.group(0))

            english_text = self._MONTH_RE_PATTERN.sub(replace_month, text)
            try:
                parsed_dt = parse(english_text, dayfirst=df, default=default_date)
            except Exception:
                return None

        normalized_dt = self._normalize_year(parsed_dt)
        return normalized_dt.strftime('%Y-%m-%d') if normalized_dt else None

    def _get_last_day_of_month(self, dt_str: str) -> Optional[str]:
        """Dada una fecha (por defecto el día 1), calcula el último día de ese mes."""
        try:
            dt = datetime.strptime(dt_str, '%Y-%m-%d')
        except ValueError:
            return None

        # Desplazar al primer día del mes siguiente y restar un día
        dt_final = dt + relativedelta(months=1, day=1, days=-1)
        return dt_final.strftime('%Y-%m-%d')

    def _apply_aggressive_cleanup(self, text: str) -> str:
        """Remueve espacios entre separadores y junta dígitos separados."""
        text = re.sub(r'\s*([/.-])\s*', r'\1', text)
        text = re.sub(r'(\d)\s+(\d)(?=\s*[/.-])', r'\1\2', text)
        text = re.sub(r'\s{2,}', ' ', text).strip()
        return text

    def _path_meses_letras(self, text: str) -> Optional[str]:
        """Procesa textos que contienen nombres de meses literales."""
        text = re.sub(r'/{1,}', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text).strip()

        month_match = re.search(r'(' + '|'.join(self.MONTH_NAMES_ES) + r')', text)
        if not month_match:
            return None

        month_word = month_match.group(1)
        before_text = re.sub(r'\s', '', text[:month_match.start()]).replace('o', '0').replace('O', '0')
        after_text = re.sub(r'\s', '', text[month_match.end():]).replace('o', '0').replace('O', '0')

        if not before_text.isdigit():
            before_text = ''
        if not after_text.isdigit():
            after_text = ''

        day = None
        year = None

        # Día a la izquierda
        day_match = re.search(r'(\d{1,2})$', before_text)
        if day_match:
            day = day_match.group(1)

        # Año a la derecha o si no hay día, a la izquierda
        year_match = re.search(r'(\d{2}|\d{4})$', after_text)
        if year_match:
            year = year_match.group(1)
        elif not day:
            year_match = re.search(r'(\d{2}|\d{4})$', before_text)
            day = False
            if year_match:
                year = year_match.group(1)
        elif not year_match:
            year_match = re.search(r'(\d{2}|\d{4})$', before_text)
            day = False
            if year_match:
                year = year_match.group(1)

        # Caso 1: Solo Mes / Año
        if not day and year:
            date_str = f"1 {month_word} {year}"
            parsed_date_str = self._parse_and_normalize(date_str, df=True)
            if parsed_date_str:
                return self._get_last_day_of_month(parsed_date_str)

        # Caso 2: Día / Mes / Año
        if day and year:
            date_str = f"{day} {month_word} {year}"
            return self._parse_and_normalize(date_str, df=True)

        return None

    def _path_meses_numero(self, text: str) -> Optional[str]:
        """Procesa textos que contienen fechas completamente numéricas."""
        text = self._apply_aggressive_cleanup(text)
        numeric_text = text.replace('o', '0').replace('O', '0')
        separators = re.findall(r'[/.]', text)
        num_separators = len(separators)

        # Caso con 2 o más separadores (DD/MM/YYYY o DD.MM.YYYY)
        if num_separators >= 2:
            text_spaced = re.sub(r'[/.]', ' ', text)
            text_spaced = ' '.join(re.findall(r'\d+', text_spaced))
            return self._parse_and_normalize(text_spaced, df=True)

        # Caso con 1 separador
        elif num_separators == 1:
            sep = separators[0]
            parts = text.split(sep, 1)
            if len(parts) != 2:
                return None

            left, right = parts
            left = ''.join(re.findall(r'\d+', left))
            right = ''.join(re.findall(r'\d+', right))

            # Subcaso: MM/YYYY o M/YY (sin día, asumir fin de mes)
            if len(left) <= 2 and (len(right) == 2 or len(right) == 4):
                date_str = f"1 {left} {right}"
                parsed_date_str = self._parse_and_normalize(date_str, df=True)
                if parsed_date_str:
                    return self._get_last_day_of_month(parsed_date_str)

            right_digits_count = len(re.sub(r'\D', '', right))

            # Subcaso: DD/MMYY o DD/MMYYYY
            if right_digits_count >= 4 and len(left) <= 2:
                m = right[:2]
                y = right[2:]
                reconstructed = f"{left}/{m}/{y}"
                return self._parse_and_normalize(reconstructed, df=True)

            # Subcaso: DDMM/YY
            if len(left) == 4 and len(right) >= 2:
                d, m, y = left[0:2], left[2:4], right
                reconstructed = f"{d}/{m}/{y}"
                return self._parse_and_normalize(reconstructed, df=True)

        # Caso sin separadores (DDMMYYYY, DDMMYY o MMYY)
        elif num_separators == 0:
            numeric_text = re.sub(r'\D', '', text)

            if len(numeric_text) == 5:
                numeric_text = text.replace('o', '0').replace(' ', '')

            # DDMMYYYY
            if len(numeric_text) == 8:
                d, m, y = numeric_text[0:2], numeric_text[2:4], numeric_text[4:8]
                reconstructed = f"{d}/{m}/{y}"
                return self._parse_and_normalize(reconstructed, df=True)

            # MMYYYY o DDMMYY
            if len(numeric_text) == 6:
                if numeric_text[2:4] == "20":
                    m, y = numeric_text[0:2], numeric_text[2:6]
                    date_str = f"1 {m} {y}"
                    parsed_date_str = self._parse_and_normalize(date_str, df=True)
                    if parsed_date_str:
                        return self._get_last_day_of_month(parsed_date_str)
                else:
                    d, m, y = numeric_text[0:2], numeric_text[2:4], numeric_text[4:6]
                    reconstructed = f"{d}/{m}/{y}"
                    return self._parse_and_normalize(reconstructed, df=True)

            # MMYY
            elif len(numeric_text) == 4:
                m, y = numeric_text[0:2], numeric_text[2:4]
                date_str = f"1 {m} {y}"
                parsed_date_str = self._parse_and_normalize(date_str, df=True)
                if parsed_date_str:
                    return self._get_last_day_of_month(parsed_date_str)

            return self._parse_and_normalize(numeric_text, df=False)

        return None

    def extraer_solo_fecha(self, texto: str) -> str:
        """Extrae la fecha en formato ##/##/## de una cadena de texto."""
        if not texto or pd.isna(texto):
            return ""
        texto = str(texto)
        match = re.search(r'(\d{1,2}/\d{1,2}/\d{2,4})', texto)
        return match.group(1) if match else texto

    def truncar_desde_patron_regex(self, texto: str, patrones_regex: List[str]) -> str:
        """Remueve cualquier prefijo de coincidencia de los patrones configurados."""
        if not texto or pd.isna(texto):
            return ""
        texto = str(texto)
        for patron in patrones_regex:
            match = re.search(patron, texto, re.IGNORECASE)
            if match:
                return texto[match.end():].strip()
        return texto

    def normalize(self, raw_input: str) -> str:
        """
        Método principal que orquesta el pipeline de limpieza y parseo de la fecha.

        Args:
            raw_input (str): Texto crudo del OCR.

        Returns:
            str: Fecha normalizada 'YYYY-MM-DD' o cadena vacía si falla.
        """
        if self.verbose:
            print(f"Input original: {raw_input}")

        # Limpiar patrones conocidos y prefijos de ruido
        raw_input = self.truncar_desde_patron_regex(raw_input, self.patrones_regex)
        cleaned_text = raw_input.strip()

        for p in self.PREFIXES_ORDENADOS:
            cleaned_text = re.sub(p, "", cleaned_text, flags=re.IGNORECASE).strip()

        text = cleaned_text.lower()

        # Corregir typos de OCR en nombres de meses
        for wrong, right in self.MONTH_FIXES.items():
            text = re.sub(wrong, right, text, flags=re.IGNORECASE)

        # Enrutamiento según el tipo de mes (letra o número)
        found_month = any(month in text for month in self.MONTH_NAMES_ES)
        result = None

        if found_month:
            result = self._path_meses_letras(text)
        else:
            result = self._path_meses_numero(text)

        # Ajustes finales y reintentos en caso de fallar los paths principales
        if not result:
            result = self._parse_and_normalize(text, df=True)

        if not result:
            result = self._parse_and_normalize(text, df=False)

        if not result:
            fecha_extraida = self.extraer_solo_fecha(text)
            result = self._parse_and_normalize(fecha_extraida, df=False)

        return result if result else ""
