"""
Actualiza automáticamente los mapas de un ciclón del Pacífico:
  1) <nombre>_trayectoria.png : trayectoria observada (best track NHC), pronóstico y
     cono del NHC, con distancia/rumbo al punto de referencia más cercano.
  2) <nombre>_lluvia.png      : lluvia acumulada pronosticada (modelo vía Open-Meteo)
  3) <nombre>_viento.png      : rachas máximas de viento pronosticadas (misma descarga)
     y máximos por estado.

Fuentes:
  - NHC CurrentStorms.json       (estado actual y aviso público)
  - NHC ATCF best track (b-deck) (posiciones observadas)
  - NHC GIS 5day_latest.zip      (pronóstico y cono)
  - Open-Meteo                   (precipitación pronosticada del modelo elegido)

Uso:  python actualizar_mapas.py
"""
import base64
import io
import json
import logging
import math
import os
import re
import sys
import textwrap
import time
import zipfile
import datetime as dt
from pathlib import Path

import numpy as np
import requests
import shapefile  # pyshp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
from matplotlib.lines import Line2D
from matplotlib.path import Path as MPath
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.patheffects as pe

# ============================ CONFIGURACIÓN ============================
SEGUIR_AUTOMATICO = True        # sigue solo al ciclón activo más cercano a Salina Cruz (dentro o cerca de la jurisdicción)
STORM_ID = ""                   # se llena solo; para fijar uno a mano: SEGUIR_AUTOMATICO = False y p. ej. "ep222026"
STORM_NAME = ""
ARCHIVO_BASE = "mapa"           # nombres fijos de las imágenes: mapa_lluvia.png y mapa_viento.png
BASE = Path(__file__).resolve().parent
# ---- PC de la oficina o respaldo en la nube (GitHub Actions) ----
# El MISMO archivo corre en los dos lados; en la nube se detecta solo.
EN_NUBE = os.environ.get("GITHUB_ACTIONS") == "true"
OUT_DIR = BASE / os.environ.get("MAPAS_SALIDA", "salida")   # en la nube: carpeta "estado" del repositorio
GUARDAR_HISTORIAL = not EN_NUBE        # copia con fecha/hora en salida/historial
QPF_MODEL = "ecmwf_ifs025"
PROB_MODEL = "gfs_seamless"     # modelo que aporta la probabilidad de lluvia (ensamble GFS); "" = no mostrarla      # alternativas: "gfs_seamless", "best_match"
QPF_DIAS = 4                    # días de lluvia acumulada a partir de hoy
QPF_RES = 0.25                  # resolución de la malla en grados (solo sobre los estados de interés)
ZONA_HORARIA = "America/Mexico_City"
N_DISTANCIAS = 4                     # puntos de referencia a medir desde la posición actual
GENERAR_TRAYECTORIA = False          # el mapa de lluvia ya incluye la trayectoria
INTERVALO_FORZADO_H = 3              # cambios menores: se acumulan y se genera imagen a lo más cada 3 h
                                     # (sin cambios, la imagen se mantiene igual)
UMBRAL_KT = 15                       # cambio de intensidad de un ciclón que se considera muy significativo
UMBRAL_PROB = 1                     # cambio (puntos %) en la probabilidad de una zona que se considera muy significativo
MOSTRAR_ZONAS_BAJA = True            # zonas de baja presión del NHC (Tropical Weather Outlook)
MOSTRAR_CONO = False                 # cono de incertidumbre (apagado para un mapa más limpio)
ESTADOS_INTERES = ["Guerrero", "Oaxaca", "Chiapas"]  # estados del mapa de lluvia
N_PUNTOS_POR_ESTADO = 3              # puntos de mayor acumulado por estado
SEPARACION_PUNTOS = 1.2              # separación mínima entre puntos (grados)
GENERAR_LLUVIA = True                # mapa de lluvia acumulada
GENERAR_VIENTO = True                # mapa de rachas máximas de viento (misma descarga, sin consultas extra)
ISTMO_DESTACAR = True                # recuadro y sección propia para el Istmo de Tehuantepec
ISTMO_CAJA = (-95.75, -94.30, 15.95, 17.30)   # lon_min, lon_max, lat_min, lat_max
ISTMO_LOCALIDADES = [                # (nombre, lat, lon, lado de la etiqueta: "d" derecha / "i" izquierda)
    ("Salina Cruz", 16.17, -95.20, "i"), ("Tehuantepec", 16.32, -95.24, "i"),
    ("Juchitán", 16.43, -95.02, "d"), ("Ixtepec", 16.56, -95.10, "i"),
    ("La Ventosa", 16.55, -94.95, "d"), ("Matías Romero", 16.88, -95.04, "d"),
]
ETIQUETAS_MAPA = "numero"             # puntos numerados: "completa", "numero", "insignia" o "guias"
ISTMO_EN_GOLFO = True                # ampliación del Istmo sobre el Golfo de México (deja libre el Pacífico)
GOLFO_CAJA = (-96.4, -91.1, 18.9)     # lon_min, lon_max, lat_min del espacio para la ampliación en el Golfo
DISENO = "clasico"                   # acomodo: "clasico", "lateral", "tablero" o "marino"
ISTMO_EN_PANEL = True                # True: ampliación debajo del panel (océano y costa libres); False: dentro del mapa
ISTMO_AMPLIACION_TAM = (0.33, 0.40)  # ancho y alto de la ampliación (fracción del mapa); se ubica sola en un hueco
ANCHO_MAX_ETIQUETAS = 20.5           # si el mapa abarca más grados de longitud, los valores van solo en el panel
EXTENSION_MAX = (-118.5, -86.0, 2.0, 33.0)  # hasta dónde puede crecer el mapa (toda la República) para que
                                            # quepan completos los sistemas de la jurisdicción
JURISDICCION = (-104.0, -89.0, 5.0, 21.0)   # lon_min, lon_max, lat_min, lat_max: área frente a Guerrero–Chiapas
                                     # (incluye la Bahía de Campeche). Solo se muestran sistemas dentro de ella.
# Datos privados FUERA del código (el código queda en un repositorio público):
#  - en la PC: archivo config_mapas.json junto a este script, p. ej. {"ntfy_tema": "..."}
#  - en la nube: Secrets del repositorio (NTFY_TEMA y MAPAS_TOKEN)
def _config_local():
    ruta = BASE / "config_mapas.json"
    try:
        return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    except Exception:
        return {}


NTFY_TEMA = os.environ.get("NTFY_TEMA") or _config_local().get("ntfy_tema", "")
MONITOR_HTML = "monitor.html"        # página de monitoreo (PC y celular) en la carpeta de salida; "" = no generarla
# Publicación de la página en GitHub Pages (mismo usuario y token que el panel de presas)
PUBLICAR_GITHUB = True
CONFIG_GITHUB = Path(r"D:\NIVEL DE PRESAS Y RIOS\config_github.json")   # se usan "usuario" y "token"
GITHUB_REPO = "monitor-mapas-sc"     # repositorio público para esta página
# Respaldo en la nube: entra solo cuando la PC lleva más de este tiempo sin publicar la página
UMBRAL_PC_MIN = 70                   # la PC publica a lo más cada 30 min; 70 min sin hacerlo = PC caída o sin internet
ETIQUETA_NUBE = "[nube]"             # prefijo de los commits que hace la nube (para distinguirlos de los de la PC)
PUBLICAR_ESTADO_MIN = 30             # sin imágenes nuevas, el estado se publica a lo más cada 30 min        # tema de ntfy.sh para recibir los mapas ("" = no enviar)

# Puntos de referencia (lat, lon) usados para distancias
REFS = {   # puntos de referencia en la costa de la jurisdicción (Guerrero a Chiapas)
    "Zihuatanejo": (17.64, -101.55), "Técpan de Galeana": (17.22, -100.63), "Acapulco": (16.85, -99.88),
    "Marquelia": (16.58, -98.82), "Punta Maldonado": (16.32, -98.57), "L. de Chacahua": (15.98, -97.68),
    "Puerto Escondido": (15.86, -97.07), "Puerto Ángel": (15.67, -96.49), "Bahías de Huatulco": (15.75, -96.13),
    "Salina Cruz": (16.17, -95.20), "Puerto Arista": (15.94, -93.81), "Puerto Chiapas": (14.71, -92.40),
}
LOCALIDADES = {  # cabeceras de Guerrero, Oaxaca y Chiapas (lat, lon)
    "Chilpancingo": (17.55, -99.50), "Tlapa": (17.55, -98.58), "Iguala": (18.35, -99.54),
    "Taxco": (18.56, -99.60), "Ometepec": (16.69, -98.41), "Cd. Altamirano": (18.36, -100.67),
    "Petatlán": (17.54, -101.27), "Atoyac": (17.21, -100.43), "San Marcos": (16.80, -99.39),
    "Tlacoachistlahuaca": (16.81, -98.31), "Huamuxtitlán": (17.81, -98.56),
    "Oaxaca": (17.06, -96.72), "Juchitán": (16.43, -95.02), "Tuxtepec": (18.09, -96.12),
    "Huajuapan": (17.80, -97.78), "Pinotepa Nacional": (16.34, -98.05), "Pochutla": (15.74, -96.47),
    "Miahuatlán": (16.33, -96.60), "Matías Romero": (16.88, -95.04), "Tlaxiaco": (17.27, -97.68),
    "Juxtlahuaca": (17.33, -98.01), "Putla": (17.03, -97.93), "Ixtlán": (17.33, -96.49),
    "Huautla": (18.13, -96.84), "María Lombardo": (17.45, -95.43), "Zacatepec": (17.20, -95.82),
    "Tuxtla Gutiérrez": (16.75, -93.12), "San Cristóbal": (16.74, -92.64), "Tapachula": (14.90, -92.26),
    "Tonalá": (16.09, -93.75), "Comitán": (16.25, -92.13), "Palenque": (17.51, -91.98),
    "Pichucalco": (17.51, -93.12), "Arriaga": (16.24, -93.90), "Motozintla": (15.37, -92.25),
    "Huixtla": (15.14, -92.47), "Ocosingo": (16.91, -92.09), "Villaflores": (16.23, -93.27),
    "Mapastepec": (15.44, -92.90), "Cintalapa": (16.69, -93.72), "Bochil": (17.00, -92.89),
}
CIUDADES_MAPA = [  # (ya no se usa; se marcan solo los puntos medidos)
    "Salina Cruz", "Puerto Ángel", "Acapulco", "Lázaro Cárdenas",
                 "Manzanillo", "Puerto Vallarta", "Mazatlán"]

HEADERS = {"User-Agent": "mapas-ciclon-uso-personal"}
URL_CURRENT = "https://www.nhc.noaa.gov/CurrentStorms.json"
URL_BTK = "https://ftp.nhc.noaa.gov/atcf/btk/b{id}.dat"
URL_FCST = "https://www.nhc.noaa.gov/gis/forecast/archive/{id}_5day_latest.zip"
URL_OM = "https://api.open-meteo.com/v1/forecast"
URL_TWO = "https://www.nhc.noaa.gov/xgtwo/gtwo_shapefiles.zip"
URL_META = "https://api.open-meteo.com/data/{modelo}/static/meta.json"
URL_BTK_DIR = "https://ftp.nhc.noaa.gov/atcf/btk/"
# ======================================================================

log = logging.getLogger("mapas")
RUMBOS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
          "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"]


# ----------------------------- utilidades -----------------------------
def http_get(url, params=None, intentos=4, timeout=40):
    for i in range(intentos):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if r.status_code == 429:
                motivo = r.text[:150].replace("\n", " ")
                log.warning("Límite de solicitudes (%s); esperando 65 s (intento %d/%d)", motivo, i + 1, intentos)
                if "daily" in motivo.lower():
                    raise RuntimeError("Se agotó el límite diario de Open-Meteo")
                time.sleep(65)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            log.warning("Fallo %s (intento %d/%d): %s", url, i + 1, intentos, str(e)[:120])
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"No se pudo descargar {url}")


def merc(lon, lat):
    return lon, math.degrees(math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)))


def dist_rumbo(desde, hacia):
    """Distancia (km) y rumbo cardinal desde 'desde' hacia 'hacia' (lat, lon)."""
    la1, lo1, la2, lo2 = map(math.radians, (*desde, *hacia))
    d = 2 * 6371 * math.asin(math.sqrt(math.sin((la2 - la1) / 2) ** 2 +
                                       math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2))
    y = math.sin(lo2 - lo1) * math.cos(la2)
    x = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(lo2 - lo1)
    b = (math.degrees(math.atan2(y, x)) + 360) % 360
    return d, RUMBOS[round(b / 22.5) % 16]


def fmt_dist(d):
    """Distancia en kilómetros y millas náuticas."""
    return f"≈{d:,.0f} Km · {d / 1.852:,.0f} NM"


def distancia_area(anillos):
    """Punto de la costa más cercano al BORDE del área de una zona de baja presión: (lugar, km, rumbo)."""
    mejor = None
    for k, (rla, rlo) in REFS.items():
        if any(MPath(np.array(a_)).contains_point((rlo, rla)) for a_ in anillos):
            return k, 0.0, ""
        for a_ in anillos:
            for (x1, y1), (x2, y2) in zip(a_[:-1], a_[1:]):
                for s in np.linspace(0, 1, 12):
                    d, r = dist_rumbo((rla, rlo), (y1 + (y2 - y1) * s, x1 + (x2 - x1) * s))
                    if mejor is None or d < mejor[1]:
                        mejor = (k, d, r)
    return mejor


REF_PRINCIPAL = "Salina Cruz"          # referencia fija para los sistemas cercanos (además de las 2 más cercanas)
REF_PRINCIPAL_MAX_NM = 500             # más lejos de esto, el sistema ya no se refiere a Salina Cruz
C_REF_PRINCIPAL = "#C2185B"


def refs_para(lat, lon, n_extra=2):
    """Salina Cruz siempre, más los n_extra puntos de la lista más cercanos al sistema."""
    todos = sorted(((k, v, *dist_rumbo(v, (lat, lon))) for k, v in REFS.items()), key=lambda t: t[2])
    # Salina Cruz solo mientras el sistema esté a 700 NM o menos; si se aleja más, se omite,
    # y si vuelve a acercarse (por ejemplo, al recurvar) se retoma sola en la siguiente actualización
    princ = [x for x in todos if x[0] == REF_PRINCIPAL and x[2] <= REF_PRINCIPAL_MAX_NM * 1.852]
    return princ + [x for x in todos if x[0] != REF_PRINCIPAL][:n_extra]


def fecha_hora_local(p):
    """'22 oct 06 h' a partir de la hora UTC del punto."""
    if p.get("t"):
        t = p["t"] - dt.timedelta(hours=6)
        return f"{t.day} {MESES[t.month - 1]} {t.hour:02d} h"
    return f"+{p['tau']} h" if p.get("tau") is not None else ""


def dibujar_trayectoria(ax, previos, actual, fc, baja=None):
    """Trayectoria con la simbología de los mapas de trayectoria: tramo previo punteado en gris,
    pronóstico coloreado por categoría y símbolo de ciclón en cada punto con fecha, categoría y nudos."""
    if not actual:
        return
    # etapa de baja presión (antes de ser ciclón): línea punteada café que llega al ciclón, con la B en su origen
    etapa_baja = [p for p in previos if p.get("tipo") in ("LO", "DB", "WV")]
    ciclon_prev = [p for p in previos if p.get("tipo") not in ("LO", "DB", "WV")]
    destino = (ciclon_prev[0] if ciclon_prev else actual)
    origen = None
    if etapa_baja:
        serie_b = etapa_baja + [destino]
        origen = etapa_baja[0]
    elif baja:
        # si el área del NHC quedó casi encima del ciclón, la B se pone al inicio de su trayectoria
        if math.hypot(baja["lat"] - actual["lat"], baja["lon"] - actual["lon"]) < 1.5 and ciclon_prev:
            serie_b = [ciclon_prev[0]]
            origen = ciclon_prev[0]
        else:
            serie_b = [baja, destino]
            origen = baja
    else:
        serie_b = []
    if serie_b:
        xy = [merc(p["lon"], p["lat"]) for p in serie_b]
        ax.plot([p[0] for p in xy], [p[1] for p in xy], ls=(0, (1, 2.2)), color="#854F0B", lw=2.0, zorder=4)
        for x, y in xy[1:-1]:
            ax.plot(x, y, "o", ms=3, color="#854F0B", zorder=4)
        xo, yo = xy[0]
        num = (baja or {}).get("num", "")
        ax.text(xo, yo, f"B{num}", fontsize=8.5, fontweight="bold", color="#854F0B", ha="center", va="center",
                zorder=9, bbox=dict(boxstyle="circle,pad=0.22", fc="white", ec="#854F0B", lw=1.6))
        etq = "Baja presión" + (f" ({origen['invest']})" if origen.get("invest") else "")
        if origen.get("t"):
            etq += f"\n{fecha_hora_local(origen)}"
        tb = ax.annotate(etq, (xo, yo), xytext=(0, 12), textcoords="offset points", fontsize=7.4,
                         ha="center", va="bottom", color="#854F0B", zorder=9)
        tb.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])
    xy_prev = [merc(p["lon"], p["lat"]) for p in ciclon_prev] + [merc(actual["lon"], actual["lat"])]
    if len(xy_prev) > 1:
        ax.plot([p[0] for p in xy_prev], [p[1] for p in xy_prev], ls=(0, (2, 2)), color="#888780", lw=1.8, zorder=4)
        for x, y in xy_prev[:-1]:
            ax.plot(x, y, "o", ms=3.5, color="#888780", zorder=4)
    serie = [actual] + [p for p in fc if p.get("kt") is not None]
    for a, b in zip(serie[:-1], serie[1:]):
        (x1, y1), (x2, y2) = merc(a["lon"], a["lat"]), merc(b["lon"], b["lat"])
        ax.plot([x1, x2], [y1, y2], color=COLOR_CAT[codigo_ciclon(b["kt"])], lw=3, zorder=4, solid_capstyle="round")
    ult = (actual["lon"], actual["lat"])
    for i, p in enumerate(serie[1:], start=1):
        x, y = merc(p["lon"], p["lat"])
        dibujar_simbolo_ciclon(ax, x, y, p["kt"], "", tam=8)
        if math.hypot(p["lon"] - ult[0], p["lat"] - ult[1]) < 0.9:
            continue   # muy cerca del punto anterior etiquetado: se omite para no encimar
        ult = (p["lon"], p["lat"])
        # etiqueta del lado izquierdo del avance (hacia mar abierto)
        a = serie[i - 1]
        dx, dy = p["lon"] - a["lon"], p["lat"] - a["lat"]
        nx, ny = -dy, dx
        if ny > 0:   # que quede hacia el sur/oeste
            nx, ny = -nx, -ny
        nn = math.hypot(nx, ny) or 1
        off = (nx / nn * 14, ny / nn * 14)
        tt = ax.annotate(f"{fecha_hora_local(p)}\n{codigo_ciclon(p['kt'])} · {p['kt']:.0f} kt", (x, y), xytext=off,
                         textcoords="offset points", fontsize=7.2, ha="right" if off[0] < 0 else "left",
                         va="top" if off[1] < 0 else "bottom", color="#2c2c2a", zorder=9)
        tt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])


def refs_cercanas(lat, lon, n):
    return sorted(((k, v, *dist_rumbo(v, (lat, lon))) for k, v in REFS.items()),
                  key=lambda t: t[2])[:n]


ABREV = {"Guerrero": "Gro.", "Oaxaca": "Oax.", "Chiapas": "Chis."}


def localidad(lat, lon, polys=None):
    """Localidad más cercana, buscando primero dentro del mismo estado."""
    todos = {**LOCALIDADES, **REFS}
    if polys:
        dentro = {k: v for k, v in todos.items()
                  if any(MPath(p).contains_point((v[1], v[0])) for p in polys)}
        todos = dentro or todos
    return min(todos, key=lambda k: dist_rumbo(todos[k], (lat, lon))[0])


def ref_cercana(lat, lon):
    nombre, (d, r) = min(((k, dist_rumbo(v, (lat, lon))) for k, v in REFS.items()),
                         key=lambda t: t[1][0])
    return f"{fmt_dist(d)} al {r} de {nombre}"


def categoria(kt):
    if kt is None:
        return ""
    if kt < 34: return "DT"
    if kt < 64: return "TT"
    if kt < 83: return "cat. 1"
    if kt < 96: return "cat. 2"
    if kt < 113: return "cat. 3"
    if kt < 137: return "cat. 4"
    return "cat. 5"


def numero_ciclon(id_nhc):
    """'ep222026' -> '22-E' (número que el NHC asigna a la depresión)."""
    m = re.match(r"(al|ep|cp)(\d{2})\d{4}", (id_nhc or "").lower())
    return f"{int(m.group(2))}-{dict(al='L', ep='E', cp='C')[m.group(1)]}" if m else ""


COLOR_CAT = {"DT": "#2E86DE", "TT": "#1E9E5A", "H1": "#E8B400", "H2": "#F08C00",
             "H3": "#E8472F", "H4": "#C2185B", "H5": "#7B1FA2"}


def codigo_ciclon(kt):
    """DT, TT o H1…H5 según la intensidad (nudos)."""
    kt = kt or 0
    for lim, c in ((137, "H5"), (113, "H4"), (96, "H3"), (83, "H2"), (64, "H1"), (34, "TT")):
        if kt >= lim:
            return c
    return "DT"


def etiqueta_ciclon(kt, nombre, num):
    """Texto junto al símbolo: la depresión se identifica por su número; tormenta y huracán por su nombre."""
    c = codigo_ciclon(kt)
    nombre_ok = nombre and not re.match(r"^(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
                                        r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
                                        r"thirty|potential|invest|\d)", nombre.strip().lower())
    if c == "DT" or not nombre_ok:
        return c, f"{c} {num}".strip()
    return c, f"{c} {nombre.strip().title()}"


_EXT_SIMBOLO = 2.45   # radio máximo de los brazos (el centro tiene radio 1)


def _ruta_brazos():
    """Brazos del símbolo de ciclón tropical (hemisferio norte: giran en sentido antihorario)."""
    from matplotlib.path import Path as P
    th = np.radians(np.linspace(-40, 110, 30))
    r = np.linspace(0.55, 2.3, 30)                 # espiral que se abre
    w = np.linspace(0.95, 0.0, 30)                 # grosor que se adelgaza hacia la punta
    verts, codes = [], []
    for rot in (0.0, math.pi):
        ext = [((ri + wi / 2) * math.cos(a + rot), (ri + wi / 2) * math.sin(a + rot)) for a, ri, wi in zip(th, r, w)]
        inte = [((ri - wi / 2) * math.cos(a + rot), (ri - wi / 2) * math.sin(a + rot)) for a, ri, wi in zip(th, r, w)]
        pts = ext + inte[::-1]
        verts += pts + [pts[0]]
        codes += [P.MOVETO] + [P.LINETO] * (len(pts) - 1) + [P.CLOSEPOLY]
    verts += [(_EXT_SIMBOLO, 0), (-_EXT_SIMBOLO, 0), (0, _EXT_SIMBOLO), (0, -_EXT_SIMBOLO)]   # fija la escala
    codes += [P.MOVETO, P.MOVETO, P.MOVETO, P.MOVETO]
    return P(verts, codes)


def dibujar_simbolo_ciclon(ax, x, y, kt, texto, color_txt="#2c2c2a", tam=13):
    """DT: círculo; TT: símbolo con brazos y centro abierto; H: símbolo con centro relleno. Con su etiqueta."""
    c = codigo_ciclon(kt)
    col = COLOR_CAT[c]
    ms_brazos = tam * _EXT_SIMBOLO * 1.55
    if c == "DT":
        ax.plot(x, y, "o", ms=tam * 1.3, mfc=col, mec="white", mew=1.5, zorder=8)
    else:
        ax.plot(x, y, marker=_ruta_brazos(), ms=ms_brazos, mfc=col, mec="white", mew=0.8, ls="none", zorder=8)
        if c == "TT":   # tormenta tropical: centro abierto
            ax.plot(x, y, "o", ms=tam * 0.95, mfc="white", mec=col, mew=2.6, zorder=9)
        else:           # huracán: centro relleno
            ax.plot(x, y, "o", ms=tam * 1.05, mfc=col, mec="white", mew=1.2, zorder=9)
    x0, x1 = ax.get_xlim()
    izq = (x - x0) / (x1 - x0) > 0.82          # cerca del borde derecho: etiqueta a la izquierda
    t = ax.annotate(texto, (x, y), xytext=(-tam * 1.9 if izq else tam * 1.9, 0), textcoords="offset points",
                    fontsize=9.5, fontweight="bold", va="center", ha="right" if izq else "left",
                    color=color_txt, zorder=9)
    t.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])


def dibujar_baja(ax, x, y, num, mb=None):
    """Zona de baja presión: letra B con el número que le asigna el NHC."""
    ax.text(x, y, f"B{num}", fontsize=10.5, fontweight="bold", color="#854F0B", ha="center", va="center",
            zorder=9, bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec="#854F0B", lw=2))
    if mb:
        tm = ax.annotate(f"{mb} mb", (x, y), xytext=(0, -17), textcoords="offset points", fontsize=9,
                         fontweight="bold", color="#854F0B", ha="center", va="top", zorder=9)
        tm.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])


def nombre_categoria(kt):
    if kt is None:
        return "sin dato"
    if kt >= 64:
        return f"huracán {categoria(kt)}"
    return "tormenta tropical" if kt >= 34 else "depresión tropical"


MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def fecha_local(t_utc):
    t = t_utc - dt.timedelta(hours=6)  # hora del centro de México (sin horario de verano)
    return f"{t.day} {MESES[t.month - 1]} {t:%H} h"


# ----------------------------- datos NHC -----------------------------
OTRAS_TORMENTAS = []  # demás ciclones activos (se llena en tormenta_activa)


def tormenta_activa():
    try:
        data = http_get(URL_CURRENT).json()
    except Exception as e:
        log.warning("CurrentStorms.json no disponible: %s", e)
        return None
    global STORM_ID, STORM_NAME
    OTRAS_TORMENTAS.clear()
    if SEGUIR_AUTOMATICO:
        # el ciclón principal es el más cercano a Salina Cruz entre los que están en la jurisdicción
        # o a pocos grados de ella (su pronóstico decide después si se muestra)
        J = JURISDICCION
        cand = []
        for s in data.get("activeStorms", []):
            try:
                la, lo = float(s["latitudeNumeric"]), float(s["longitudeNumeric"])
            except (KeyError, TypeError, ValueError):
                continue
            if J[0] - 6 <= lo <= J[1] + 4 and J[2] - 3 <= la <= J[3] + 3:
                d = dist_rumbo(REFS[REF_PRINCIPAL], (la, lo))[0]
                cand.append((0 if en_jurisdiccion(lo, la) else 1, d, s))
        if cand:
            s = min(cand, key=lambda c: (c[0], c[1]))[2]
            STORM_ID, STORM_NAME = s.get("id", "").lower(), s.get("name", "") or numero_ciclon(s.get("id", ""))
        else:
            STORM_ID, STORM_NAME = "", ""
    for s in data.get("activeStorms", []):
        if s.get("id", "").lower() != STORM_ID and (not STORM_NAME or s.get("name", "").lower() != STORM_NAME.lower()):
            try:
                OTRAS_TORMENTAS.append(dict(nombre=s.get("name", "?"), clase=s.get("classification", ""),
                                            num=numero_ciclon(s.get("id", "")),
                                            kt=_num(s.get("intensity")),
                                            lat=float(s["latitudeNumeric"]), lon=float(s["longitudeNumeric"])))
            except (KeyError, TypeError, ValueError):
                pass
    for s in data.get("activeStorms", []):
        if STORM_ID and s.get("id", "").lower() == STORM_ID:
            return s
    return None


def best_track():
    if not STORM_ID:
        return []
    txt = http_get(URL_BTK.format(id=STORM_ID)).text
    pts = {}
    for linea in txt.splitlines():
        p = [c.strip() for c in linea.split(",")]
        if len(p) < 11 or not p[6] or not p[7]:
            continue
        t = dt.datetime.strptime(p[2], "%Y%m%d%H")
        lat = int(p[6][:-1]) / 10 * (1 if p[6][-1] == "N" else -1)
        lon = int(p[7][:-1]) / 10 * (-1 if p[7][-1] == "W" else 1)
        pts[t] = dict(t=t, lat=lat, lon=lon, kt=int(p[8] or 0), mb=_num(p[9]), tipo=p[10])
    return [pts[k] for k in sorted(pts)]


def _lector(z, sufijo):
    nombres = {n.lower(): n for n in z.namelist()}
    base = next((n[:-4] for n in nombres if n.endswith(sufijo + ".shp")), None)
    if base is None:
        return None
    return shapefile.Reader(shp=io.BytesIO(z.read(nombres[base + ".shp"])),
                            dbf=io.BytesIO(z.read(nombres[base + ".dbf"])),
                            shx=io.BytesIO(z.read(nombres[base + ".shx"])))


def pronostico():
    if not STORM_ID:
        return [], [], ""
    z = zipfile.ZipFile(io.BytesIO(http_get(URL_FCST.format(id=STORM_ID)).content))
    pts_r = _lector(z, "_5day_pts")
    puntos, cono, adv = [], [], ""
    if pts_r:
        for sr in pts_r.iterShapeRecords():
            rec = {k.upper(): v for k, v in sr.record.as_dict().items()}
            lon, lat = sr.shape.points[0]
            tau = rec.get("TAU")
            puntos.append(dict(lon=lon, lat=lat, tau=int(tau) if tau not in (None, "") else None,
                               kt=_num(rec.get("MAXWIND")),
                               fecha=str(rec.get("FLDATELBL") or rec.get("DATELBL") or "")))
            adv = str(rec.get("ADVISNUM") or adv)
        if all(p["tau"] is not None for p in puntos):
            puntos.sort(key=lambda p: p["tau"])
    pgn_r = _lector(z, "_5day_pgn")
    if pgn_r:
        for shp in pgn_r.shapes():
            idx = list(shp.parts) + [len(shp.points)]
            for a, b in zip(idx[:-1], idx[1:]):
                cono.append(shp.points[a:b])
    return puntos, cono, adv


def _num(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def texto_lluvia_nhc(storm):
    try:
        url = storm["publicAdvisory"]["url"]
        txt = re.sub(r"<[^>]+>", " ", http_get(url).text)
        m = re.search(r"RAINFALL:(.*?)(?:\n\s*\n|STORM SURGE:|SURF:|WIND:|TORNADOES:)", txt, re.S)
        return " ".join(m.group(1).split()) if m else ""
    except Exception as e:
        log.info("Sin texto de lluvia del NHC: %s", e)
        return ""


# ----------------------------- lluvia (Open-Meteo) -----------------------------
def lluvia_modelo(lon_min, lon_max, lat_min, lat_max):
    lons = np.arange(lon_min, lon_max + 1e-6, QPF_RES)
    lats = np.arange(lat_min, lat_max + 1e-6, QPF_RES)
    LON, LAT = np.meshgrid(lons, lats)
    planos = list(zip(LAT.ravel(), LON.ravel()))
    total = np.zeros(len(planos))
    viento = np.full(len(planos), np.nan)
    rachas = np.full(len(planos), np.nan)
    prob = np.full(len(planos), np.nan)
    modelos = QPF_MODEL + ("," + PROB_MODEL if PROB_MODEL else "")
    variables = "precipitation_sum,wind_speed_10m_max,wind_gusts_10m_max" + (
        ",precipitation_probability_max" if PROB_MODEL else "")
    for i in range(0, len(planos), 100):
        lote = planos[i:i + 100]
        r = http_get(URL_OM, params=dict(
            latitude=",".join(f"{a:.2f}" for a, _ in lote),
            longitude=",".join(f"{b:.2f}" for _, b in lote),
            daily=variables, forecast_days=QPF_DIAS,
            models=modelos, timezone=ZONA_HORARIA)).json()
        if isinstance(r, dict):
            r = [r]
        for j, item in enumerate(r):
            d = item.get("daily", {})
            def serie(var, modelo):   # con varios modelos, Open-Meteo agrega el sufijo del modelo
                return [v for v in (d.get(f"{var}_{modelo}") or d.get(var) or []) if v is not None]
            total[i + j] = sum(serie("precipitation_sum", QPF_MODEL))
            for nom, arr in (("wind_speed_10m_max", viento), ("wind_gusts_10m_max", rachas)):
                vv = serie(nom, QPF_MODEL)
                if vv:
                    arr[i + j] = max(vv)
            if PROB_MODEL:
                pp = serie("precipitation_probability_max", PROB_MODEL)
                if pp:
                    prob[i + j] = max(pp)   # probabilidad máxima de lluvia en el periodo
        if i + 100 < len(planos):
            time.sleep(11)  # Open-Meteo cuenta cada punto; ~550 puntos/min queda bajo el límite de 600
    return (LON, LAT, total.reshape(LON.shape), viento.reshape(LON.shape), rachas.reshape(LON.shape),
            prob.reshape(LON.shape))


def valor_en(LON, LAT, Z, lat, lon):
    """Interpolación bilineal en la malla regular del modelo."""
    lons, lats = LON[0, :], LAT[:, 0]
    if not (lons[0] <= lon <= lons[-1] and lats[0] <= lat <= lats[-1]):
        return float("nan")
    i = min(max(np.searchsorted(lats, lat) - 1, 0), len(lats) - 2)
    j = min(max(np.searchsorted(lons, lon) - 1, 0), len(lons) - 2)
    fy = (lat - lats[i]) / (lats[i + 1] - lats[i]); fx = (lon - lons[j]) / (lons[j + 1] - lons[j])
    return float((1 - fy) * ((1 - fx) * Z[i, j] + fx * Z[i, j + 1]) + fy * ((1 - fx) * Z[i + 1, j] + fx * Z[i + 1, j + 1]))


def en_jurisdiccion(lon, lat):
    return JURISDICCION[0] <= lon <= JURISDICCION[1] and JURISDICCION[2] <= lat <= JURISDICCION[3]


def dominio_jurisdiccion(base, actual, fc, otras, zonas, margen=0.8, rel_min=0.48, rel_max=0.85):
    """Ajusta el encuadre a los estados + los sistemas presentes en el área marina de la jurisdicción.
    Se incluyen completos: estados, ciclones (posición y pronóstico dentro del área) y zonas de baja presión.
    Para no deformar el mapa se ensancha hacia los lados, aunque abarque toda la República."""
    J = EXTENSION_MAX
    lo0, lo1, la0, la1 = base
    serie = ([actual] if actual else []) + list(fc or [])
    # si el ciclón afecta la jurisdicción, se muestra toda su trayectoria pronosticada
    obl = ([(p_["lon"], p_["lat"]) for p_ in serie]
           if any(en_jurisdiccion(p_["lon"], p_["lat"]) for p_ in serie) else [])
    obl += [(t_["lon"], t_["lat"]) for t_ in otras]
    opc = []
    for zn in zonas:
        pz = [c for a_ in zn["anillos"] for c in a_]
        la_c, lo_c, _ = centro_zona(zn)
        obl.append((lo_c, la_c))
        opc += [(min(max(c[0], J[0]), J[1]), min(max(c[1], J[2]), J[3])) for c in pz]
    obl = [(min(max(lo, J[0]), J[1]), min(max(la, J[2]), J[3])) for lo, la in obl]   # dentro de los límites
    for lo, la in obl:
        lo0, lo1 = min(lo0, lo - margen), max(lo1, lo + margen)
        la0, la1 = min(la0, la - margen), max(la1, la + margen)
    lim_lo0, lim_lo1 = min(lo0, J[0]), max(lo1, J[1])   # hasta dónde se puede ensanchar
    if opc:
        o_lo0 = min(min(c[0] for c in opc) - margen * 0.5, lo0); o_lo1 = max(max(c[0] for c in opc) + margen * 0.5, lo1)
        o_la0 = min(min(c[1] for c in opc) - margen * 0.5, la0); o_la1 = max(max(c[1] for c in opc) + margen * 0.5, la1)
        lo0, lo1 = max(o_lo0, lim_lo0), min(o_lo1, lim_lo1)
        la1 = o_la1
        # la zona completa; solo si ni ensanchando a toda la República cabe, se recorta su parte más al sur
        ancho_pos = lim_lo1 - lim_lo0
        la0 = max(o_la0, la1 - rel_max * ancho_pos)
        la0 = min(la0, min(la for _, la in obl) - margen)   # el centro de cada zona y los ciclones siempre
    ancho, alto = lo1 - lo0, la1 - la0
    if alto > rel_max * ancho:              # muy alto: ensanchar hacia los lados (sin pasar del límite)
        falta = alto / rel_max - ancho
        lo0, lo1 = lo0 - falta / 2, lo1 + falta / 2
        if lo0 < lim_lo0:
            lo1 += lim_lo0 - lo0; lo0 = lim_lo0
        if lo1 > lim_lo1:
            lo0 -= lo1 - lim_lo1; lo1 = lim_lo1
        lo0 = max(lo0, lim_lo0)
    elif alto < rel_min * ancho:            # muy bajo: agregar mar hacia el sur
        la0 = max(la1 - rel_min * ancho, J[2])
    lo0, lo1 = max(lo0, J[0]), min(lo1, J[1])
    la0, la1 = max(la0, J[2]), min(la1, J[3])
    return (round(lo0, 2), round(lo1, 2), round(la0, 2), round(la1, 2))


def fmt_valor(v, unidad, sep=" · "):
    """Texto de un valor: viento en km/h y nudos, lluvia en mm."""
    if unidad == "km/h":
        return f"{v:.0f} km/h{sep}{v / 1.852:.0f} kt"
    return f"{v:.0f} {unidad}"


def fmt_extra(cfg, v2=None, prob=None):
    """Complemento entre paréntesis: viento sostenido o probabilidad de lluvia."""
    if v2 is not None and not math.isnan(v2):
        return f" (sost. {fmt_valor(v2, cfg['unidad'])})"
    if prob is not None and not math.isnan(prob):
        return f" (prob. {prob:.0f}%)"
    return ""


def maximos_por_estado(estados, LON, LAT, Z):
    pts = np.c_[LON.ravel(), LAT.ravel()]
    z = Z.ravel()
    res = {}
    for nombre, polys in estados.items():
        dentro = np.zeros(len(pts), bool)
        for p in polys:
            dentro |= MPath(p).contains_points(pts)
        if dentro.any():
            k = np.argmax(np.where(dentro, z, -1))
            res[nombre] = (z[k], pts[k][1], pts[k][0])
    return res


# ----------------------------- zonas de baja presión (NHC TWO) -----------------------------
def _leer_shp(z, palabra):
    nombres = {n.lower(): n for n in z.namelist()}
    base = next((n[:-4] for n in nombres if palabra in n and n.endswith(".shp")), None)
    if not base or base + ".dbf" not in nombres:
        return []
    lector = shapefile.Reader(shp=io.BytesIO(z.read(nombres[base + ".shp"])),
                              dbf=io.BytesIO(z.read(nombres[base + ".dbf"])),
                              shx=io.BytesIO(z.read(nombres[base + ".shx"])) if base + ".shx" in nombres else None)
    return [({k.upper(): v for k, v in sr.record.as_dict().items()}, sr.shape) for sr in lector.iterShapeRecords()]


def _clave(rec):
    """Identificador de la perturbación en los archivos del NHC (cuenca + número de área), si existe."""
    b = next((str(v).strip() for k, v in rec.items() if k.startswith("BASIN")), "")
    a = next((str(v).strip() for k, v in rec.items() if k == "AREA" or k.startswith("AREA_")), "")
    return f"{b}|{a}" if a else None


def _normalizar_anillo(pts):
    """Longitudes en -180..180 y continuas (un área del Pacífico central puede cruzar la línea de cambio
    de fecha y mezclar +179 con -179, lo que la estiraría de un lado a otro del mapa)."""
    pts = [((x - 360) if x > 180 else x, y) for x, y in pts]
    if pts and max(x for x, _ in pts) - min(x for x, _ in pts) > 180:
        pts = [((x - 360) if x > 0 else x, y) for x, y in pts]
    return pts


def zona_en_jurisdiccion(zn):
    """La zona afecta la jurisdicción: tamaño razonable, su centro cerca de la región y parte de su área dentro."""
    pz = [c for a_ in zn["anillos"] for c in a_]
    if not pz:
        return False
    xs = [c[0] for c in pz]
    if max(xs) - min(xs) > 40:          # área imposible para una perturbación: dato malformado
        return False
    la, lo, _ = centro_zona(zn)
    J = JURISDICCION
    if not (J[0] - 8 <= lo <= J[1] + 6 and J[2] - 6 <= la <= J[3] + 4):
        return False
    return any(en_jurisdiccion(*c) for c in pz)


def zonas_baja_presion():
    """Zonas del Tropical Weather Outlook del NHC: área sombreada de posible formación, probabilidades a 2 y 7 días
    y, si el NHC la publica, la posición actual de la perturbación (la "X" del mapa del NHC) y la flecha de su
    desplazamiento esperado."""
    z = zipfile.ZipFile(io.BytesIO(http_get(URL_TWO).content))
    areas, puntos, lineas = _leer_shp(z, "areas"), _leer_shp(z, "points"), _leer_shp(z, "lines")
    zonas = []
    for rec, shp in areas:
        p2 = next((str(v) for k, v in rec.items() if "PROB2" in k), "")
        p7 = next((str(v) for k, v in rec.items() if "PROB7" in k), "")
        idx = list(shp.parts) + [len(shp.points)]
        anillos = [_normalizar_anillo(shp.points[a:b]) for a, b in zip(idx[:-1], idx[1:])]
        zn = dict(anillos=anillos, p2=p2.strip(), p7=p7.strip(), clave=_clave(rec))
        zn["num"] = zn["clave"].split("|")[1] if zn["clave"] else str(len(zonas) + 1)
        # posición actual: el punto con la misma clave; si no hay clave, el más cercano al área
        cand = [(r, s) for r, s in puntos if s.points and zn["clave"] and _clave(r) == zn["clave"]]
        if not cand and puntos:
            def lejania(s):
                x, y = s.points[0]
                if any(MPath(np.array(a_)).contains_point((x, y)) for a_ in anillos):
                    return 0.0
                return min(math.hypot(x - c[0], y - c[1]) for a_ in anillos for c in a_)
            mejor = min((ps for ps in puntos if ps[1].points), key=lambda ps: lejania(ps[1]), default=None)
            if mejor and lejania(mejor[1]) < 3.0:
                cand = [mejor]
        if cand:
            zn["centro"] = tuple(cand[0][1].points[0])
            fl = [s.points for r, s in lineas if s.points and math.hypot(s.points[0][0] - zn["centro"][0],
                                                                         s.points[0][1] - zn["centro"][1]) < 1.0]
            if fl:
                zn["flecha"] = fl[0]
        zonas.append(zn)
    return zonas


def invests_activos(horas_max=24):
    """Invests del NHC (perturbaciones 90-99 en vigilancia) con su posición y presión mínima (b-deck ATCF)."""
    lista = []
    anio = dt.datetime.utcnow().year
    try:
        indice = http_get(URL_BTK_DIR, intentos=2).text
    except Exception as e:
        log.warning("No se pudo leer la lista de invests del NHC: %s", e)
        return lista
    for id_ in sorted(set(re.findall(rf"b((?:ep|al|cp)9\d{anio})\.dat", indice, re.I))):
        try:
            lineas = [l for l in http_get(URL_BTK.format(id=id_.lower()), intentos=2).text.splitlines() if l.strip()]
            p = [c.strip() for c in lineas[-1].split(",")]
            t = dt.datetime.strptime(p[2], "%Y%m%d%H")
            if (dt.datetime.utcnow() - t).total_seconds() > horas_max * 3600:
                continue   # invest viejo (ya desactivado)
            lista.append(dict(id=id_.upper(), t=t, mb=_num(p[9]), kt=_num(p[8]),
                              lat=int(p[6][:-1]) / 10 * (1 if p[6][-1] == "N" else -1),
                              lon=int(p[7][:-1]) / 10 * (-1 if p[7][-1] == "W" else 1)))
        except Exception as e:
            log.warning("Invest %s no disponible: %s", id_, e)
    return lista


def presion_modelo(lat, lon, radio=1.5, paso=0.5):
    """Presión mínima a nivel del mar (hPa) del modelo, en la hora actual, alrededor de un punto."""
    lats = np.arange(lat - radio, lat + radio + 1e-6, paso)
    lons = np.arange(lon - radio, lon + radio + 1e-6, paso)
    pl = [(a, b) for a in lats for b in lons]
    datos = http_get(URL_OM, params=dict(
        latitude=",".join(f"{a:.2f}" for a, _ in pl), longitude=",".join(f"{b:.2f}" for _, b in pl),
        hourly="pressure_msl", forecast_days=1, timezone="UTC"), intentos=2).json()
    datos = datos if isinstance(datos, list) else [datos]
    ahora = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:00")
    valores = []
    for d in datos:
        h = d.get("hourly", {})
        if ahora in (h.get("time") or []):
            v = h["pressure_msl"][h["time"].index(ahora)]
            if v is not None:
                valores.append(v)
    return min(valores) if valores else None


def completar_presion_zonas(zonas):
    """Presión de cada zona de baja presión: la del invest del NHC si existe; si no, estimada con el modelo."""
    if not zonas:
        return
    invs = invests_activos()
    for zn in zonas:
        la, lo, _ = centro_zona(zn)
        cerca = [i for i in invs if i["mb"] and math.hypot(i["lat"] - la, i["lon"] - lo) < 3.0]
        if cerca:
            inv = min(cerca, key=lambda i: math.hypot(i["lat"] - la, i["lon"] - lo))
            zn["mb"], zn["mb_fuente"] = int(round(inv["mb"])), f"NHC, invest {inv['id'][:4]}"
            continue
        try:
            mb = presion_modelo(la, lo)
            if mb:
                zn["mb"], zn["mb_fuente"] = int(round(mb)), "estimada con modelo"
        except Exception as e:
            log.warning("No se pudo estimar la presión de la zona B%s: %s", zn.get("num", ""), e)


def quitar_zonas_ya_ciclon(zonas, posiciones, radio=3.5):
    """Quita las zonas de baja presión que ya se convirtieron en ciclón: el NHC puede seguir publicando
    el área unas horas después de iniciar los avisos. posiciones: [(lat, lon)] de los ciclones activos."""
    quedan, omitidas = [], []
    for zn in zonas:
        la, lo, _ = centro_zona(zn)
        if any(math.hypot(la - pla, lo - plo) < radio for pla, plo in posiciones):
            log.info("Zona de baja presión B%s omitida: ya es un ciclón tropical", zn.get("num", ""))
            omitidas.append(zn)
            continue
        quedan.append(zn)
    quitar_zonas_ya_ciclon.omitidas = omitidas
    return quedan


def trayectoria_invest(primer, horas=24, dist=2.5):
    """Posiciones del invest (EP90-99) que terminó en el punto 'primer' del ciclón: su etapa de baja presión."""
    try:
        indice = http_get(URL_BTK_DIR, intentos=2).text
    except Exception:
        return []
    anio = primer["t"].year
    for id_ in sorted(set(re.findall(rf"b((?:ep|al|cp)9\d{anio})\.dat", indice, re.I))):
        try:
            pts = []
            for linea in http_get(URL_BTK.format(id=id_.lower()), intentos=1).text.splitlines():
                p = [c.strip() for c in linea.split(",")]
                if len(p) < 11 or not p[6] or not p[7]:
                    continue
                t = dt.datetime.strptime(p[2], "%Y%m%d%H")
                pts.append(dict(t=t, lat=int(p[6][:-1]) / 10 * (1 if p[6][-1] == "N" else -1),
                                lon=int(p[7][:-1]) / 10 * (-1 if p[7][-1] == "W" else 1),
                                kt=int(p[8] or 0), mb=_num(p[9]), tipo="LO", invest=id_.upper()[:4]))
            pts = [q for q in pts if q["t"] < primer["t"]]
            if pts and 0 <= (primer["t"] - pts[-1]["t"]).total_seconds() <= horas * 3600 and \
                    math.hypot(pts[-1]["lat"] - primer["lat"], pts[-1]["lon"] - primer["lon"]) < dist:
                log.info("Etapa de baja presión tomada del invest %s", id_.upper())
                return pts[-8:]
        except Exception:
            continue
    return []


def centro_zona(zn):
    """(lat, lon, es_posición): posición actual de la perturbación si el NHC la publica; si no, centro del área."""
    if zn.get("centro"):
        return zn["centro"][1], zn["centro"][0], True
    pz = [c for a_ in zn["anillos"] for c in a_]
    return sum(c[1] for c in pz) / len(pz), sum(c[0] for c in pz) / len(pz), False


# ----------------------------- dibujo -----------------------------
ROJO, ROJO_OSC, GRIS = "#E24B4A", "#A32D2D", "#5F5E5A"
CAJA = dict(fc="white", ec="none", alpha=0.85, pad=1.5)


def cargar_estados():
    gj = json.loads((BASE / "mex.geojson").read_text(encoding="utf-8"))
    est = {}
    for f in gj["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        est[f["properties"]["name"]] = [p[0] for p in polys]
    return est


def centro_estado(polys):
    p = max(polys, key=len)
    lon = sum(c[0] for c in p) / len(p); lat = sum(c[1] for c in p) / len(p)
    return lon, lat


def base_mapa(ax, estados, colores, dominio, fondo="#dcecf7", nombres=True):
    lon0, lon1, lat0, lat1 = dominio
    ax.set_facecolor(fondo)
    for n, polys in estados.items():
        for p in polys:
            ax.add_patch(Polygon([merc(*c) for c in p], closed=True,
                                 fc=colores.get(n, "#e8e6df"), ec="white", lw=0.8, zorder=1))
        if nombres and n in colores and n not in ("Colima", "Aguascalientes", "Tlaxcala", "Morelos"):
            lo, la = centro_estado(polys)
            if lon0 < lo < lon1 and lat0 < la < lat1:
                ax.text(*merc(lo, la), n.upper(), fontsize=8.5, ha="center", color=GRIS,
                        style="italic", zorder=3, clip_on=True)
    x0, y0 = merc(lon0, lat0); x1, y1 = merc(lon1, lat1)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    xt = range(math.ceil(lon0 / 2) * 2, int(lon1) + 1, 2)
    ax.set_xticks(xt); ax.set_xticklabels([f"{-v}°O" for v in xt], fontsize=8)
    yt = range(math.ceil(lat0 / 2) * 2, int(lat1) + 1, 2)
    ax.set_yticks([merc(0, v)[1] for v in yt]); ax.set_yticklabels([f"{v}°N" for v in yt], fontsize=8)
    ax.grid(color="white", lw=0.5, alpha=0.7)


def fecha_corta(p):
    """Fecha del punto pronosticado a partir de su TAU (hora local)."""
    if p.get("t"):
        t = p["t"] - dt.timedelta(hours=6)
        return f"{t.day} {MESES[t.month - 1]}"
    return p.get("fecha") or f"+{p['tau']} h"


def marcar_ref(ax, nombre, ya):
    if nombre in ya:
        return
    ya.add(nombre)
    lat, lon = REFS[nombre]
    x, y = merc(lon, lat)
    ax.plot(x, y, "o", ms=4, color="#2c2c2a", zorder=6)
    ax.text(x + 0.12, y + 0.15, nombre, fontsize=8.5, color="#2c2c2a", zorder=6, clip_on=True)


def dibujar_pronostico(ax, actual, fc, etiquetas_largas, ya_refs):
    """Línea discontinua y un punto por día (24, 48, 72, 96, 120 h)."""
    if not fc:
        return
    cadena = ([actual] if actual else []) + [p for p in fc if p["tau"] not in (0, None)]
    xy = [merc(p["lon"], p["lat"]) for p in cadena]
    ax.plot(*zip(*xy), color=ROJO, lw=2.4, ls=(0, (5, 4)), zorder=4)
    for p in fc:
        if p["tau"] not in (24, 48, 72, 96, 120):
            continue
        x, y = merc(p["lon"], p["lat"])
        ax.plot(x, y, "o", ms=8, mfc="white", mec=ROJO, mew=2, zorder=5)
        txt = f"{fecha_corta(p)} · {categoria(p['kt'])}"
        if etiquetas_largas:
            nombre, (rla, rlo), d, r = refs_cercanas(p["lat"], p["lon"], 1)[0]
            txt += f"\n{fmt_dist(d)} al {r} de {nombre}"
            ax.plot([x, merc(rlo, rla)[0]], [y, merc(rlo, rla)[1]], ls=":", color=GRIS, lw=1.1, zorder=3)
            marcar_ref(ax, nombre, ya_refs)
            x0, x1 = ax.get_xlim()
            izq = (x - x0) / (x1 - x0) > 0.30   # sin espacio a la izquierda → etiqueta a la derecha
            ax.annotate(txt, (x, y), xytext=(-12, 0) if izq else (12, 0), textcoords="offset points",
                        fontsize=9, ha="right" if izq else "left", va="center", bbox=CAJA, zorder=6)
        else:
            x0, x1 = ax.get_xlim()
            izq = (x - x0) / (x1 - x0) > 0.22   # cerca del borde izquierdo → etiqueta a la derecha
            ax.annotate(txt, (x, y), xytext=(-10, 0) if izq else (10, 0), textcoords="offset points",
                        fontsize=9, ha="right" if izq else "left", va="center", color=ROJO_OSC,
                        zorder=6, annotation_clip=True)


def dibujar_actual(ax, p, con_distancias, ya_refs):
    x, y = merc(p["lon"], p["lat"])
    if con_distancias:
        for nombre, (rla, rlo), d, r in refs_cercanas(p["lat"], p["lon"], N_DISTANCIAS):
            x2, y2 = merc(rlo, rla)
            ax.plot([x, x2], [y, y2], color=ROJO_OSC, lw=1.2, alpha=0.85, zorder=3)
            xm, ym = x + (x2 - x) * 0.6, y + (y2 - y) * 0.6
            ang = math.degrees(math.atan2(y2 - y, x2 - x))
            ang = ang - 180 if ang > 90 else ang + 180 if ang < -90 else ang
            ax.text(xm, ym, fmt_dist(d), fontsize=8, color=ROJO_OSC, rotation=ang,
                    ha="center", va="center", rotation_mode="anchor", zorder=6,
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))
            marcar_ref(ax, nombre, ya_refs)
    ax.plot(x, y, "o", ms=12, color=ROJO, zorder=5)
    ax.plot(x, y, "o", ms=22, mfc="none", mec=ROJO_OSC, mew=1.5, zorder=5)


def dibujar_zonas(ax, zonas, lat_piso=None):
    for zn in zonas:
        for anillo in zn["anillos"]:
            ax.add_patch(Polygon([merc(*c) for c in anillo], closed=True, fc="#FAC775", alpha=0.35,
                                 ec="#BA7517", lw=1.5, ls="--", hatch="///", zorder=2))
        if zn.get("flecha") and len(zn["flecha"]) >= 2:   # desplazamiento esperado (flecha del NHC)
            xy = [merc(*c) for c in zn["flecha"]]
            ax.plot([p[0] for p in xy], [p[1] for p in xy], color="#854F0B", lw=1.8, zorder=4)
            ax.annotate("", xy=xy[-1], xytext=xy[-2], arrowprops=dict(arrowstyle="-|>", color="#854F0B", lw=1.8,
                                                                     mutation_scale=16), zorder=4)
        pts = [c for a in zn["anillos"] for c in a]
        lo = sum(c[0] for c in pts) / len(pts); la = min(c[1] for c in pts)
        txt = f"Zona de baja presión B{zn.get('num', '')}"
        if zn["p2"] or zn["p7"]:
            txt += f"\n2 días: {zn['p2'] or '–'} · 7 días: {zn['p7'] or '–'}"
        abajo = lat_piso is not None and la < lat_piso
        if abajo:  # la zona sale del mapa: etiqueta en el borde inferior, dentro del recuadro
            la = lat_piso
        ax.annotate(txt, merc(lo, la), xytext=(0, 4 if abajo else -6), textcoords="offset points",
                    fontsize=8.5, ha="center", va="bottom" if abajo else "top", color="#633806", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BA7517", lw=0.8, alpha=0.9))


def caja_info(ax, p, adv):
    nombre_cat = ("Huracán " + categoria(p["kt"])) if p["kt"] >= 64 else \
                 ("Tormenta tropical" if p["kt"] >= 34 else "Depresión tropical")
    lineas = [f"POSICIÓN ACTUAL ({fecha_local(p['t'])})", nombre_cat,
              f"{p['lat']:.1f}°N  {-p['lon']:.1f}°O",
              f"Vientos: {p['kt'] * 1.852:.0f} km/h"]
    if p.get("mb"):
        lineas.append(f"Presión: {p['mb']} hPa")
    lineas.append("")
    lineas += [f"{fmt_dist(d)} al {r} de {k}" for k, _, d, r in refs_cercanas(p["lat"], p["lon"], 3)]
    if adv:
        lineas.append(f"\nAviso NHC n.º {adv}")
    ax.text(0.985, 0.985, "\n".join(lineas), transform=ax.transAxes, fontsize=9, va="top", ha="right",
            multialignment="left",
            zorder=9, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=ROJO_OSC, lw=1))


def dominio_auto(puntos, zonas):
    lons = [p["lon"] for p in puntos] + [c[0] for z in zonas for a in z["anillos"] for c in a]
    lats = [p["lat"] for p in puntos] + [c[1] for z in zonas for a in z["anillos"] for c in a]
    if not lons:
        lons, lats = [-100], [15]
    lon0 = max(min(min(lons) - 3, -106), -118)
    lon1 = min(max(max(lons) + 2, -89.8), -86)  # incluye todo Chiapas
    lat0 = max(min(min(lats) - 2.5, 12), 5)
    lat1 = min(max(max(lats) + 2, 21.5), 32)
    return lon0, lon1, lat0, lat1


def generar(corrida_modelo=""):
    ahora = dt.datetime.now()
    estados = cargar_estados()
    storm = tormenta_activa()
    activa = storm is not None
    log.info("Ciclón activo%s: %s", f" ({STORM_NAME})" if STORM_NAME else "", activa)

    try:
        bt = best_track()
    except Exception as e:
        log.error("Sin best track: %s", e); bt = []
    fc, cono, adv = [], [], ""
    if activa:
        try:
            fc, cono, adv = pronostico()
        except Exception as e:
            log.error("Sin pronóstico NHC: %s", e)
    try:
        zonas = zonas_baja_presion() if MOSTRAR_ZONAS_BAJA else []
        log.info("Zonas de baja presión: %d", len(zonas))
    except Exception as e:
        log.warning("Sin zonas de baja presión: %s", e); zonas = []
    if not bt and not fc:
        if GENERAR_TRAYECTORIA:
            log.error("No hay datos de trayectoria; no se actualizan los mapas.")
            return 1
        log.info("Sin ciclón activo%s; los mapas de lluvia y viento se generan sin trayectoria.",
                 f" ({STORM_NAME})" if STORM_NAME else "")

    actual = bt[-1] if bt else None
    pos_ciclones = ([(actual["lat"], actual["lon"])] if actual else []) + [(t_["lat"], t_["lon"]) for t_ in OTRAS_TORMENTAS]
    zonas = quitar_zonas_ya_ciclon(zonas, pos_ciclones)
    baja_origen = None   # la zona de baja presión de la que se desarrolló el ciclón principal
    if actual:
        cerca = [z for z in getattr(quitar_zonas_ya_ciclon, "omitidas", [])
                 if math.hypot(centro_zona(z)[0] - actual["lat"], centro_zona(z)[1] - actual["lon"]) < 3.5]
        if cerca:
            la_, lo_, _ = centro_zona(cerca[0])
            baja_origen = dict(num=cerca[0].get("num", ""), lat=la_, lon=lo_)
        if bt and not any(p_.get("tipo") in ("LO", "DB", "WV") for p_ in bt):
            bt = trayectoria_invest(bt[0]) + bt   # etapa de baja presión (invest) antes de ser ciclón
    if actual and fc:  # fecha de cada punto pronosticado a partir de TAU
        for p in fc:
            if p["tau"] is not None:
                p["t"] = actual["t"] + dt.timedelta(hours=p["tau"])
    puntos = ([actual] if actual else []) + fc
    dom = dominio_auto(puntos, zonas)
    # solo zonas que caen dentro del mapa
    zonas_todas = list(zonas)
    completar_presion_zonas([z for z in zonas_todas
                             if zona_en_jurisdiccion(z)])
    zonas = [z for z in zonas if any(dom[0] < c[0] < dom[1] and dom[2] < c[1] < dom[3]
                                     for a in z["anillos"] for c in a)]

    interes = {n: estados[n] for n in ESTADOS_INTERES if n in estados}
    pts_lon = [c[0] for ps in interes.values() for p in ps for c in p]
    pts_lat = [c[1] for ps in interes.values() for p in ps for c in p]
    dom2 = (round(min(pts_lon) - 1.2, 1), round(max(pts_lon) + 1.0, 1),
            round(min(pts_lat) - 1.6, 1), round(max(pts_lat) + 0.8, 1))
    respaldo = OUT_DIR / "ultima_lluvia.npz"
    lluvia_vieja = ""
    W = G = P = None   # P: probabilidad de lluvia (%) en la malla, si está disponible
    try:
        cache = np.load(respaldo) if respaldo.exists() else None
        if (cache is not None and corrida_modelo and "corrida" in cache.files and "P" in cache.files
                and str(cache["corrida"]) == corrida_modelo):
            LON, LAT, Z, W, G, P = cache["LON"], cache["LAT"], cache["Z"], cache["W"], cache["G"], cache["P"]
            log.info("El modelo no tiene corrida nueva (%s): se reutilizan lluvia y viento guardados", corrida_modelo)
        else:
            LON, LAT, Z, W, G, P = lluvia_modelo(dom2[0], dom2[1], dom2[2], dom2[3])
            np.savez(respaldo, LON=LON, LAT=LAT, Z=Z, W=W, G=G, P=P, fecha=ahora.strftime("%d/%m/%Y %H:%M"),
                     corrida=corrida_modelo)
    except Exception as e:
        log.error("Sin datos del modelo: %s", e)
        if respaldo.exists():
            d = np.load(respaldo)
            LON, LAT, Z = d["LON"], d["LAT"], d["Z"]
            if "G" in d.files:
                W, G = d["W"], d["G"]
            if "P" in d.files:
                P = d["P"]
            lluvia_vieja = str(d["fecha"])
            log.warning("Se usan los datos del modelo descargados el %s", lluvia_vieja)
        else:
            LON = None
    if P is not None and np.all(np.isnan(P)):
        if PROB_MODEL:
            log.warning("El modelo %s no entregó probabilidad de lluvia; el mapa sale sin porcentajes", PROB_MODEL)
        P = None
    maxs = maximos_por_estado(estados, LON, LAT, Z) if LON is not None else {}
    nombre = ARCHIVO_BASE
    tipo = "Huracán" if actual and actual["kt"] >= 64 else "Tormenta tropical" if actual and actual["kt"] >= 34 else "Ciclón"
    hora = f"{fecha_local(actual['t'])}" if actual else ahora.strftime("%d/%m %H:%M")
    pie_hora = f"Generado {ahora:%d/%m/%Y %H:%M} (hora local)."
    if not activa:
        pie_hora += " El NHC ya no emite avisos: se muestra la última posición observada."

    if GENERAR_TRAYECTORIA:
        # ================= Mapa 1: posición actual, pronóstico y distancias =================
        col = {}
        for n, (v, _, _) in maxs.items():
            if v >= 150: col[n] = "#F0997B"
            elif v >= 75: col[n] = "#FAC775"
        fig, ax = plt.subplots(figsize=(10, 9), dpi=150)
        base_mapa(ax, estados, col, dom)
        ya = set()
        if MOSTRAR_CONO:
            for anillo in cono:
                ax.add_patch(Polygon([merc(*c) for c in anillo], closed=True, fc="#888780",
                                     alpha=0.15, ec="none", zorder=2))
        dibujar_zonas(ax, zonas)
        dibujar_pronostico(ax, actual, fc, True, ya)
        if actual:
            dibujar_actual(ax, actual, True, ya)
            caja_info(ax, actual, adv)
        leyenda = [Line2D([], [], marker="o", ls="none", color=ROJO, ms=10, label="Posición actual"),
                   Line2D([], [], color=ROJO, lw=2.4, ls="--", label="Pronóstico NHC"),
                   Line2D([], [], color=ROJO_OSC, lw=1.2, label="Distancias desde la posición actual")]
        if zonas:
            leyenda.append(Patch(fc="#FAC775", alpha=0.5, ec="#BA7517", hatch="///", label="Zona de baja presión (NHC)"))
        if col:
            leyenda += [Patch(fc="#F0997B", label="Lluvia ≥150 mm (modelo)"),
                        Patch(fc="#FAC775", label="Lluvia 75–150 mm (modelo)")]
        ax.legend(handles=leyenda, loc="lower left", fontsize=9, framealpha=0.95)
        ax.set_title(f"{tipo} {STORM_NAME} — posición actual y distancias a la costa\n"
                     f"Posición del {hora} (NHC)", fontsize=13)
        fig.text(0.5, ax.get_position().y0 - 0.06,
                 "Distancias (≈) calculadas con la posición del NHC. Fuente: NHC/NOAA y SMN/Conagua.\n"
                 f"{pie_hora} Consulte el aviso oficial más reciente.",
                 ha="center", va="top", fontsize=8, color=GRIS)
        guardar(fig, f"{nombre}_trayectoria.png", ahora)

    # ================= Mapa 2: lluvia acumulada en los estados de interés =================
    if LON is not None:
        # sistemas oceánicos cercanos (otros ciclones y zonas de baja presión)
        sist_pts = [(t["lon"], t["lat"]) for t in OTRAS_TORMENTAS]
        # solo sistemas que afectan la jurisdicción (Guerrero a Chiapas): posición actual o pronosticada dentro del área
        zonas2 = [z for z in zonas_todas if zona_en_jurisdiccion(z)]
        otras2 = [t for t in OTRAS_TORMENTAS if en_jurisdiccion(t["lon"], t["lat"])]
        actual_m, fc_m = actual, fc
        if actual and not any(en_jurisdiccion(p_["lon"], p_["lat"]) for p_ in [actual] + fc):
            log.info("%s está fuera de la jurisdicción: no se muestra en los mapas", STORM_NAME)
            actual_m, fc_m = None, []
        dom2 = dominio_jurisdiccion(dom2, actual_m, fc_m, otras2, zonas2)
        if ISTMO_EN_GOLFO and ISTMO_DESTACAR:
            # se abre espacio al norte para que la ampliación quepa sobre el Golfo de México
            g0, g1, gla = GOLFO_CAJA
            alto_lat = (g1 - g0) * 0.93 + 0.4
            lo0_, lo1_, la0_, la1_ = (min(dom2[0], g0 - 0.6), max(dom2[1], g1 + 0.6), dom2[2],
                                      max(dom2[3], gla + alto_lat + 0.3))
            falta = (la1_ - la0_) / 0.8 - (lo1_ - lo0_)   # ensanchar para no deformar ni achicar los estados
            if falta > 0:
                lo0_ = max(lo0_ - falta / 2, EXTENSION_MAX[0]); lo1_ = min(lo1_ + falta / 2, EXTENSION_MAX[1])
            dom2 = (lo0_, lo1_, la0_, la1_)

        mapas = []
        if GENERAR_LLUVIA:
            mapas.append(dict(archivo="lluvia", Z=Z, Z2=None, P=P, unidad="mm",
                              niveles=[10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 600],
                              colores=["#E6F1FB", "#B5D4F4", "#85B7EB", "#378ADD", "#185FA5",
                                       "#0C447C", "#7F77DD", "#534AB7", "#D4537E", "#993556"],
                              barra=f"Lluvia acumulada a {QPF_DIAS} días (mm)",
                              panel="MAYOR ACUMULADO (probabilidad)" if P is not None else "MAYOR ACUMULADO",
                              titulo="Lluvia acumulada pronosticada"))
        if GENERAR_VIENTO and G is not None:
            mapas.append(dict(archivo="viento", Z=G, Z2=W, unidad="km/h",
                              niveles=[20, 30, 40, 50, 63, 75, 90, 118, 150, 180, 250],
                              colores=["#EAF3DE", "#C0DD97", "#97C459", "#F2E48C", "#FAC775",
                                       "#EF9F27", "#D85A30", "#A32D2D", "#72243E", "#4A1B5C"],
                              barra=f"Rachas máximas de viento en {QPF_DIAS} días  ·  34 kt (63 km/h) = tormenta tropical, 64 kt (118 km/h) = huracán",
                              panel="RACHAS MÁXIMAS (viento sostenido)",
                              titulo="Rachas máximas de viento pronosticadas"))
        elif GENERAR_VIENTO:
            log.warning("No hay datos de viento guardados; se omite el mapa de viento hasta la próxima descarga")
        for cfg in mapas:
            fig, ax = plt.subplots(figsize=(13, 7.8), dpi=150)
            fig.subplots_adjust(left=0.05, right=0.76, top=0.9, bottom=0.12)
            base_mapa(ax, estados, {}, dom2, fondo="#f4f8fb", nombres=False)
            # campo de lluvia solo dentro de los estados de interés
            Z, niveles, cmap = cfg["Z"], cfg["niveles"], ListedColormap(cfg["colores"])
            planos = np.c_[LON.ravel(), LAT.ravel()]
            dentro = np.zeros(len(planos), bool)
            for ps in interes.values():
                for p in ps:
                    dentro |= MPath(p).contains_points(planos)
            XM = np.vectorize(lambda lo, la: merc(lo, la)[0])(LON, LAT)
            YM = np.vectorize(lambda lo, la: merc(lo, la)[1])(LON, LAT)
            cf = ax.contourf(XM, YM, np.ma.masked_less(Z, niveles[0]), levels=niveles, cmap=cmap,
                             norm=BoundaryNorm(niveles, cmap.N), zorder=2)
            # recortar el campo a la silueta de los estados
            verts, codes = [], []
            for ps in interes.values():
                for p in ps:
                    v = [merc(*c) for c in p]
                    verts += v; codes += [MPath.MOVETO] + [MPath.LINETO] * (len(v) - 2) + [MPath.CLOSEPOLY]
            clip = matplotlib.patches.PathPatch(MPath(verts, codes), transform=ax.transData, fc="none", ec="none")
            ax.add_patch(clip)
            cf.set_clip_path(clip)
            for n, ps in interes.items():
                for p in ps:
                    ax.add_patch(Polygon([merc(*c) for c in p], closed=True, fc="none", ec="#2c2c2a",
                                         lw=1.0, zorder=3))
                lo, _ = centro_estado(ps)
                la = max(c[1] for p in ps for c in p)
                t = ax.text(*merc(lo, la + 0.15), n.upper(), fontsize=10, ha="center", va="bottom",
                            color="#2c2c2a", fontweight="bold", zorder=4)
                t.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
            ax.apply_aspect()
            pos = ax.get_position()
            cax = fig.add_axes([pos.x0, pos.y0 - (0.105 if cfg["unidad"] == "km/h" else 0.075), pos.width, 0.022])
            cb = fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=niveles)
            cb.set_label(cfg["barra"], fontsize=9)
            cb.ax.tick_params(labelsize=8)
            if cfg["unidad"] == "km/h":   # equivalencia en nudos sobre la barra
                for ii, vv in enumerate(niveles):
                    cb.ax.text(ii / (len(niveles) - 1), 1.25, f"{vv / 1.852:.0f}", transform=cb.ax.transAxes,
                               ha="center", va="bottom", fontsize=8, color="#444441")
                cb.ax.text(1.025, 1.25, "kt", transform=cb.ax.transAxes, ha="left", va="bottom",
                           fontsize=8, color="#444441", fontweight="bold")
                cb.ax.text(1.025, -0.6, "km/h", transform=cb.ax.transAxes, ha="left", va="top",
                           fontsize=8, color="#444441", fontweight="bold")

            # puntos de mayor acumulado por estado, separados entre sí
            z = Z.ravel()
            filas, k = [], 0
            elegidos_todos = []   # (estado, lat, lon, núm) para no taparlos con la ampliación
            etiquetas_mapa = []   # etiquetas de valores que se reacomodan para no encimarse ("guias")
            for n, ps in interes.items():
                dentro_n = np.zeros(len(planos), bool)
                for p in ps:
                    dentro_n |= MPath(p).contains_points(planos)
                orden = [i for i in np.argsort(-np.nan_to_num(z, nan=-1)) if dentro_n[i] and z[i] >= niveles[0]]
                elegidos = []
                for i in orden:
                    lo, la = planos[i]
                    if all(math.hypot(lo - a, la - b) >= SEPARACION_PUNTOS for a, b in elegidos):
                        elegidos.append((lo, la))
                        k += 1
                        elegidos_todos.append((n, la, lo, k))
                        x, y = merc(lo, la)
                        P_ = cfg.get("P")
                        modo = ETIQUETAS_MAPA if dom2[1] - dom2[0] <= ANCHO_MAX_ETIQUETAS else "numero"
                        if modo == "insignia":   # una sola etiqueta compacta: número y valor principal
                            ax.plot(x, y, "o", ms=4.5, color="#2c2c2a", mec="white", mew=0.8, zorder=6)
                            corto = f"{z[i]:.0f} {cfg['unidad']}"
                            etiquetas_mapa.append(ax.annotate(
                                f"{k}  {corto}", (x, y), xytext=(0, 9), textcoords="offset points",
                                fontsize=7.6, fontweight="bold", va="bottom", ha="center", color="white",
                                zorder=7, bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.5",
                                                    fc="#2c2c2a", ec="white", lw=0.6, alpha=0.92),
                                arrowprops=dict(arrowstyle="-", color="#2c2c2a", lw=0.7, shrinkA=0, shrinkB=2)))
                        else:
                            ax.plot(x, y, "^", ms=8, color="#2c2c2a", mec="white", mew=0.8, zorder=6)
                            if modo != "guias":   # en "guias" el número va dentro de la etiqueta
                                ax.annotate(f"{k}", (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8.5,
                                            fontweight="bold", zorder=7,
                                            bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="#2c2c2a", lw=0.6))
                        etq = fmt_valor(z[i], cfg["unidad"], "\n")
                        if P_ is not None and not math.isnan(P_.ravel()[i]):
                            etq += f"\n{P_.ravel()[i]:.0f}%"
                        if modo == "completa":
                            ax.annotate(etq, (x, y), xytext=(0, -9), textcoords="offset points",
                                        fontsize=8, ha="center", va="top", color="#2c2c2a", fontweight="bold",
                                        zorder=7, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6))
                        elif modo == "guias":
                            etiquetas_mapa.append(ax.annotate(
                                f"({k}) " + etq, (x, y), xytext=(0, 10), textcoords="offset points", fontsize=7.6,
                                ha="center", va="bottom", color="#2c2c2a", fontweight="bold", zorder=7,
                                bbox=dict(fc="white", ec="#9aa4ad", lw=0.5, alpha=0.92, pad=0.8,
                                          boxstyle="round,pad=0.25"),
                                arrowprops=dict(arrowstyle="-", color="#5f5e5a", lw=0.6, shrinkA=0, shrinkB=3)))
                        extra = fmt_extra(cfg, cfg["Z2"].ravel()[i] if cfg.get("Z2") is not None else None,
                                          P_.ravel()[i] if P_ is not None else None)
                        filas.append(f"{k}. {fmt_valor(z[i], cfg['unidad'])}{extra} · {localidad(la, lo, ps)}, {ABREV.get(n, n)}")
                    if len(elegidos) >= N_PUNTOS_POR_ESTADO:
                        break
            if etiquetas_mapa:
                prohib = []
                if ISTMO_EN_GOLFO and ISTMO_DESTACAR:   # el espacio de la ampliación sobre el Golfo
                    g0, g1, gla = GOLFO_CAJA
                    p0 = ax.transData.transform(merc(g0, gla))
                    p1 = ax.transData.transform(merc(g1, dom2[3]))
                    prohib.append((p0[0] - 4, p0[1] - 4, p1[0] + 4, p1[1]))
                reacomodar_etiquetas(ax, etiquetas_mapa, prohibidas=prohib)
            # Istmo de Tehuantepec: valores en las localidades (para el panel y la ampliación)
            filas_istmo, etiq_istmo = [], []
            if ISTMO_DESTACAR:
                for nom, la, lo, lado in ISTMO_LOCALIDADES:
                    v = valor_en(LON, LAT, Z, la, lo)
                    etiq_istmo.append((nom, la, lo, lado, v, None))
                    if math.isnan(v):
                        continue
                    v2 = valor_en(LON, LAT, cfg["Z2"], la, lo) if cfg.get("Z2") is not None else None
                    pr = valor_en(LON, LAT, cfg["P"], la, lo) if cfg.get("P") is not None else None
                    etiq_istmo[-1] = (nom, la, lo, lado, v, pr)
                    filas_istmo.append(f"{nom}: {fmt_valor(v, cfg['unidad'])}{fmt_extra(cfg, v2, pr)}")
            if actual_m:
                dibujar_trayectoria(ax, [b_ for b_ in bt[:-1]][-12:] if bt else [], actual_m, fc_m, baja_origen)
            if actual_m:
                x, y = merc(actual_m["lon"], actual_m["lat"])
                dibujar_simbolo_ciclon(ax, x, y, actual_m["kt"],
                                       etiqueta_ciclon(actual_m["kt"], STORM_NAME, numero_ciclon(STORM_ID))[1], ROJO_OSC)
            # sistemas hidrometeorológicos oceánicos
            for t in otras2:
                x, y = merc(t["lon"], t["lat"])
                dibujar_simbolo_ciclon(ax, x, y, t["kt"], etiqueta_ciclon(t["kt"], t["nombre"], t.get("num", ""))[1],
                                       "#712B13")
            dibujar_zonas(ax, zonas2, dom2[2] + 0.25)
            # distancia de cada sistema a los 2 puntos de la costa más cercanos (líneas del color del sistema)
            C_TRUDY, C_ZONA, C_OTRO, C_TXT = "#A32D2D", "#854F0B", "#993C1D", "#2c2c2a"
            objetivos = []
            if actual_m:
                objetivos.append((f"● {etiqueta_ciclon(actual_m['kt'], STORM_NAME, numero_ciclon(STORM_ID))[1]}: "
                                  f"{nombre_categoria(actual_m['kt'])}, {actual_m['kt'] * 1.852:.0f} km/h · {actual_m['kt']:.0f} kt",
                                  actual_m["lat"], actual_m["lon"], C_TRUDY, False))
            objetivos += [(f"● {etiqueta_ciclon(t['kt'], t['nombre'], t.get('num', ''))[1]}: {nombre_categoria(t['kt'])}"
                           + (f", {t['kt'] * 1.852:.0f} km/h · {t['kt']:.0f} kt" if t.get("kt") else ""),
                           t["lat"], t["lon"], C_OTRO, False) for t in otras2]
            anillos_zona = {}
            for zn in zonas2:
                pz = [c for a_ in zn["anillos"] for c in a_]
                la_c, lo_c, _ = centro_zona(zn)
                anillos_zona[(la_c, lo_c)] = zn["anillos"]
                objetivos.append((f"▨ Zona de baja presión B{zn.get('num', '')}", la_c, lo_c, C_ZONA, True))
            ya_ref = set()
            lineas_panel = []   # (texto, color, negrita)
            if filas_istmo:
                lineas_panel.append(("ISTMO DE TEHUANTEPEC", "#0C447C", True))
                lineas_panel += [(f, "#0C447C", False) for f in filas_istmo]
                lineas_panel.append(("", C_TXT, False))
            if filas:
                lineas_panel.append((cfg["panel"], C_TXT, True))
                lineas_panel += [(f, C_TXT, False) for f in filas]
                lineas_panel.append(("", C_TXT, False))
            lineas_panel.append(("SISTEMAS OCEÁNICOS Y DISTANCIA A LA COSTA", C_TXT, True))
            try:   # resumen para la portada "Boletín del día" de Pronóstico EIOS
                tipo_resumen = "viento" if cfg["unidad"] == "km/h" else "lluvia"
                (OUT_DIR / f"resumen_{tipo_resumen}.json").write_text(json.dumps({
                    "generado": ahora.isoformat(timespec="minutes"), "titulo": cfg["panel"],
                    "filas": filas, "istmo": filas_istmo,
                    "sistemas": [o[0].lstrip("●▨ ").strip() for o in objetivos]}, ensure_ascii=False),
                    encoding="utf-8")
            except Exception as e_res:
                log.warning("No se pudo guardar el resumen para la portada: %s", e_res)
            if not objetivos:
                lineas_panel.append(("Sin sistemas en la jurisdicción", C_TXT, False))
            for titulo, la, lo, color, es_zona in objetivos:
                lineas_panel.append((titulo, color, True))
                if es_zona:
                    zn_ = zonas2[[o for o in objetivos if o[4]].index((titulo, la, lo, color, es_zona))]
                    lineas_panel.append((f"   Desarrollo: {zn_['p2'] or '–'} en 2 días · {zn_['p7'] or '–'} en 7 días",
                                         color, False))
                    es_pos = centro_zona(zn_)[2]
                    lineas_panel.append((f"   {'Perturbación (posición actual)' if es_pos else 'Centro del área'} "
                                         f"≈{la:.1f}°N {-lo:.1f}°O", color, False))
                    if zn_.get("mb"):
                        lineas_panel.append((f"   Presión mínima: {zn_['mb']} mb ({zn_['mb_fuente']})", color, False))
                x1, y1 = merc(lo, la)
                xl_, yl_ = ax.get_xlim(), ax.get_ylim()
                fuera = not es_zona and not (xl_[0] <= x1 <= xl_[1] and yl_[0] <= y1 <= yl_[1])
                if es_zona:
                    dibujar_baja(ax, x1, y1, zn_["num"] if zn_.get("num") else "", zn_.get("mb"))
                if fuera:   # ciclón fuera del mapa: flecha en el borde apuntando hacia él
                    fx = (x1 - xl_[0]) / (xl_[1] - xl_[0]); fy = (y1 - yl_[0]) / (yl_[1] - yl_[0])
                    cx, cy = 0.5, 0.5
                    s_ = min(abs(0.47 / (fx - cx)) if fx != cx else 9, abs(0.45 / (fy - cy)) if fy != cy else 9)
                    bx, by = cx + (fx - cx) * s_, cy + (fy - cy) * s_
                    ix, iy = cx + (fx - cx) * s_ * 0.92, cy + (fy - cy) * s_ * 0.92
                    nom_s = titulo.split(":")[0].replace("● ", "")
                    ax.annotate("", xy=(bx, by), xytext=(ix, iy), xycoords="axes fraction",
                                arrowprops=dict(arrowstyle="-|>", color=color, lw=2), zorder=9)
                    ta = ax.text(ix, iy, f"{nom_s}\n(fuera del mapa)", transform=ax.transAxes, fontsize=8.5,
                                 color=color, fontweight="bold", ha="left" if fx < 0.5 else "right",
                                 va="top" if fy > 0.5 else "bottom", zorder=9,
                                 bbox=dict(fc="white", ec=color, lw=0.8, alpha=0.9, pad=2))
                if es_zona:
                    lineas_panel.append((f"   Desde {'la perturbación' if es_pos else 'el centro del área'}:", color, False))
                for j, (k, (rla, rlo), d, r) in enumerate(refs_para(la, lo)):
                    lineas_panel.append((f"   {fmt_dist(d)} al {r} de {k}", C_REF_PRINCIPAL if k == REF_PRINCIPAL else color,
                                         k == REF_PRINCIPAL))
                    if fuera:
                        continue
                    x2, y2 = merc(rlo, rla)
                    es_sc = k == REF_PRINCIPAL
                    col_l = C_REF_PRINCIPAL if es_sc else color
                    ax.plot([x1, x2], [y1, y2], ls=(0, (4, 2)) if es_sc else (0, (1, 2)),
                            color=col_l, lw=1.6 if es_sc else 1.2, zorder=5)
                    y_vis = merc(0, dom2[2] + 0.5)[1]
                    t0 = (y_vis - y1) / (y2 - y1) if y1 < y_vis < y2 else 0.0   # tramo fuera del mapa
                    f = t0 + (1 - t0) * (0.42, 0.62, 0.80)[min(j, 2)]   # escalonadas, sin tapar la marca del sistema
                    ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
                    ang = ang - 180 if ang > 90 else ang + 180 if ang < -90 else ang
                    ax.text(x1 + (x2 - x1) * f, y1 + (y2 - y1) * f, f"{d:,.0f} Km · {d / 1.852:,.0f} NM", fontsize=7.5,
                            rotation=ang, rotation_mode="anchor", ha="center", va="center", color=col_l,
                            fontweight="bold", zorder=7, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))
                    ya_ref.add(k)
                if actual_m and not es_zona and (la, lo) == (actual_m["lat"], actual_m["lon"]) and fc_m:
                    # resumen del pronóstico respecto a Salina Cruz
                    rla, rlo = REFS[REF_PRINCIPAL]
                    serie = [actual_m] + [p_ for p_ in fc_m if p_.get("kt") is not None]
                    dists = [(dist_rumbo((rla, rlo), (p_["lat"], p_["lon"]))[0], p_) for p_ in serie]
                    dmin, pmin = min(dists, key=lambda x: x[0])
                    pmax = max(serie, key=lambda p_: p_["kt"])
                    previa = [p_ for p_ in (bt or []) if p_.get("tipo") in ("LO", "DB", "WV")]
                    if previa or baja_origen:
                        nb = (baja_origen or {}).get("num", "")
                        desde = f" desde el {fecha_hora_local(previa[0])}" if previa else ""
                        lineas_panel.append((f"   Se desarrolló de la zona de baja presión B{nb}{desde}", color, False))
                    lineas_panel.append((f"   Máximo pronosticado: {codigo_ciclon(pmax['kt'])} {pmax['kt']:.0f} kt"
                                         + (f" ({fecha_hora_local(pmax)})" if pmax is not actual_m else ""), color, False))
                    lim_km = REF_PRINCIPAL_MAX_NM * 1.852
                    d_act = dists[0][0]
                    if dmin > lim_km:
                        pass   # nunca se acerca a 700 NM de Salina Cruz: no se menciona
                    elif d_act > lim_km:
                        p_ent = next(p_ for d_, p_ in dists if d_ <= lim_km)
                        lineas_panel.append((f"   Entraría a {REF_PRINCIPAL_MAX_NM} NM de {REF_PRINCIPAL} el {fecha_hora_local(p_ent)}",
                                             C_REF_PRINCIPAL, True))
                        lineas_panel.append((f"   Mayor cercanía: {fmt_dist(dmin)} el {fecha_hora_local(pmin)}", C_REF_PRINCIPAL, True))
                    elif pmin is actual_m:
                        lineas_panel.append((f"   Se aleja de {REF_PRINCIPAL}: la mayor cercanía es la actual", C_REF_PRINCIPAL, False))
                    else:
                        lineas_panel.append((f"   Mayor cercanía a {REF_PRINCIPAL}: {fmt_dist(dmin)}", C_REF_PRINCIPAL, True))
                        lineas_panel.append((f"   el {fecha_hora_local(pmin)}", C_REF_PRINCIPAL, True))
                if es_zona and (la, lo) in anillos_zona:   # qué tan cerca de la costa llega el área
                    lugar, dd, rr = distancia_area(anillos_zona[(la, lo)])
                    if dd == 0:
                        lineas_panel.append((f"   El área ya incluye {lugar}", color, True))
                    else:
                        lineas_panel.append(("   Desde el borde del área:", color, False))
                        lineas_panel.append((f"   {fmt_dist(dd)} al {rr} de {lugar}", color, True))
            for k in ya_ref:
                rla, rlo = REFS[k]
                x2, y2 = merc(rlo, rla)
                if k == REF_PRINCIPAL:   # Salina Cruz: estrella, como en el mapa de trayectoria
                    ax.plot(x2, y2, marker="*", ms=15, color=C_REF_PRINCIPAL, mec="white", mew=1.1, zorder=10)
                    ts_ = ax.annotate(k, (x2, y2), xytext=(-6, -8), textcoords="offset points", fontsize=9.5, va="top",
                                      fontweight="bold", color="#8a1240", ha="right", zorder=10)
                    ts_.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
                    continue
                ax.plot(x2, y2, "o", ms=4.5, color=C_TXT, mec="white", mew=0.6, zorder=7)
                vecinos = [REFS[o][1] for o in ya_ref if o != k and abs(REFS[o][1] - rlo) < 1.5]
                der = not vecinos or rlo > max(vecinos)   # el más al oeste va a la izquierda
                t = ax.text(x2 + (0.12 if der else -0.12), y2 - 0.32, k, fontsize=8.5, color=C_TXT,
                            ha="left" if der else "right", va="top", zorder=7)
                t.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])
            xl_, yl_ = ax.get_xlim(), ax.get_ylim()
            for txt_ in ax.texts:   # quita textos cuyo punto de anclaje cae fuera del mapa
                if txt_.get_transform() == ax.transData:
                    tx_, ty_ = txt_.get_position()
                    if not (xl_[0] <= tx_ <= xl_[1] and yl_[0] <= ty_ <= yl_[1]):
                        txt_.set_visible(False)
            ampliacion_pendiente = None
            # ampliación del Istmo en el hueco del mapa que no tape sistemas, líneas ni estados
            if ISTMO_DESTACAR:
                C_ISTMO = "#0C447C"
                xl, yl = ax.get_xlim(), ax.get_ylim()
                def a_frac(pts):
                    if not pts:
                        return np.zeros((0, 2))
                    xy = np.array([merc(lo, la) for lo, la in pts])
                    f_ = np.column_stack(((xy[:, 0] - xl[0]) / (xl[1] - xl[0]), (xy[:, 1] - yl[0]) / (yl[1] - yl[0])))
                    return f_[(f_[:, 0] >= 0) & (f_[:, 0] <= 1) & (f_[:, 1] >= 0) & (f_[:, 1] <= 1)]
                # pesado (no tapar nunca): territorio de los estados, puntos y sus etiquetas, sistemas, localidades
                duro = [tuple(c) for ps in interes.values() for p_ in ps for c in p_]
                for _, la, lo, _ in elegidos_todos:
                    duro += [(lo, la), (lo, la - 0.45), (lo + 0.4, la - 0.45), (lo - 0.4, la - 0.45)]
                for lo_, la_ in [(p_["lon"], p_["lat"]) for p_ in puntos] + [(t_["lon"], t_["lat"]) for t_ in otras2]:
                    duro += [(lo_ + dx, la_ + dy) for dx in (-0.6, 0, 0.6, 1.6) for dy in (-0.6, 0, 0.6)]   # símbolo y etiqueta
                duro += [(rlo, rla) for k_ in ya_ref for rla, rlo in [REFS[k_]]]
                # medio: líneas de distancia visibles
                medio = []
                for _, la, lo, _, es_z in objetivos:
                    x1, y1 = merc(lo, la)
                    if not es_z and not (xl[0] <= x1 <= xl[1] and yl[0] <= y1 <= yl[1]):
                        continue
                    for _, (rla, rlo), _, _ in refs_para(la, lo):
                        medio += [(lo + (rlo - lo) * s, la + (rla - la) * s) for s in np.linspace(0, 1, 40)]
                # ligero: área sombreada de las zonas de baja presión (se puede encimar si no hay otro hueco)
                ligero = []
                for zn in zonas2:   # la flecha de desplazamiento del NHC tampoco se tapa
                    fl = zn.get("flecha") or []
                    for c1, c2 in zip(fl[:-1], fl[1:]):
                        duro += [(c1[0] + (c2[0] - c1[0]) * s, c1[1] + (c2[1] - c1[1]) * s) for s in np.linspace(0, 1, 15)]
                for zn in zonas2:
                    for a_ in zn["anillos"]:
                        rutaz = MPath(np.array(a_))
                        for gx in np.arange(dom2[0], dom2[1], 0.4):
                            for gy in np.arange(dom2[2], dom2[3], 0.4):
                                if rutaz.contains_point((gx, gy)):
                                    ligero.append((gx, gy))
                    pz = [c for a_ in zn["anillos"] for c in a_]
                    duro.append((sum(c[0] for c in pz) / len(pz), dom2[2] + 0.3))   # etiqueta de la zona
                F_duro, F_medio, F_lig = a_frac(duro), a_frac(medio), a_frac(ligero)
                bx0, by0 = merc(ISTMO_CAJA[0], ISTMO_CAJA[2]); bx1, by1 = merc(ISTMO_CAJA[1], ISTMO_CAJA[3])
                relacion = ((bx1 - bx0) / (by1 - by0)) * ((yl[1] - yl[0]) / (xl[1] - xl[0]))   # ancho/alto en fracción
                caja_c = (((bx0 + bx1) / 2 - xl[0]) / (xl[1] - xl[0]), ((by0 + by1) / 2 - yl[0]) / (yl[1] - yl[0]))
                def cuenta(F, r, m=0.012):
                    return int(np.sum((F[:, 0] > r[0] - m) & (F[:, 0] < r[0] + r[2] + m) &
                                      (F[:, 1] > r[1] - m) & (F[:, 1] < r[1] + r[3] + m))) if len(F) else 0
                pos_i, mejor = None, None
                for escala in (1.0, 0.87, 0.75, 0.64):
                    fh = ISTMO_AMPLIACION_TAM[1] * escala; fw = fh * relacion
                    if fw > ISTMO_AMPLIACION_TAM[0] * escala:
                        fw = ISTMO_AMPLIACION_TAM[0] * escala; fh = fw / relacion
                    for x0_ in np.arange(0.01, 0.99 - fw, 0.02):
                        for y0_ in np.arange(0.02, 0.98 - fh, 0.02):
                            r = [x0_, y0_, fw, fh]
                            dist = math.hypot(x0_ + fw / 2 - caja_c[0], y0_ + fh / 2 - caja_c[1])
                            costo = (1000 * cuenta(F_duro, r) + 20 * cuenta(F_medio, r)
                                     + 1 * cuenta(F_lig, r) + 5 * dist)
                            if mejor is None or costo < mejor:
                                pos_i, mejor = r, costo
                    if mejor is not None and mejor < 1000:   # cupo sin tapar territorio: no hace falta achicar
                        break
                def _dibujar_ampliacion(axi):
                    base_mapa(axi, estados, {}, ISTMO_CAJA, fondo="#f4f8fb", nombres=False)
                    axi.set_xticks([]); axi.set_yticks([]); axi.grid(False)
                    for s in axi.spines.values():
                        s.set_edgecolor(C_ISTMO); s.set_linewidth(1.6)
                    cfi = axi.contourf(XM, YM, np.ma.masked_less(Z, niveles[0]), levels=niveles, cmap=cmap,
                                       norm=BoundaryNorm(niveles, cmap.N), zorder=2)
                    clip_i = matplotlib.patches.PathPatch(MPath(verts, codes), transform=axi.transData,
                                                          fc="none", ec="none")
                    axi.add_patch(clip_i)
                    cfi.set_clip_path(clip_i)
                    for ps in interes.values():
                        for p_ in ps:
                            axi.add_patch(Polygon([merc(*c) for c in p_], closed=True, fc="none", ec="#2c2c2a",
                                                  lw=0.9, zorder=3))
                    axi.text(0.03, 0.97, "ISTMO DE TEHUANTEPEC", transform=axi.transAxes, fontsize=8.5,
                             ha="left", va="top", color=C_ISTMO, fontweight="bold", zorder=9,
                             bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
                    etqs_i = []
                    for nom, la, lo, lado, v, pr in etiq_istmo:   # solo el nombre: los valores van en el panel
                        x, y = merc(lo, la)
                        axi.plot(x, y, "o", ms=5, color=C_ISTMO, mec="white", mew=0.8, zorder=8)
                        etqs_i.append(axi.annotate(
                            nom, (x, y), xytext=(8 if lado == "d" else -8, 0), textcoords="offset points",
                            fontsize=7.6, ha="left" if lado == "d" else "right", va="center", color="#2c2c2a",
                            fontweight="bold", zorder=9,
                            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.8),
                            arrowprops=dict(arrowstyle="-", color="#5f5e5a", lw=0.5, shrinkA=0, shrinkB=3)))
                    reacomodar_etiquetas(axi, etqs_i, iteraciones=120, max_px=45, min_y=-45)
                if ISTMO_EN_GOLFO:
                    g0, g1, gla = GOLFO_CAJA
                    gx0, gy0 = merc(g0, gla); gx1, _ = merc(g1, gla)
                    fw_ = (gx1 - gx0) / (xl[1] - xl[0])
                    fh_ = fw_ / relacion
                    fx_ = (gx0 - xl[0]) / (xl[1] - xl[0]); fy_ = (gy0 - yl[0]) / (yl[1] - yl[0])
                    fy_ = min(fy_, 0.985 - fh_)
                    axi = ax.inset_axes([fx_, fy_, fw_, fh_])
                    _dibujar_ampliacion(axi)
                    ax.indicate_inset_zoom(axi, edgecolor=C_ISTMO, lw=1.2, alpha=0.9)
                elif ISTMO_EN_PANEL:
                    # en el mapa solo queda el recuadro del Istmo; la ampliación va debajo del panel,
                    # para dejar libres el océano y la costa
                    ax.add_patch(matplotlib.patches.Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False,
                                                              ec=C_ISTMO, lw=1.4, zorder=8))
                    ampliacion_pendiente = _dibujar_ampliacion
                else:
                    axi = ax.inset_axes(pos_i)
                    _dibujar_ampliacion(axi)
                    ax.indicate_inset_zoom(axi, edgecolor=C_ISTMO, lw=1.4, alpha=0.9)
            if DISENO == "clasico":
                # panel lateral con colores (una línea de texto por renglón)
                if lineas_panel:
                    alto_fig = fig.get_size_inches()[1] * 72
                    paso = 8.6 * 1.45 / alto_fig
                    x_txt, y_txt = pos.x1 + 0.02, pos.y1 - 0.012
                    textos = [fig.text(x_txt, y_txt - i * paso, txt, fontsize=8.6, color=color, va="top", ha="left",
                                       fontweight="bold" if negrita else "normal")
                              for i, (txt, color, negrita) in enumerate(lineas_panel)]
                    rend = fig.canvas.get_renderer()
                    ancho_txt = max(t.get_window_extent(rend).transformed(fig.transFigure.inverted()).width
                                    for t in textos)
                    alto = paso * len(lineas_panel) + 0.02
                    fig.patches.append(matplotlib.patches.FancyBboxPatch(
                        (pos.x1 + 0.01, pos.y1 - alto), ancho_txt + 0.02, alto, boxstyle="round,pad=0.004",
                        transform=fig.transFigure, fc="white", ec="#0C447C", lw=0.8, zorder=-1))
                if ampliacion_pendiente:   # ampliación del Istmo debajo del panel lateral
                    y_tope = (pos.y1 - alto) if lineas_panel else pos.y1
                    w_f = max(ancho_txt + 0.02, 0.22) if lineas_panel else 0.24
                    fw_in, fh_in = fig.get_size_inches()
                    h_f = w_f * fw_in / fh_in * (by1 - by0) / (bx1 - bx0)
                    axi = fig.add_axes([pos.x1 + 0.01, y_tope - 0.018 - h_f, w_f, h_f])
                    ampliacion_pendiente(axi)
            fin = ahora + dt.timedelta(days=QPF_DIAS - 1)
            region_txt = f"{', '.join(ESTADOS_INTERES[:-1])} y {ESTADOS_INTERES[-1]}"
            periodo_txt = f"Del {ahora.day} al {fin.day} {MESES[fin.month - 1]} {fin.year}"
            if DISENO == "clasico":
                ax.set_title(f"{cfg['titulo']} en {region_txt}\n{periodo_txt}", fontsize=12)
            else:
                aplicar_diseno(fig, ax, cax, lineas_panel, ampliacion_pendiente, (bx1 - bx0, by1 - by0),
                               cfg["titulo"], region_txt, periodo_txt, cfg["unidad"] == "km/h")
            if lluvia_vieja:  # solo se avisa si se usó el respaldo
                ax.text(0.01, 0.99, f"Datos del modelo del {lluvia_vieja}", transform=ax.transAxes,
                        fontsize=8, va="top", color="#A32D2D", zorder=9,
                        bbox=dict(fc="white", ec="#A32D2D", lw=0.6, pad=2))
            guardar(fig, f"{nombre}_{cfg['archivo']}.png", ahora)
    log.info("Listo.")
    return 0


def reacomodar_etiquetas(ax, etiquetas, iteraciones=160, max_px=80, min_y=6, prohibidas=()):
    """Separa las etiquetas que se enciman empujándolas entre sí (y lejos de los otros puntos);
    una línea guía une cada etiqueta con su punto."""
    fig = ax.figure
    ax.apply_aspect()
    rend = fig.canvas.get_renderer()
    pts = [ax.transData.transform(e.xy) for e in etiquetas]
    off = [list(e.get_position()) for e in etiquetas]
    esc = fig.dpi / 72.0
    for _ in range(iteraciones):
        cajas = [e.get_window_extent(rend).expanded(1.08, 1.15) for e in etiquetas]
        movido = False
        for i in range(len(etiquetas)):
            for j in range(len(etiquetas)):
                if i == j:
                    continue
                a = cajas[i]
                # choque con otra etiqueta o con el punto de otra etiqueta
                b = cajas[j]
                px, py = pts[j]
                choca_caja = a.overlaps(b)
                choca_punto = a.x0 - 6 < px < a.x1 + 6 and a.y0 - 6 < py < a.y1 + 6
                if not (choca_caja or choca_punto):
                    continue
                cx_a, cy_a = (a.x0 + a.x1) / 2, (a.y0 + a.y1) / 2
                cx_b, cy_b = ((b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2) if choca_caja else (px, py)
                dx, dy = cx_a - cx_b, cy_a - cy_b
                if abs(dx) < 1e-3 and abs(dy) < 1e-3:
                    dx = 1.0
                if cy_a >= cy_b:              # se separan de preferencia hacia el norte (tierra), no al océano
                    dy = abs(dy) + 4
                n = math.hypot(dx, dy)
                off[i][0] += dx / n * 2.2 / esc
                off[i][1] += dy / n * 2.2 / esc
                off[i][0] = max(-max_px, min(max_px, off[i][0]))
                off[i][1] = max(min_y, min(max_px, off[i][1]))   # siempre arriba del punto: el océano queda libre
                etiquetas[i].set_position(off[i])
                movido = True
                cajas[i] = etiquetas[i].get_window_extent(rend).expanded(1.08, 1.15)
            caja_ax = ax.get_window_extent(rend)   # que no se salga del mapa
            a = cajas[i]
            if a.x1 > caja_ax.x1 - 3 or a.x0 < caja_ax.x0 + 3 or a.y1 > caja_ax.y1 - 3:
                off[i][0] += (-(a.x1 - caja_ax.x1 + 3) if a.x1 > caja_ax.x1 - 3 else
                              (caja_ax.x0 + 3 - a.x0) if a.x0 < caja_ax.x0 + 3 else 0) / esc
                if a.y1 > caja_ax.y1 - 3:
                    off[i][1] -= (a.y1 - caja_ax.y1 + 3) / esc
                etiquetas[i].set_position(off[i])
                cajas[i] = etiquetas[i].get_window_extent(rend).expanded(1.08, 1.15)
            for (bx0_, by0_, bx1_, by1_) in prohibidas:   # p. ej. la ampliación del Istmo
                a = cajas[i]
                if a.x1 > bx0_ and a.x0 < bx1_ and a.y1 > by0_ and a.y0 < by1_:
                    hacia = -1 if (a.x0 + a.x1) / 2 < (bx0_ + bx1_) / 2 else 1
                    off[i][0] = max(-max_px * 2, min(max_px * 2, off[i][0] + hacia * 3 / esc))
                    etiquetas[i].set_position(off[i])
                    movido = True
        if not movido:
            break


# ----------------------------- diseños de acomodo -----------------------------
COL_TARJETA = {"ISTMO": "#0C447C", "MAYOR": "#2E5E7E", "RACHAS": "#2E5E7E", "SISTEMAS": "#7A2E2E"}


def _grupos_panel(lineas):
    """Separa las líneas del panel en secciones (cada una empieza con su título)."""
    grupos, actual = [], []
    for ln in lineas:
        if not ln[0].strip():
            if actual:
                grupos.append(actual); actual = []
        else:
            actual.append(ln)
    if actual:
        grupos.append(actual)
    return grupos


def _tarjeta(fig, x, ytop, w, grupo, fs=8.2, extra_h=0.0, oscuro=False):
    """Tarjeta con franja de título de color y sus renglones. Devuelve (y_inferior, y_tope_extra)."""
    fh = fig.get_size_inches()[1] * 72
    paso = fs * 1.5 / fh
    # renglones que no caben en el ancho de la tarjeta se parten en dos o más
    max_car = max(20, int((w - 0.018) * fig.get_size_inches()[0] * 72 / (fs * 0.50)))
    titulo = grupo[0][0]
    if len(titulo) > int(max_car * 0.72):
        titulo = titulo.split(" Y ")[0].split(" (")[0]
    lineas = [grupo[0]]
    for txt, color, negrita in grupo[1:]:
        partes = textwrap.wrap(txt.strip(), max_car, subsequent_indent="   ") or [""]
        lineas += [(pt, color, negrita) for pt in partes]
    grupo = [(titulo, grupo[0][1], grupo[0][2])] + lineas[1:]
    col = next((c for k, c in COL_TARJETA.items() if titulo.upper().startswith(k)), "#0C447C")
    hh = paso * 1.7
    cuerpo = paso * (len(grupo) - 1) + paso * 0.8 + (extra_h if extra_h else 0)
    y0 = ytop - hh - cuerpo
    fondo = matplotlib.patches.FancyBboxPatch(
        (x, y0), w, hh + cuerpo, boxstyle="round,pad=0,rounding_size=0.006", transform=fig.transFigure,
        fc="white", ec="none" if oscuro else "#d3dce5", lw=0.8, zorder=-1)
    fig.patches.append(fondo)
    _tarjeta.ultimo_fondo = fondo
    fig.patches.append(matplotlib.patches.FancyBboxPatch(
        (x, ytop - hh), w, hh, boxstyle="round,pad=0,rounding_size=0.006", transform=fig.transFigure,
        fc=col, ec="none", zorder=-1))
    fig.text(x + 0.009, ytop - hh / 2, titulo, fontsize=fs + 0.6, color="white", fontweight="bold",
             va="center", ha="left")
    y = ytop - hh - paso * 0.45
    for txt, color, negrita in grupo[1:]:
        fig.text(x + 0.009, y, txt.strip(), fontsize=fs, color=color, va="top", ha="left",
                 fontweight="bold" if negrita else "normal")
        y -= paso
    return y0, y + paso * 0.1


def aplicar_diseno(fig, ax, cax, lineas, ampliacion, caja_wh, titulo, region, periodo, con_nudos):
    grupos = _grupos_panel(lineas)
    orden = {"SISTEMAS": 0, "MAYOR": 1, "RACHAS": 1, "ISTMO": 2}
    grupos.sort(key=lambda g: next((v for k, v in orden.items() if g[0][0].upper().startswith(k)), 3))
    istmo = [g for g in grupos if g[0][0].upper().startswith("ISTMO")]
    otros = [g for g in grupos if not g[0][0].upper().startswith("ISTMO")]
    sep_cb = 0.105 if con_nudos else 0.075

    def colocar_mapa(rect):
        ax.set_position(rect); ax.apply_aspect()
        p = ax.get_position()
        cax.set_position([p.x0, p.y0 - sep_cb * (8.4 / fig.get_size_inches()[1]), p.width, 0.022 * (8.4 / fig.get_size_inches()[1])])
        return p

    def alto_inset(w):
        fw, fh = fig.get_size_inches()
        return (w - 0.012) * fw / fh * caja_wh[1] / caja_wh[0] if ampliacion else 0.0

    def inset_en(x, ytop_extra, w):
        """Coloca la ampliación pegada al texto y ajusta el fondo de la tarjeta a su tamaño real."""
        if not ampliacion:
            return None
        fondo = _tarjeta.ultimo_fondo
        h = alto_inset(w)
        axi = fig.add_axes([x + 0.006, ytop_extra - h, w - 0.012, h])
        ampliacion(axi)
        axi.apply_aspect()
        p = axi.get_position()
        axi.set_position([x + (w - p.width) / 2, ytop_extra - p.height, p.width, p.height])
        nuevo_y0 = ytop_extra - p.height - 0.008
        fondo.set_height(fondo.get_y() + fondo.get_height() - nuevo_y0)
        fondo.set_y(nuevo_y0)
        return nuevo_y0

    if DISENO == "lateral":
        fig.set_size_inches(14.5, 8.4)
        p = colocar_mapa([0.035, 0.15, 0.645, 0.75])
        ax.set_title(f"{titulo} en {region}\n{periodo}", fontsize=12.5, fontweight="bold", color="#15212C")
        x, w, y = 0.70, 0.285, p.y1
        for g in otros:
            y = _tarjeta(fig, x, y, w, g)[0] - 0.014
        for g in istmo:
            y0, ye = _tarjeta(fig, x, y, w, g, extra_h=alto_inset(w))
            inset_en(x, ye, w)
    elif DISENO == "tablero":
        fig.set_size_inches(13, 12.5)
        p = colocar_mapa([0.04, 0.43, 0.92, 0.50])
        ax.set_title(f"{titulo} en {region}\n{periodo}", fontsize=13, fontweight="bold", color="#15212C")
        ytop = cax.get_position().y0 - 0.045
        cols = otros + istmo
        n = max(len(cols), 1)
        gap = 0.015
        w = (p.width - gap * (n - 1)) / n
        for i, g in enumerate(cols):
            x = p.x0 + i * (w + gap)
            es_istmo = g[0][0].upper().startswith("ISTMO")
            y0, ye = _tarjeta(fig, x, ytop, w, g, fs=8.0, extra_h=alto_inset(w) if es_istmo else 0)
            if es_istmo:
                inset_en(x, ye, w)
    elif DISENO == "marino":
        fig.set_size_inches(14.5, 8.4)
        colocar_mapa([0.31, 0.14, 0.675, 0.78])
        x, w = 0.016, 0.262
        tit = textwrap.wrap(titulo, 28)
        yy = 0.968
        for ln in tit:
            fig.text(x, yy, ln, fontsize=13, color="white", fontweight="bold", va="top"); yy -= 0.033
        fig.text(x, yy - 0.004, region, fontsize=10.5, color="#BFD3E6", va="top")
        fig.text(x, yy - 0.034, periodo, fontsize=10.5, color="#4CC3BB", fontweight="bold", va="top")
        y = yy - 0.075
        for g in otros:
            y = _tarjeta(fig, x, y, w, g, oscuro=True)[0] - 0.012
        for g in istmo:
            y0, ye = _tarjeta(fig, x, y, w, g, extra_h=alto_inset(w), oscuro=True)
            y = inset_en(x, ye, w) or y0
        fig.patches.append(matplotlib.patches.Rectangle((0, min(0, y - 0.02)), 0.294, 1 - min(0, y - 0.02),
                                                        transform=fig.transFigure, fc="#0E2A44", ec="none", zorder=-2))


def enviar_ntfy(ruta, titulo):
    """Manda la imagen a ntfy.sh. Si falla, solo deja aviso en el registro."""
    if not NTFY_TEMA:
        return
    for intento in (1, 2):
        try:
            r = requests.put(f"https://ntfy.sh/{NTFY_TEMA}", data=ruta.read_bytes(),
                             headers={"Filename": ruta.name, "Title": titulo,
                                      "Tags": "cloud_with_rain"},
                             timeout=90)
            r.raise_for_status()
            log.info("Enviado a ntfy (%s): %s", NTFY_TEMA, ruta.name)
            return
        except Exception as e:
            log.warning("No se pudo enviar a ntfy (intento %d): %s", intento, e)
            time.sleep(15)


def guardar(fig, archivo, ahora):
    ruta = OUT_DIR / archivo
    fig.savefig(ruta, bbox_inches="tight")
    if GUARDAR_HISTORIAL:
        h = OUT_DIR / "historial"; h.mkdir(exist_ok=True)
        fig.savefig(h / f"{ahora:%Y%m%d_%H%M}_{archivo}", bbox_inches="tight")
    plt.close(fig)
    log.info("Guardado %s", ruta)
    if archivo.endswith(("_lluvia.png", "_viento.png")):
        tipo = "lluvia" if archivo.endswith("_lluvia.png") else "viento"
        enviar_ntfy(ruta, f"Mapa de {tipo} {ahora:%d/%m %H:%M} h")


# ----------------------------- revisión de cambios -----------------------------
def _cat(kt):
    kt = kt or 0
    for lim, c in ((137, "H5"), (113, "H4"), (96, "H3"), (83, "H2"), (64, "H1"), (34, "TT")):
        if kt >= lim:
            return c
    return "DT"


def _pct(s):
    try:
        return int(str(s).replace("%", "").strip())
    except ValueError:
        return 0


def _nivel(p):   # categorías del NHC: baja < 40 %, media 40-60 %, alta > 60 %
    return "alta" if p > 60 else "media" if p >= 40 else "baja"


def revisar_fuentes(estado):
    """Consulta ligera de las fuentes, solo de lo que está en la jurisdicción.
    Devuelve (huella, resumen, corrida): la huella detecta cualquier cambio; el resumen sirve para
    decidir si el cambio es muy significativo."""
    h, res = {}, {"ciclones": {}, "zonas": None}
    data = {}
    try:  # ciclones activos dentro de la jurisdicción
        data = http_get(URL_CURRENT, intentos=2).json()
        lista = []
        for s in data.get("activeStorms", []):
            try:
                la, lo = float(s["latitudeNumeric"]), float(s["longitudeNumeric"])
            except (KeyError, TypeError, ValueError):
                continue
            if not en_jurisdiccion(lo, la):
                continue
            kt = _num(s.get("intensity")) or 0
            res["ciclones"][s.get("id", "?")] = {"nombre": s.get("name", "?"), "kt": kt, "cat": _cat(kt)}
            lista.append(f"{s.get('id')}|{(s.get('publicAdvisory') or {}).get('advNum')}|{kt}|{la}|{lo}")
        h["nhc"] = sorted(lista)
    except Exception as e:
        log.warning("Revisión NHC falló: %s", e)
        res["ciclones"] = None
    try:  # zonas de baja presión: solo se descargan si el NHC publicó un archivo nuevo
        r = requests.head(URL_TWO, headers=HEADERS, timeout=30)
        lm = r.headers.get("Last-Modified") or r.headers.get("ETag") or ""
        if lm and lm == estado.get("two_lm") and "zonas" in estado and estado.get("zonas_v") == 5:
            res["zonas"] = estado["zonas"]
        else:
            zs = []
            pos_c = [(float(s["latitudeNumeric"]), float(s["longitudeNumeric"]))
                     for s in (data.get("activeStorms", []) if res["ciclones"] is not None else [])
                     if s.get("latitudeNumeric") is not None]
            for z in quitar_zonas_ya_ciclon(zonas_baja_presion(), pos_c):
                pz = [c for a_ in z["anillos"] for c in a_]
                if not zona_en_jurisdiccion(z):
                    continue
                la_c, lo_c, _ = centro_zona(z)
                zs.append({"lat": round(la_c, 1), "lon": round(lo_c, 1), "n": z.get("num", ""),
                           "p2": _pct(z["p2"]), "p7": _pct(z["p7"])})
            res["zonas"] = sorted(zs, key=lambda z: z["lon"])
            estado["two_lm"] = lm
            estado["zonas_v"] = 5   # versión del cálculo: posición de la perturbación del NHC
        estado["zonas"] = res["zonas"]
        h["zonas"] = [f"{z['lat']}|{z['lon']}|{z['p2']}|{z['p7']}" for z in res["zonas"]]
    except Exception as e:
        log.warning("Revisión zonas de baja presión falló: %s", e)
    corrida = ""
    try:  # corrida más reciente del modelo de lluvia y viento
        meta = http_get(URL_META.format(modelo=QPF_MODEL), intentos=2).json()
        corrida = str(meta.get("last_run_initialisation_time", ""))
        h["modelo"] = corrida
    except Exception as e:
        log.warning("Revisión del modelo falló: %s", e)
    return h, res, corrida


def cambios_significativos(antes, ahora):
    """Compara el resumen de la última imagen con el actual. Devuelve la lista de motivos."""
    motivos = []
    if not antes:
        return motivos
    c0, c1 = antes.get("ciclones"), ahora.get("ciclones")
    if c0 is not None and c1 is not None:
        for i_, s in c1.items():
            if i_ not in c0:
                motivos.append(f"{s['nombre']} entró a la jurisdicción ({s['cat']})")
            else:
                if s["cat"] != c0[i_]["cat"]:
                    motivos.append(f"{s['nombre']} cambió de categoría {c0[i_]['cat']} → {s['cat']}")
                elif abs(s["kt"] - c0[i_]["kt"]) >= UMBRAL_KT:
                    motivos.append(f"{s['nombre']} cambió {s['kt'] - c0[i_]['kt']:+.0f} kt")
        for i_, s in c0.items():
            if i_ not in c1:
                motivos.append(f"{s['nombre']} salió de la jurisdicción o se disipó")
    z0, z1 = antes.get("zonas"), ahora.get("zonas")
    if z0 is not None and z1 is not None:
        usadas = set()
        for z in z1:
            cerca = [k for k, a in enumerate(z0) if k not in usadas and math.hypot(a["lat"] - z["lat"], a["lon"] - z["lon"]) < 4]
            if not cerca:
                motivos.append(f"nueva zona de baja presión ({z['p7']}% en 7 días)")
                continue
            k = min(cerca, key=lambda k: math.hypot(z0[k]["lat"] - z["lat"], z0[k]["lon"] - z["lon"]))
            usadas.add(k)
            a = z0[k]
            if (_nivel(a["p2"]) != _nivel(z["p2"]) or _nivel(a["p7"]) != _nivel(z["p7"])
                    or abs(a["p2"] - z["p2"]) >= UMBRAL_PROB or abs(a["p7"] - z["p7"]) >= UMBRAL_PROB):
                motivos.append(f"zona de baja presión: {a['p2']}/{a['p7']}% → {z['p2']}/{z['p7']}% (2/7 días)")
        if len(z0) > len(usadas):
            motivos.append("una zona de baja presión dejó de afectar la jurisdicción")
    return motivos


# ----------------------------- página de monitoreo -----------------------------
PLANTILLA_MONITOR = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta http-equiv="refresh" content="300">
<title>Monitoreo de mapas · Guerrero–Chiapas</title>
<style>
:root{--bg:#f4f6f8;--card:#fff;--txt:#1f2328;--sub:#5b6470;--borde:#dde2e8;--azul:#0c447c;
--ok:#2e7d32;--ok-bg:#e8f5e9;--av:#b26a00;--av-bg:#fff4e0;--mal:#b3261e;--mal-bg:#fdecea;--ambar:#8a5a00;--rojo:#a32d2d;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--bg:#14171b;--card:#1d2228;--txt:#e7eaee;--sub:#9aa4b1;--borde:#2f3640;--azul:#7fb3ea;
--ok:#7bd88a;--ok-bg:#16301b;--av:#f2b75b;--av-bg:#33270f;--mal:#ff8a80;--mal-bg:#3a1714;--ambar:#f2b75b;--rojo:#ff8a80;color-scheme:dark}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);font:15px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
padding:env(safe-area-inset-top,0) 0 env(safe-area-inset-bottom,0)}
header{background:var(--azul);color:#fff;padding:14px 18px}
@media (prefers-color-scheme:dark){header{background:#0f2a47}.pest button.act{color:#0b1b2e}}
header h1{margin:0;font-size:1.15rem}
header p{margin:2px 0 0;font-size:.85rem;opacity:.85}
main{max-width:1150px;margin:0 auto;padding:14px}
.tarjetas{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin-bottom:14px}
.t{background:var(--card);border:1px solid var(--borde);border-radius:12px;padding:12px 14px}
.t h2{margin:0 0 4px;font-size:.78rem;text-transform:uppercase;letter-spacing:.04em;color:var(--sub)}
.t .v{font-size:1.05rem;font-weight:600}
.t .s{font-size:.85rem;color:var(--sub);margin-top:2px}
.est{display:inline-block;padding:3px 10px;border-radius:999px;font-weight:700;font-size:.9rem}
.est.ok{background:var(--ok-bg);color:var(--ok)}.est.av{background:var(--av-bg);color:var(--av)}.est.mal{background:var(--mal-bg);color:var(--mal)}
.sis{margin:4px 0 0;padding:0;list-style:none;font-size:.9rem}
.sis li{padding:3px 0;border-top:1px dashed var(--borde)}.sis li:first-child{border-top:0}
.ciclon{color:var(--rojo);font-weight:600}.zona{color:var(--ambar);font-weight:600}
.pest{display:flex;gap:6px;margin-bottom:8px}
.pest button{flex:1;max-width:200px;padding:9px 12px;border:1px solid var(--borde);background:var(--card);color:var(--txt);
border-radius:10px;font:inherit;font-weight:600;cursor:pointer}
.pest button.act{background:var(--azul);color:#fff;border-color:var(--azul)}
.mapa{background:var(--card);border:1px solid var(--borde);border-radius:12px;padding:8px;margin-bottom:14px}
.mapa img{width:100%;height:auto;display:block;border-radius:6px;cursor:zoom-in;background:#fff}
.mapa .pie{font-size:.82rem;color:var(--sub);padding:6px 4px 2px}
.vacio{padding:40px 10px;text-align:center;color:var(--sub)}
section.t{margin-bottom:14px}
table{width:100%;border-collapse:collapse;font-size:.88rem}
td{padding:6px 4px;border-top:1px solid var(--borde);vertical-align:top}
td:first-child{white-space:nowrap;color:var(--sub);width:1%}
.sig{color:var(--mal);font-weight:600}
details summary{cursor:pointer;font-weight:600}
pre{overflow-x:auto;font-size:.75rem;background:var(--bg);padding:10px;border-radius:8px;white-space:pre}
#zoom{position:fixed;inset:0;background:rgba(0,0,0,.92);display:none;overflow:auto;z-index:10;cursor:zoom-out}
#zoom img{display:block;width:1600px;max-width:none;margin:auto}
@media (max-width:640px){#zoom img{width:220vw}}
</style>
</head>
<body>
<header><h1>Monitoreo de mapas de lluvia y viento</h1><p>Guerrero · Oaxaca · Chiapas — página generada el <span id="gen"></span></p></header>
<main>
<div class="tarjetas">
 <div class="t"><h2>Estado del monitoreo</h2><div id="estado"></div><div class="s" id="estado_s"></div></div>
 <div class="t"><h2>Última imagen</h2><div class="v" id="img_t"></div><div class="s" id="img_m"></div></div>
 <div class="t"><h2>Siguiente imagen</h2><div class="v" id="sig_t"></div><div class="s" id="sig_s"></div></div>
 <div class="t"><h2>Sistemas en la jurisdicción</h2><ul class="sis" id="sistemas"></ul></div>
</div>
<div class="pest"><button id="b_lluvia" class="act">Lluvia</button><button id="b_viento">Viento</button></div>
<div class="mapa" id="mapa"></div>
<section class="t" id="sec_log"><details><summary>Registro (últimas líneas)</summary><pre id="log"></pre></details></section>
</main>
<div id="zoom"><img alt=""></div>
<script id="datos" type="application/json">__DATOS__</script>
<script>
const D = JSON.parse(document.getElementById("datos").textContent);
const $ = id => document.getElementById(id);
const f = s => s ? new Date(s).toLocaleString("es-MX",{day:"2-digit",month:"short",hour:"2-digit",minute:"2-digit"}) : "—";
const hace = s => { if(!s) return null; return (Date.now()-new Date(s).getTime())/60000; };
const txtMin = m => m < 60 ? Math.round(m)+" min" : (m/60).toFixed(1)+" h";
$("gen").textContent = f(D.generado);
// estado: la revisión corre cada 10 min; si pasan más de 25 min sin revisión, algo falla
const mr = hace(D.ultima_revision);
let cls="ok", t="En funcionamiento";
if (mr===null || mr>D.umbral_mal){cls="mal"; t="Sin revisiones recientes";} else if (mr>D.umbral_av){cls="av"; t="Revisión retrasada";}
$("estado").innerHTML = `<span class="est ${cls}">${t}</span>`;
$("estado_s").textContent = mr===null ? "Aún no hay revisiones" : `Última revisión hace ${txtMin(mr)} · ${D.decision||""}`;
$("img_t").textContent = f(D.ultima_imagen);
$("img_m").textContent = D.motivo_imagen || "";
if (D.pendientes && D.pendientes.length){
  const mi = hace(D.ultima_imagen) || 0, falta = Math.max(0, D.intervalo_h*60 - mi);
  $("sig_t").textContent = falta>0 ? `En ${txtMin(falta)}` : "En la próxima revisión";
  $("sig_s").textContent = "Cambios menores acumulados: " + D.pendientes.join(", ");
} else {
  $("sig_t").textContent = "Sin cambios pendientes";
  $("sig_s").textContent = "Se genera de inmediato si hay un cambio significativo";
}
const R = D.resumen || {}, ul = $("sistemas"); let n=0;
for (const s of Object.values(R.ciclones||{})){ n++; ul.insertAdjacentHTML("beforeend",
  `<li><span class="ciclon">● ${s.nombre}</span> · ${s.cat} · ${Math.round(s.kt)} kt (${Math.round(s.kt*1.852)} km/h)</li>`); }
for (const z of (R.zonas||[])){ n++; ul.insertAdjacentHTML("beforeend",
  `<li><span class="zona">▨ Zona de baja presión B${z.n||""}</span> · ${z.p2}% / ${z.p7}% (2 / 7 días) · ≈${z.lat}°N ${Math.abs(z.lon)}°O</li>`); }
if (!n) ul.innerHTML = "<li>Sin sistemas en la jurisdicción</li>";
function mostrar(tipo){
  for (const k of ["lluvia","viento"]) $("b_"+k).classList.toggle("act", k===tipo);
  const im = D.imagenes[tipo];
  $("mapa").innerHTML = im ? `<img src="${im.src}" alt="Mapa de ${tipo}"><div class="pie">Generada: ${f(im.fecha)} · toque la imagen para ampliar</div>`
                           : `<div class="vacio">Aún no hay mapa de ${tipo}</div>`;
  const img = $("mapa").querySelector("img");
  if (img) img.onclick = () => { $("zoom").querySelector("img").src = img.src; $("zoom").style.display="block"; };
}
$("zoom").onclick = () => $("zoom").style.display="none";
$("b_lluvia").onclick = () => mostrar("lluvia"); $("b_viento").onclick = () => mostrar("viento");
mostrar("lluvia");
if (D.log && D.log.length) $("log").textContent = D.log.join("\n"); else $("sec_log").style.display = "none";
</script>
</body>
</html>
"""


def armar_monitor(estado, web=False):
    """HTML de la página. Local: imágenes incluidas y registro. Web: imágenes como archivos aparte, sin registro."""
    nombre = ARCHIVO_BASE
    imagenes = {}
    for tipo in ("lluvia", "viento"):
        ruta = OUT_DIR / f"{nombre}_{tipo}.png"
        if ruta.exists():
            fecha = dt.datetime.fromtimestamp(ruta.stat().st_mtime)
            src = (f"{tipo}.jpg?v={fecha:%Y%m%d%H%M}" if web else
                   "data:image/png;base64," + base64.b64encode(ruta.read_bytes()).decode())
            imagenes[tipo] = {"src": src, "fecha": fecha.isoformat()}
        elif web and tipo in estado.get("imagenes_web", {}):
            imagenes[tipo] = estado["imagenes_web"][tipo]
    log_lineas = []
    reg = OUT_DIR / "registro.log"
    if reg.exists() and not web:
        log_lineas = reg.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
    av = 25 if not web else PUBLICAR_ESTADO_MIN + 20   # minutos sin revisión antes de marcar retraso
    datos = {
        "generado": dt.datetime.now().isoformat(),
        "ultima_revision": estado.get("ultima_revision"), "decision": estado.get("decision", ""),
        "ultima_imagen": estado.get("ultima_imagen"), "motivo_imagen": estado.get("motivo_imagen", ""),
        "pendientes": estado.get("pendientes", []), "intervalo_h": INTERVALO_FORZADO_H,
        "resumen": estado.get("resumen_actual") or {}, "historial": estado.get("historial", []),
        "imagenes": imagenes, "log": log_lineas, "umbral_av": av, "umbral_mal": av * 2 + 10,
    }
    js = json.dumps(datos, ensure_ascii=False).replace("</", "<\\/")
    return PLANTILLA_MONITOR.replace("__DATOS__", js)


def escribir_monitor(estado):
    """Página local autocontenida. Se actualiza en cada revisión."""
    if not MONITOR_HTML:
        return
    try:
        tmp = OUT_DIR / (MONITOR_HTML + ".tmp")
        tmp.write_text(armar_monitor(estado), encoding="utf-8")
        tmp.replace(OUT_DIR / MONITOR_HTML)   # reemplazo atómico: nunca queda a medias
    except Exception as e:
        log.warning("No se pudo actualizar %s: %s", MONITOR_HTML, e)


def _jpg(ruta):
    """Copia ligera de un mapa para la web (JPEG), para no inflar el repositorio."""
    from PIL import Image
    buf = io.BytesIO()
    Image.open(ruta).convert("RGB").save(buf, "JPEG", quality=85, optimize=True)
    return buf.getvalue()


def credenciales_github():
    """(usuario, token): en la nube desde los Secrets; en la PC desde config_github.json."""
    if os.environ.get("MAPAS_TOKEN"):
        return os.environ.get("GITHUB_REPOSITORY_OWNER", ""), os.environ["MAPAS_TOKEN"]
    if CONFIG_GITHUB.exists():
        try:
            cfg = json.loads(CONFIG_GITHUB.read_text(encoding="utf-8"))
            return cfg["usuario"], cfg["token"]
        except Exception as e:
            log.warning("No se pudo leer %s: %s", CONFIG_GITHUB, e)
    return None


def minutos_desde_publicacion_pc():
    """Minutos desde el último commit de la PC en el repositorio de la página
    (se ignoran los commits de la nube). None si no se pudo consultar."""
    cred = credenciales_github()
    if not cred:
        return None
    usuario, token = cred
    try:
        r = requests.get(f"https://api.github.com/repos/{usuario}/{GITHUB_REPO}/commits",
                         params={"per_page": 50}, timeout=30,
                         headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                                  "User-Agent": usuario})
        r.raise_for_status()
        for c in r.json():
            if not c["commit"]["message"].startswith(ETIQUETA_NUBE):
                fecha = dt.datetime.fromisoformat(c["commit"]["committer"]["date"].replace("Z", "+00:00"))
                return (dt.datetime.now(dt.timezone.utc) - fecha).total_seconds() / 60
        return 1e9   # los últimos 50 commits son de la nube: la PC lleva mucho sin publicar
    except Exception as e:
        log.warning("No se pudo consultar la última publicación de la PC: %s", e)
        return None


def publicar_github(estado, con_imagenes):
    """Sube index.html (y los mapas si hay nuevos) en un solo commit al repositorio de GitHub Pages."""
    if not PUBLICAR_GITHUB:
        return False
    cred = credenciales_github()
    if not cred:
        log.info("Sin credenciales de GitHub (%s o Secret MAPAS_TOKEN); no se publica.", CONFIG_GITHUB)
        return False
    try:
        usuario, token = cred
        api = f"https://api.github.com/repos/{usuario}/{GITHUB_REPO}"
        h = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json", "User-Agent": usuario}
        archivos = {"index.html": armar_monitor(estado, web=True).encode("utf-8")}
        for tipo_resumen in ("lluvia", "viento"):   # datos para la portada "Boletín del día"
            ruta_resumen = OUT_DIR / f"resumen_{tipo_resumen}.json"
            if ruta_resumen.exists():
                archivos[f"resumen_{tipo_resumen}.json"] = ruta_resumen.read_bytes()
        if con_imagenes:
            for tipo in ("lluvia", "viento"):
                ruta = OUT_DIR / f"{ARCHIVO_BASE}_{tipo}.png"
                if ruta.exists():
                    archivos[f"{tipo}.jpg"] = _jpg(ruta)
        r = requests.get(f"{api}/git/ref/heads/main", headers=h, timeout=30)
        if r.status_code in (404, 409):   # repositorio vacío: se inicializa con index.html
            r2 = requests.put(f"{api}/contents/index.html", headers=h, timeout=60, json={
                "message": "Inicio del monitor", "content": base64.b64encode(archivos["index.html"]).decode()})
            r2.raise_for_status()
            r = requests.get(f"{api}/git/ref/heads/main", headers=h, timeout=30)
        r.raise_for_status()
        commit_padre = r.json()["object"]["sha"]
        arbol_base = requests.get(f"{api}/git/commits/{commit_padre}", headers=h, timeout=30).json()["tree"]["sha"]
        arbol = []
        for ruta_repo, contenido in archivos.items():
            rb = requests.post(f"{api}/git/blobs", headers=h, timeout=90,
                               json={"content": base64.b64encode(contenido).decode(), "encoding": "base64"})
            rb.raise_for_status()
            arbol.append({"path": ruta_repo, "mode": "100644", "type": "blob", "sha": rb.json()["sha"]})
        rt = requests.post(f"{api}/git/trees", headers=h, timeout=60, json={"base_tree": arbol_base, "tree": arbol})
        rt.raise_for_status()
        rc = requests.post(f"{api}/git/commits", headers=h, timeout=60, json={
            "message": (f"{ETIQUETA_NUBE} " if EN_NUBE else "") + f"Monitor {dt.datetime.now():%d/%m %H:%M}"
                       + (" (mapas nuevos)" if con_imagenes else ""),
            "tree": rt.json()["sha"], "parents": [commit_padre]})
        rc.raise_for_status()
        rr = requests.patch(f"{api}/git/refs/heads/main", headers=h, timeout=30, json={"sha": rc.json()["sha"]})
        rr.raise_for_status()
        log.info("Monitor publicado en GitHub (%s)%s", GITHUB_REPO, " con mapas nuevos" if con_imagenes else "")
        if con_imagenes:   # recuerda las imágenes publicadas (la nube no conserva los PNG entre corridas)
            estado["imagenes_web"] = {t: {"src": f"{t}.jpg?v={dt.datetime.now():%Y%m%d%H%M}",
                                          "fecha": dt.datetime.now().isoformat()}
                                      for t in ("lluvia", "viento") if f"{t}.jpg" in archivos}
        return True
    except Exception as e:
        log.warning("No se pudo publicar en GitHub: %s", e)
        return False


def publicar_si_toca(estado, con_imagenes):
    ult = estado.get("ultima_publicacion")
    minutos = (dt.datetime.now() - dt.datetime.fromisoformat(ult)).total_seconds() / 60 if ult else 1e9
    if con_imagenes or not estado.get("web_tiene_mapas") or minutos >= PUBLICAR_ESTADO_MIN:
        con_img = con_imagenes or not estado.get("web_tiene_mapas")
        if publicar_github(estado, con_img):
            estado["ultima_publicacion"] = dt.datetime.now().isoformat()
            if con_img:
                estado["web_tiene_mapas"] = True

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from logging.handlers import RotatingFileHandler
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[RotatingFileHandler(OUT_DIR / "registro.log", maxBytes=1_000_000,
                                                      backupCount=2, encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)])
    forzar = "--forzar" in sys.argv

    # evitar dos ejecuciones al mismo tiempo (una generación tarda unos minutos)
    candado = OUT_DIR / "en_proceso.lock"
    if candado.exists() and time.time() - candado.stat().st_mtime < 20 * 60 and not forzar:
        log.info("Ya hay una actualización en curso; se omite esta revisión.")
        return 0
    if EN_NUBE and not forzar and os.environ.get("MODO_ACTIVO", "").lower() != "si":
        minutos_pc = minutos_desde_publicacion_pc()
        if minutos_pc is not None and minutos_pc < UMBRAL_PC_MIN:
            log.info("Respaldo en la nube: la PC publicó hace %.0f min; la nube queda en espera.", minutos_pc)
            return 0
        log.info("Respaldo en la nube: la PC no publica desde hace %s; la nube toma el relevo.",
                 "un tiempo desconocido" if minutos_pc is None else f"{minutos_pc:.0f} min")

    archivo_estado = OUT_DIR / "estado.json"
    estado = json.loads(archivo_estado.read_text(encoding="utf-8")) if archivo_estado.exists() else {}

    huella, resumen, corrida = revisar_fuentes(estado)
    huella_img = estado.get("huella_imagen", {})
    # cambios desde la última imagen (una fuente que no respondió no cuenta como cambio)
    pendientes = [k for k, v in huella.items() if huella_img.get(k) != v]
    significativos = cambios_significativos(estado.get("resumen_imagen"), resumen)
    ultima = estado.get("ultima_imagen")
    horas = (dt.datetime.now() - dt.datetime.fromisoformat(ultima)).total_seconds() / 3600 if ultima else 99

    if forzar:
        motivo = "manual (--forzar)"
    elif not ultima:
        motivo = "primera imagen"
    elif significativos:
        motivo = "CAMBIO SIGNIFICATIVO: " + "; ".join(significativos)
    elif pendientes and horas >= INTERVALO_FORZADO_H:
        motivo = f"cambios acumulados en {INTERVALO_FORZADO_H} h: {', '.join(pendientes)}"
    else:
        if pendientes:
            log.info("Cambios menores (%s); se esperan las %d h. Última imagen hace %.1f h, siguiente en %.0f min.",
                     ", ".join(pendientes), INTERVALO_FORZADO_H, horas, (INTERVALO_FORZADO_H - horas) * 60)
            estado["decision"] = "cambios menores en espera"
        else:
            log.info("Sin cambios; la imagen se mantiene (última hace %.1f h).", horas)
            estado["decision"] = "sin cambios"
        if EN_NUBE:
            estado["decision"] += " · respaldo en la nube"
        estado.update({"ultima_revision": dt.datetime.now().isoformat(), "pendientes": pendientes,
                       "resumen_actual": resumen})
        escribir_monitor(estado)
        publicar_si_toca(estado, False)
        archivo_estado.write_text(json.dumps(estado, ensure_ascii=False, indent=1), encoding="utf-8")
        return 0

    log.info("Generando imágenes (%s)", motivo)
    candado.write_text(dt.datetime.now().isoformat())
    try:
        resultado = generar(corrida)
    finally:
        candado.unlink(missing_ok=True)
    ahora_ = dt.datetime.now().isoformat()
    estado.update({"ultima_revision": ahora_, "resumen_actual": resumen})
    if resultado == 0:
        huella_img.update(huella)
        estado.update({"huella_imagen": huella_img, "resumen_imagen": resumen, "ultima_imagen": ahora_,
                       "motivo_imagen": motivo, "pendientes": [], "decision": "imágenes generadas",
                       "historial": (estado.get("historial", []) + [{"fecha": ahora_, "motivo": motivo}])[-15:]})
    else:
        estado["decision"] = "falló la generación (ver registro)"
    if EN_NUBE:
        estado["decision"] += " · respaldo en la nube"
    escribir_monitor(estado)
    publicar_si_toca(estado, resultado == 0)
    archivo_estado.write_text(json.dumps(estado, ensure_ascii=False, indent=1), encoding="utf-8")
    return resultado


if __name__ == "__main__":
    sys.exit(main())
