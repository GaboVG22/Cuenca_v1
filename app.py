import io
import json
import math
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.features import shapes
from rasterio.mask import mask
from rasterio.transform import xy
import simplekml
import streamlit as st
from pyproj import CRS, Transformer
from shapely.geometry import (
    shape,
    mapping,
    Point,
    LineString,
    MultiLineString,
    GeometryCollection,
    MultiPolygon,
)
from shapely.ops import transform as shp_transform, unary_union

try:
    import ee
except Exception:
    ee = None

try:
    from pysheds.grid import Grid
except Exception:
    Grid = None

st.set_page_config(
    page_title="Cuenca desde KMZ + ASTER GDEM v3",
    layout="wide",
    initial_sidebar_state="expanded",
)

ASTER_GEE_ASSET = "projects/sat-io/open-datasets/ASTER/GDEM"

# =============================================================================
# Lectura KMZ/KML
# =============================================================================

def read_point_from_kmz(uploaded_file):
    """Lee el primer punto encontrado en un KMZ/KML. Retorna lon, lat."""
    uploaded_file.seek(0)
    data = uploaded_file.read()
    name = uploaded_file.name.lower()

    if name.endswith(".kmz"):
        with zipfile.ZipFile(io.BytesIO(data), "r") as z:
            kml_names = [n for n in z.namelist() if n.lower().endswith(".kml")]
            if not kml_names:
                raise ValueError("El KMZ no contiene un archivo KML interno.")
            kml_text = z.read(kml_names[0]).decode("utf-8", errors="ignore")
    else:
        kml_text = data.decode("utf-8", errors="ignore")

    root = ET.fromstring(kml_text)

    # Búsqueda tolerante a namespaces.
    for elem in root.iter():
        if elem.tag.lower().endswith("coordinates") and elem.text:
            # Preferir coordenadas dentro de Point si existe.
            txt = elem.text.strip().replace("\n", " ").replace("\t", " ")
            first = txt.split()[0]
            parts = first.split(",")
            if len(parts) >= 2:
                lon = float(parts[0])
                lat = float(parts[1])
                if -180 <= lon <= 180 and -90 <= lat <= 90:
                    return lon, lat

    raise ValueError("No se encontró una coordenada válida en el KMZ/KML. Use un archivo con punto de control.")

# =============================================================================
# Utilidades espaciales
# =============================================================================

def lonlat_to_local_utm_crs(lon, lat):
    zone = int((lon + 180) // 6) + 1
    epsg = 32700 + zone if lat < 0 else 32600 + zone
    return CRS.from_epsg(epsg)


def project_geom(geom, src_crs, dst_crs):
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shp_transform(transformer.transform, geom)


def polygon_area_km2_ha(poly_wgs84):
    centroid = poly_wgs84.centroid
    utm = lonlat_to_local_utm_crs(centroid.x, centroid.y)
    poly_utm = project_geom(poly_wgs84, CRS.from_epsg(4326), utm)
    area_m2 = abs(poly_utm.area)
    return area_m2 / 1_000_000.0, area_m2 / 10_000.0


def km_buffer_to_deg(lat, km):
    deg_lat = km / 111.0
    deg_lon = km / (111.0 * max(0.15, math.cos(math.radians(lat))))
    return deg_lon, deg_lat


def bbox_from_point_radius(lon, lat, radius_km):
    deg_lon, deg_lat = km_buffer_to_deg(lat, radius_km)
    south = max(-83.0, lat - deg_lat)
    north = min(83.0, lat + deg_lat)
    west = max(-180.0, lon - deg_lon)
    east = min(180.0, lon + deg_lon)
    return west, south, east, north

# =============================================================================
# Earth Engine / ASTER GDEM v3
# =============================================================================

def _json_from_streamlit_secret(value):
    """Convierte secrets de Streamlit a dict serializable."""
    if value is None:
        return None
    try:
        # Streamlit puede entregar AttrDict.
        return dict(value)
    except Exception:
        pass
    if isinstance(value, str):
        return json.loads(value)
    return value


def initialize_earth_engine(service_account_json_file=None):
    """Inicializa Google Earth Engine para descargar ASTER GDEM v3."""
    if ee is None:
        raise RuntimeError("No se pudo importar earthengine-api. Revise requirements.txt y el log de instalación.")

    # 1) JSON subido por interfaz.
    if service_account_json_file is not None:
        service_account_json_file.seek(0)
        info = json.loads(service_account_json_file.read().decode("utf-8"))
        service_account = info.get("client_email")
        project = info.get("project_id")
        if not service_account:
            raise ValueError("El JSON de cuenta de servicio no contiene client_email.")
        credentials = ee.ServiceAccountCredentials(service_account, key_data=json.dumps(info))
        ee.Initialize(credentials, project=project)
        return "JSON de cuenta de servicio cargado desde la app."

    # 2) Formato recomendado: [earthengine] en secrets.
    try:
        if "earthengine" in st.secrets:
            info = _json_from_streamlit_secret(st.secrets["earthengine"])
            service_account = info.get("client_email") or info.get("EE_SERVICE_ACCOUNT")
            project = info.get("project_id") or info.get("EE_PROJECT")
            private_key = info.get("private_key") or info.get("EE_PRIVATE_KEY")
            if service_account and private_key:
                info.setdefault("type", "service_account")
                info.setdefault("client_email", service_account)
                info.setdefault("private_key", private_key.replace("\\n", "\n"))
                info.setdefault("token_uri", "https://oauth2.googleapis.com/token")
                credentials = ee.ServiceAccountCredentials(service_account, key_data=json.dumps(info))
                ee.Initialize(credentials, project=project)
                return "Credenciales Earth Engine desde [earthengine] en Secrets."
    except Exception as exc:
        raise RuntimeError(f"Secrets [earthengine] existen, pero no pudieron inicializar Earth Engine: {exc}") from exc

    # 3) Formato alternativo compatible con ejemplos Streamlit: [gcp_service_account].
    try:
        if "gcp_service_account" in st.secrets:
            info = _json_from_streamlit_secret(st.secrets["gcp_service_account"])
            service_account = info.get("client_email")
            project = info.get("project_id")
            if service_account:
                if "private_key" in info:
                    info["private_key"] = info["private_key"].replace("\\n", "\n")
                credentials = ee.ServiceAccountCredentials(service_account, key_data=json.dumps(info))
                ee.Initialize(credentials, project=project)
                return "Credenciales Earth Engine desde [gcp_service_account] en Secrets."
    except Exception as exc:
        raise RuntimeError(f"Secrets [gcp_service_account] existen, pero no pudieron inicializar Earth Engine: {exc}") from exc

    # 4) Formato plano.
    try:
        service_account = st.secrets.get("EE_SERVICE_ACCOUNT", "")
        private_key = st.secrets.get("EE_PRIVATE_KEY", "")
        project = st.secrets.get("EE_PROJECT", None)
        if service_account and private_key:
            key_data = {
                "type": "service_account",
                "client_email": service_account,
                "private_key": private_key.replace("\\n", "\n"),
                "token_uri": "https://oauth2.googleapis.com/token",
            }
            credentials = ee.ServiceAccountCredentials(service_account, key_data=json.dumps(key_data))
            ee.Initialize(credentials, project=project)
            return "Credenciales Earth Engine desde variables planas EE_*."
    except Exception as exc:
        raise RuntimeError(f"Secrets EE_* existen, pero no pudieron inicializar Earth Engine: {exc}") from exc

    # 5) Credenciales locales, útil si se ejecuta en PC con earthengine authenticate.
    try:
        ee.Initialize()
        return "Credenciales locales Earth Engine."
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine no está autenticado. Configure Secrets en Streamlit Cloud o suba un JSON de cuenta de servicio habilitada en Earth Engine."
        ) from exc


def download_aster_gdem_gee(lon, lat, radius_km, out_dir, service_account_json_file=None):
    """Descarga ASTER GDEM v3 desde Earth Engine para un bbox alrededor del punto."""
    auth_msg = initialize_earth_engine(service_account_json_file)
    west, south, east, north = bbox_from_point_radius(lon, lat, radius_km)
    region = ee.Geometry.Rectangle([west, south, east, north], geodesic=False)
    image = ee.Image(ASTER_GEE_ASSET).rename("elevation")

    url = image.getDownloadURL({
        "region": region,
        "scale": 30,
        "crs": "EPSG:4326",
        "format": "GEO_TIFF",
        "filePerBand": False,
    })

    r = requests.get(url, timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f"Earth Engine respondió {r.status_code}: {r.text[:800]}")

    out_dir = Path(out_dir)
    out_tif = out_dir / "aster_gdem_v3.tif"

    if r.content[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(r.content), "r") as z:
            tif_names = [n for n in z.namelist() if n.lower().endswith((".tif", ".tiff"))]
            if not tif_names:
                raise RuntimeError("La descarga de Earth Engine no contiene GeoTIFF.")
            with z.open(tif_names[0]) as src, open(out_tif, "wb") as dst:
                dst.write(src.read())
    else:
        with open(out_tif, "wb") as f:
            f.write(r.content)

    with rasterio.open(out_tif) as src:
        if src.crs is None:
            raise RuntimeError("El GeoTIFF ASTER descargado no tiene CRS.")
        if src.width < 10 or src.height < 10:
            raise RuntimeError("La descarga ASTER es demasiado pequeña; aumente el radio.")

    return out_tif, auth_msg, (west, south, east, north)

# =============================================================================
# OpenTopography fallback: NO ASTER, sólo DEMs del API Global.
# =============================================================================

def download_opentopography_dem(lon, lat, radius_km, demtype, api_key, out_path):
    west, south, east, north = bbox_from_point_radius(lon, lat, radius_km)
    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": demtype,
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }
    r = requests.get(url, params=params, timeout=240)
    if r.status_code != 200:
        raise RuntimeError(f"OpenTopography respondió {r.status_code}: {r.text[:800]}")
    ctype = r.headers.get("content-type", "").lower()
    if "html" in ctype or r.content[:50].lower().startswith(b"<html"):
        raise RuntimeError(r.text[:800])
    with open(out_path, "wb") as f:
        f.write(r.content)
    with rasterio.open(out_path) as src:
        if src.crs is None:
            raise RuntimeError("El DEM descargado desde OpenTopography no tiene CRS.")
    return out_path, (west, south, east, north)

# =============================================================================
# Hidrología: D8 con pysheds
# =============================================================================

def _largest_polygon(geom):
    if geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type == "MultiPolygon":
        return max(list(geom.geoms), key=lambda g: g.area)
    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if polys:
            merged = unary_union(polys)
            return _largest_polygon(merged)
    return geom


def delineate_watershed(dem_path, lon, lat, acc_threshold_cells=200):
    if Grid is None:
        raise RuntimeError("No se pudo importar pysheds. Revise requirements.txt y el log de instalación.")

    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        if dem_crs is None:
            raise ValueError("El DEM no tiene sistema de referencia definido.")
        to_dem = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
        x_dem, y_dem = to_dem.transform(lon, lat)
        bounds = src.bounds
        if not (bounds.left <= x_dem <= bounds.right and bounds.bottom <= y_dem <= bounds.top):
            raise ValueError("El punto de control está fuera de la extensión del DEM.")

    grid = Grid.from_raster(str(dem_path))
    dem = grid.read_raster(str(dem_path))

    # Corrección básica del DEM para flujo superficial.
    pit_filled = grid.fill_pits(dem)
    flooded = grid.fill_depressions(pit_filled)
    inflated = grid.resolve_flats(flooded)

    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    fdir = grid.flowdir(inflated, dirmap=dirmap)
    acc = grid.accumulation(fdir, dirmap=dirmap)

    # Ajuste del punto al drenaje cercano.
    snap_mask = acc > int(acc_threshold_cells)
    try:
        x_snap, y_snap = grid.snap_to_mask(snap_mask, (x_dem, y_dem))
    except Exception:
        x_snap, y_snap = x_dem, y_dem

    catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap, xytype="coordinate")
    catch_full = grid.view(catch, dtype=np.uint8)
    touches_dem_edge = bool(
        catch_full[0, :].any()
        or catch_full[-1, :].any()
        or catch_full[:, 0].any()
        or catch_full[:, -1].any()
    )

    # Recorte sólo para vectorizar y reducir tamaño.
    grid.clip_to(catch)
    catch_view = grid.view(catch, dtype=np.uint8)
    affine = grid.affine

    catch_polys = []
    for geom, value in shapes(catch_view.astype(np.uint8), mask=catch_view.astype(bool), transform=affine):
        if int(value) == 1:
            catch_polys.append(shape(geom))

    if not catch_polys:
        raise RuntimeError("No se pudo generar polígono de cuenca. Revise DEM, punto y umbral de acumulación.")

    watershed_dem_crs = unary_union(catch_polys).buffer(0)
    watershed_dem_crs = _largest_polygon(watershed_dem_crs).buffer(0)

    watershed_wgs84 = project_geom(watershed_dem_crs, dem_crs, CRS.from_epsg(4326)).buffer(0)
    watershed_wgs84 = _largest_polygon(watershed_wgs84).buffer(0)
    snap_wgs84 = project_geom(Point(x_snap, y_snap), dem_crs, CRS.from_epsg(4326))

    return {
        "polygon_dem_crs": watershed_dem_crs,
        "polygon_wgs84": watershed_wgs84,
        "snap_point_wgs84": snap_wgs84,
        "dem_crs": dem_crs,
        "touches_edge": touches_dem_edge,
    }

# =============================================================================
# Curvas de nivel
# =============================================================================

def iter_lines(geom):
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, LineString):
        yield geom
    elif isinstance(geom, MultiLineString):
        for g in geom.geoms:
            yield g
    elif isinstance(geom, GeometryCollection):
        for g in geom.geoms:
            yield from iter_lines(g)


def generate_contours(dem_path, watershed_dem_crs, interval_m, max_cells_for_contours=3_000_000):
    with rasterio.open(dem_path) as src:
        out_image, out_transform = mask(src, [mapping(watershed_dem_crs)], crop=True, filled=False)
        z = out_image[0]
        dem_crs = src.crs
        nodata = src.nodata

    z_arr = z.filled(np.nan).astype(float) if np.ma.is_masked(z) else z.astype(float)
    if nodata is not None:
        z_arr[z_arr == nodata] = np.nan
    z_arr[z_arr <= -9990] = np.nan

    rows0, cols0 = z_arr.shape
    stride = max(1, int(math.ceil(math.sqrt((rows0 * cols0) / max_cells_for_contours))))
    if stride > 1:
        z_arr_small = z_arr[::stride, ::stride]
    else:
        z_arr_small = z_arr

    rows, cols = z_arr_small.shape
    finite = np.isfinite(z_arr_small)
    if finite.sum() < 10:
        raise RuntimeError("No hay suficientes celdas válidas de elevación dentro de la cuenca.")

    z_min = float(np.nanmin(z_arr_small))
    z_max = float(np.nanmax(z_arr_small))
    start = math.ceil(z_min / interval_m) * interval_m
    end = math.floor(z_max / interval_m) * interval_m
    if end < start:
        raise ValueError("La equidistancia de curvas es mayor que el rango altimétrico de la cuenca.")
    levels = np.arange(start, end + interval_m, interval_m, dtype=float)

    col_idx = np.arange(cols) * stride
    row_idx = np.arange(rows) * stride
    xs = np.array([xy(out_transform, 0, int(c), offset="center")[0] for c in col_idx])
    ys = np.array([xy(out_transform, int(r), 0, offset="center")[1] for r in row_idx])
    X, Y = np.meshgrid(xs, ys)

    fig, ax = plt.subplots(figsize=(8, 6))
    cs = ax.contour(X, Y, z_arr_small, levels=levels)
    plt.close(fig)

    contour_wgs84 = []
    for level, segs in zip(cs.levels, cs.allsegs):
        for seg in segs:
            if len(seg) < 2:
                continue
            line = LineString(seg)
            if not line.is_valid or line.length == 0:
                continue
            clipped = line.intersection(watershed_dem_crs)
            for ln in iter_lines(clipped):
                if ln.length > 0:
                    geom_wgs84 = project_geom(ln, dem_crs, CRS.from_epsg(4326))
                    contour_wgs84.append({"elev_m": float(level), "geometry_wgs84": geom_wgs84})

    return contour_wgs84, levels, stride

# =============================================================================
# KMZ
# =============================================================================

def exterior_coords_wgs84(poly):
    return [(float(x), float(y), 0) for x, y in poly.exterior.coords]


def interior_coords_wgs84(poly):
    return [[(float(x), float(y), 0) for x, y in ring.coords] for ring in poly.interiors]


def add_polygon_to_kml(folder, geom, name, description, style):
    polygons = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    for i, poly in enumerate(polygons, start=1):
        p = folder.newpolygon(
            name=name if len(polygons) == 1 else f"{name} parte {i}",
            outerboundaryis=exterior_coords_wgs84(poly),
            innerboundaryis=interior_coords_wgs84(poly),
            description=description,
        )
        p.style = style


def add_line_to_kml(folder, geom, name, style):
    for ln in iter_lines(geom):
        coords = [(float(x), float(y), 0) for x, y in ln.coords]
        if len(coords) >= 2:
            kline = folder.newlinestring(name=name, coords=coords)
            kline.style = style


def create_result_kmz(out_path, lon, lat, snap_point, watershed_wgs84, area_km2, area_ha, contours, dem_label):
    kml = simplekml.Kml(name="Cuenca delimitada")

    sty_poly = simplekml.Style()
    sty_poly.polystyle.color = simplekml.Color.changealphaint(70, simplekml.Color.blue)
    sty_poly.linestyle.color = simplekml.Color.blue
    sty_poly.linestyle.width = 3

    sty_point = simplekml.Style()
    sty_point.iconstyle.color = simplekml.Color.red
    sty_point.iconstyle.scale = 1.1

    sty_snap = simplekml.Style()
    sty_snap.iconstyle.color = simplekml.Color.green
    sty_snap.iconstyle.scale = 1.0

    sty_contour = simplekml.Style()
    sty_contour.linestyle.color = simplekml.Color.rgb(120, 80, 40, 255)
    sty_contour.linestyle.width = 1.2

    f_main = kml.newfolder(name="Cuenca")
    desc = (
        f"DEM utilizado: {dem_label}<br>"
        f"Superficie cuenca: {area_km2:,.3f} km²<br>"
        f"Superficie cuenca: {area_ha:,.2f} ha<br>"
        f"Punto original: lon {lon:.7f}, lat {lat:.7f}<br>"
        f"Punto ajustado a drenaje: lon {snap_point.x:.7f}, lat {snap_point.y:.7f}"
    )
    add_polygon_to_kml(f_main, watershed_wgs84, f"Cuenca - {area_km2:.3f} km²", desc, sty_poly)

    p0 = f_main.newpoint(name="Punto de control original", coords=[(lon, lat, 0)])
    p0.style = sty_point
    ps = f_main.newpoint(name="Punto ajustado al drenaje", coords=[(snap_point.x, snap_point.y, 0)])
    ps.style = sty_snap

    f_contours = kml.newfolder(name="Curvas de nivel")
    for rec in contours:
        add_line_to_kml(f_contours, rec["geometry_wgs84"], f"Curva {rec['elev_m']:.0f} m", sty_contour)

    kml.savekmz(out_path)
    return out_path

# =============================================================================
# Vista previa
# =============================================================================

def preview_plot(dem_path, watershed_dem_crs, contours, dem_crs, lon, lat):
    with rasterio.open(dem_path) as src:
        out_image, out_transform = mask(src, [mapping(watershed_dem_crs)], crop=True, filled=False)
        z = out_image[0]
        nodata = src.nodata
    z_arr = z.filled(np.nan).astype(float) if np.ma.is_masked(z) else z.astype(float)
    if nodata is not None:
        z_arr[z_arr == nodata] = np.nan

    rows0, cols0 = z_arr.shape
    max_cells = 1_200_000
    stride = max(1, int(math.ceil(math.sqrt((rows0 * cols0) / max_cells))))
    z_small = z_arr[::stride, ::stride]
    rows, cols = z_small.shape
    col_idx = np.arange(cols) * stride
    row_idx = np.arange(rows) * stride
    xs = np.array([xy(out_transform, 0, int(c), offset="center")[0] for c in col_idx])
    ys = np.array([xy(out_transform, int(r), 0, offset="center")[1] for r in row_idx])

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(z_small, extent=[xs.min(), xs.max(), ys.min(), ys.max()], origin="upper")

    for ln in iter_lines(watershed_dem_crs.boundary):
        x, y = ln.xy
        ax.plot(x, y, linewidth=2)

    to_dem = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
    x0, y0 = to_dem.transform(lon, lat)
    ax.scatter([x0], [y0], s=45)

    transformer = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
    count = 0
    for rec in contours:
        if count > 800:
            break
        geom_d = shp_transform(transformer.transform, rec["geometry_wgs84"])
        for ln in iter_lines(geom_d):
            x, y = ln.xy
            ax.plot(x, y, linewidth=0.45)
            count += 1

    ax.set_title("Vista preliminar DEM + cuenca + curvas de nivel")
    ax.set_xlabel(f"X ({dem_crs})")
    ax.set_ylabel(f"Y ({dem_crs})")
    ax.set_aspect("equal", adjustable="box")
    return fig

# =============================================================================
# Interfaz
# =============================================================================

st.title("Delineación automática de cuenca desde KMZ con ASTER GDEM v3")
st.caption("KMZ/KML con punto de control → DEM ASTER/GeoTIFF → cuenca, superficie y curvas de nivel → KMZ final.")

with st.expander("Diagnóstico de esta versión corregida", expanded=True):
    st.markdown(
        """
        Esta versión corrige dos problemas frecuentes:
        
        1. El archivo principal queda en la **raíz del repositorio**, por lo que en Streamlit Cloud el **Main file path es `app.py`**.
        2. ASTER GDEM v3 se trabaja mediante **Google Earth Engine** o mediante **DEM GeoTIFF ASTER cargado manualmente**. Si no se configuran credenciales Earth Engine, la descarga automática no puede ejecutarse.
        """
    )

left, right = st.columns([0.38, 0.62])

with left:
    st.subheader("1. Entrada")
    kmz_file = st.file_uploader("KMZ/KML con punto de control de la cuenca", type=["kmz", "kml"])

    dem_source = st.radio(
        "Fuente del DEM",
        [
            "ASTER GDEM v3 automático - Earth Engine",
            "ASTER GDEM v3 manual - GeoTIFF",
            "OpenTopography alternativo - no ASTER",
        ],
        index=0,
    )

    radius_km = 35.0
    ee_json_file = None
    dem_file = None
    api_key = ""
    demtype = "COP30"
    auth_mode = "Secrets/local"

    if dem_source == "ASTER GDEM v3 automático - Earth Engine":
        st.info(f"Asset usado: `{ASTER_GEE_ASSET}`")
        radius_km = st.number_input(
            "Radio inicial de descarga DEM alrededor del punto (km)",
            min_value=5.0,
            max_value=100.0,
            value=35.0,
            step=5.0,
            help="Debe cubrir toda la cuenca. Si el resultado toca el borde, aumente el radio o use DEM GeoTIFF manual.",
        )
        auth_mode = st.selectbox(
            "Autenticación Earth Engine",
            ["Secrets/local", "Subir JSON de cuenta de servicio"],
            index=0,
        )
        if auth_mode == "Subir JSON de cuenta de servicio":
            ee_json_file = st.file_uploader("JSON de cuenta de servicio Earth Engine", type=["json"])

    elif dem_source == "ASTER GDEM v3 manual - GeoTIFF":
        st.info("Use un GeoTIFF ASTER GDEM v3 descargado previamente. Esta opción no requiere credenciales Earth Engine.")
        dem_file = st.file_uploader("DEM ASTER GeoTIFF", type=["tif", "tiff"])

    else:
        st.warning("OpenTopography se deja como respaldo. Su API Global no usa ASTER GDEM v3; use COP30/NASADEM/SRTM/AW3D30.")
        api_key = st.text_input("OpenTopography API Key", type="password")
        demtype = st.selectbox("DEM OpenTopography", ["COP30", "NASADEM", "SRTMGL1", "SRTMGL3", "COP90", "AW3D30"], index=0)
        radius_km = st.number_input("Radio de descarga DEM alrededor del punto (km)", min_value=5.0, max_value=200.0, value=35.0, step=5.0)

    st.subheader("2. Parámetros")
    contour_interval = st.number_input("Equidistancia curvas de nivel (m)", min_value=1.0, max_value=500.0, value=25.0, step=1.0)
    acc_threshold = st.number_input(
        "Umbral de acumulación para ajustar punto al cauce (celdas)",
        min_value=1,
        max_value=100000,
        value=200,
        step=50,
        help="Si el punto se ajusta mal, pruebe 50, 100, 200, 500 o 1000.",
    )

    run = st.button("Generar cuenca y KMZ", type="primary")

with right:
    st.subheader("Resultado")
    if not run:
        st.info("Cargue el KMZ/KML, seleccione la fuente DEM y presione **Generar cuenca y KMZ**.")
    else:
        if kmz_file is None:
            st.error("Debe cargar un KMZ/KML con el punto de control.")
            st.stop()
        if dem_source == "ASTER GDEM v3 automático - Earth Engine" and auth_mode == "Subir JSON de cuenta de servicio" and ee_json_file is None:
            st.error("Debe subir el JSON de cuenta de servicio o usar Secrets/local.")
            st.stop()
        if dem_source == "ASTER GDEM v3 manual - GeoTIFF" and dem_file is None:
            st.error("Debe cargar un GeoTIFF ASTER GDEM v3.")
            st.stop()
        if dem_source == "OpenTopography alternativo - no ASTER" and not api_key:
            st.error("Debe ingresar API Key de OpenTopography.")
            st.stop()

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            try:
                lon, lat = read_point_from_kmz(kmz_file)
                st.success(f"Punto leído: lon {lon:.7f}, lat {lat:.7f}")

                dem_label = ""
                bbox = None
                dem_path = tmpdir / "dem.tif"

                if dem_source == "ASTER GDEM v3 automático - Earth Engine":
                    with st.spinner("Descargando ASTER GDEM v3 desde Earth Engine..."):
                        dem_path, auth_msg, bbox = download_aster_gdem_gee(lon, lat, radius_km, tmpdir, ee_json_file)
                    dem_label = f"ASTER GDEM v3 / {ASTER_GEE_ASSET}"
                    st.caption(f"Autenticación: {auth_msg}")

                elif dem_source == "ASTER GDEM v3 manual - GeoTIFF":
                    dem_file.seek(0)
                    with open(dem_path, "wb") as f:
                        f.write(dem_file.read())
                    dem_label = "ASTER GDEM v3 GeoTIFF manual"

                else:
                    with st.spinner("Descargando DEM desde OpenTopography..."):
                        dem_path, bbox = download_opentopography_dem(lon, lat, radius_km, demtype, api_key, dem_path)
                    dem_label = f"OpenTopography {demtype}"

                with rasterio.open(dem_path) as src:
                    st.caption(f"DEM: {src.width:,} x {src.height:,} celdas | CRS: {src.crs}")

                with st.spinner("Procesando DEM y delineando cuenca..."):
                    result = delineate_watershed(dem_path, lon, lat, acc_threshold)

                watershed_wgs84 = result["polygon_wgs84"]
                watershed_dem_crs = result["polygon_dem_crs"]
                area_km2, area_ha = polygon_area_km2_ha(watershed_wgs84)

                with st.spinner("Generando curvas de nivel..."):
                    contours, levels, stride = generate_contours(dem_path, watershed_dem_crs, contour_interval)

                out_kmz = tmpdir / "cuenca_curvas_nivel.kmz"
                create_result_kmz(out_kmz, lon, lat, result["snap_point_wgs84"], watershed_wgs84, area_km2, area_ha, contours, dem_label)

                c1, c2, c3 = st.columns(3)
                c1.metric("Superficie", f"{area_km2:,.3f} km²")
                c2.metric("Superficie", f"{area_ha:,.2f} ha")
                c3.metric("Curvas", f"{len(contours):,}")

                if result["touches_edge"]:
                    st.warning("La cuenca toca el borde del DEM. Aumente el radio de descarga o cargue un DEM más amplio para evitar truncamiento.")

                rows = [
                    {"Concepto": "DEM utilizado", "Valor": dem_label},
                    {"Concepto": "Longitud punto original", "Valor": f"{lon:.7f}"},
                    {"Concepto": "Latitud punto original", "Valor": f"{lat:.7f}"},
                    {"Concepto": "Longitud punto ajustado", "Valor": f"{result['snap_point_wgs84'].x:.7f}"},
                    {"Concepto": "Latitud punto ajustado", "Valor": f"{result['snap_point_wgs84'].y:.7f}"},
                    {"Concepto": "Superficie km²", "Valor": f"{area_km2:.6f}"},
                    {"Concepto": "Superficie ha", "Valor": f"{area_ha:.2f}"},
                    {"Concepto": "Equidistancia curvas m", "Valor": f"{contour_interval:.1f}"},
                    {"Concepto": "Rango curvas m", "Valor": f"{float(levels.min()):.0f} - {float(levels.max()):.0f}"},
                    {"Concepto": "Submuestreo curvas", "Valor": f"1 cada {stride} celda(s)"},
                ]
                if bbox:
                    rows.append({"Concepto": "BBOX DEM", "Valor": f"W {bbox[0]:.5f}, S {bbox[1]:.5f}, E {bbox[2]:.5f}, N {bbox[3]:.5f}"})
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

                fig = preview_plot(dem_path, watershed_dem_crs, contours, result["dem_crs"], lon, lat)
                st.pyplot(fig, clear_figure=True)

                with open(out_kmz, "rb") as f:
                    st.download_button(
                        "Descargar KMZ cuenca + curvas de nivel",
                        data=f.read(),
                        file_name="cuenca_curvas_nivel.kmz",
                        mime="application/vnd.google-earth.kmz",
                    )

            except Exception as exc:
                st.error(f"No se pudo completar el procesamiento: {exc}")
                st.exception(exc)
