# Aplicación: Cuenca desde KMZ con ASTER GDEM v3 y curvas de nivel

Aplicación Streamlit para cargar un KMZ/KML con un punto de control o punto final de una cuenca, procesar un DEM, delinear la cuenca aportante, calcular superficie y generar un KMZ con:

- Polígono de cuenca.
- Superficie en km² y ha.
- Punto original.
- Punto ajustado al drenaje.
- Curvas de nivel con equidistancia ingresada manualmente.

## Main file path

En Streamlit Cloud use exactamente:

```text
app.py
```

Esta versión deja `app.py` en la raíz del repositorio.

## Fuente DEM principal

La opción principal usa ASTER GDEM v3 mediante Google Earth Engine:

```text
projects/sat-io/open-datasets/ASTER/GDEM
```

Para que funcione en Streamlit Cloud debe configurar credenciales de Earth Engine en `Settings > Secrets`, o subir un JSON de cuenta de servicio desde la interfaz de la app.

## Opción sin credenciales

Use la opción:

```text
ASTER GDEM v3 manual - GeoTIFF
```

Debe cargar un archivo `.tif` o `.tiff` ASTER GDEM v3 descargado previamente. Esta modalidad no requiere Earth Engine.

## Secrets de Streamlit Cloud

Copie el contenido de `.streamlit/secrets.toml.example` en `Settings > Secrets` de Streamlit Cloud y reemplace por los datos reales de su cuenta de servicio.

No suba el archivo `secrets.toml` real a GitHub.

## Uso

1. Cargar KMZ/KML con punto de control.
2. Seleccionar fuente DEM.
3. Ingresar equidistancia de curvas de nivel.
4. Ajustar umbral de acumulación si el punto no cae bien sobre el cauce.
5. Presionar **Generar cuenca y KMZ**.
6. Descargar el KMZ final.

## Notas técnicas

- Si la cuenca toca el borde del DEM, aumente el radio o cargue un DEM manual más amplio.
- ASTER GDEM v3 es adecuado para análisis preliminar regional. Para diseño definitivo, verificar con topografía local, cauces observados y cartografía oficial.
- OpenTopography se deja como respaldo con DEMs compatibles con su Global DEM API, pero no como ASTER GDEM v3 automático.
