#!/bin/bash
# Paper 2 (jet en capas bajas): reproduce resultados, figuras y el PDF del borrador a partir de los datos descargados.
# Descargas (lentas, una sola vez): IGRA v2 en data/igra_sa/, ERA5 con analisis/jet/02_descarga_era5_perfiles.py,
# CHIRPS diario con analisis/jet/18a_descarga_chirps_diario.sh, NOAA ISD en data/isd_full/ (ver DATOS.md).
# Uso: ./REPRODUCIR_JET.sh  |  ./REPRODUCIR_JET.sh desde 12
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=analisis
export SSL_CERT_FILE="$(.venv/bin/python -m certifi)"
PY=.venv/bin/python
mkdir -p logs/jet figuras/jet
DESDE="${2:-}"
activo=$([ -z "$DESDE" ] && echo 1 || echo 0)

paso() {   # paso <nombre> <comando...>
  local nombre="$1"; shift
  if [ "$activo" = 0 ] && [[ "$nombre" == "$DESDE"* ]]; then activo=1; fi
  [ "$activo" = 1 ] || return 0
  printf '%s  %-34s' "$(date +%H:%M)" "$nombre"
  if env "$@" > "logs/jet/$nombre.log" 2>&1; then echo "ok"; else echo "FALLÓ (ver logs/jet/$nombre.log)"; exit 1; fi
}

echo "== Sondeos: perfiles, climatología, homogeneidad"
paso 01_perfiles            $PY analisis/jet/01_perfiles_igra.py
paso 03_climatologia        $PY analisis/jet/03_climatologia_observada.py
paso 04_resolucion          $PY analisis/jet/04_mecanismo_resolucion.py
paso 05_saltos_925          $PY analisis/jet/05_saltos_925.py
paso 06_viento_superficie   $PY analisis/jet/06_viento_superficie.py
paso 07_compuestos          $PY analisis/jet/07_compuestos.py
paso 08_hora                $PY analisis/jet/08_hora_lanzamiento.py
paso 03b_tendencias_est   $PY analisis/jet/03b_tendencias_estaciones.py
paso 09_fuente              $PY analisis/jet/09_por_fuente.py
paso 10_modelo_fuente       $PY analisis/jet/10_modelo_fuente.py
echo "== ERA5 y homogeneización"
paso 12_era5_vs_sondeos     $PY analisis/jet/12_era5_vs_sondeos.py
paso 22_homogeneizar        $PY analisis/jet/22_homogeneizar_fijos.py
paso 13_departures          $PY analisis/jet/13_departures_era5.py
paso 19_emulacion           $PY analisis/jet/19_emulacion_resolucion.py
paso 24_perfil_ajustado     $PY analisis/jet/24_perfil_ajustado.py
paso 34_eras_did            $PY analisis/jet/34_eras_y_did.py
paso 35_evaluabilidad       $PY analisis/jet/35_evaluabilidad.py
paso 36_ventanas_escalon     $PY analisis/jet/36_ventanas_y_escalon.py
paso 31_ruido_emulado       $PY analisis/jet/31_ruido_emulado.py
paso 32_equivalencia        $PY analisis/jet/32_equivalencia.py
paso 33_seleccion_925       $PY analisis/jet/33_seleccion_925.py
paso 15_ciclo_diario        $PY analisis/jet/15_ciclo_diario.py
paso 20_habilidad           $PY analisis/jet/20_habilidad_temporal.py
paso 23_habilidad_robusta   $PY analisis/jet/23_habilidad_robusta.py
paso 25_estadistica         $PY analisis/jet/25_estadistica_robusta.py
paso 26_geostrofico         $PY analisis/jet/26_geostrofico.py
paso 21_rama_norte          $PY analisis/jet/21_rama_norte.py
echo "== Física de las clases"
paso 28_tabla_estaciones     $PY analisis/jet/28_tabla_estaciones.py
paso 14_clases              $PY analisis/jet/14_dinamica_clases.py
paso 16_sinoptica           $PY analisis/jet/16_evolucion_sinoptica.py
paso 17_era5_nocturno       $PY analisis/jet/17_era5_nocturno.py
paso 18_lluvia              $PY analisis/jet/18_lluvia_clases.py
paso 27_clases_altura       $PY analisis/jet/27_clases_altura.py
echo "== Figuras y PDF"
paso figuras                $PY analisis/jet/figuras_jet.py
paso 37_resumen_grafico     $PY analisis/jet/37_resumen_grafico.py
paso 30_referencias         $PY analisis/jet/30_referencias.py
paso 29_ensamblar           $PY analisis/jet/29_ensamblar.py
paso pdf                    bash -c '.venv/bin/python analisis/borrador_html.py borrador_jet && .venv/bin/python analisis/borrador_html.py suplemento_jet && mv paper/borrador_jet.pdf paper/suplemento_jet.pdf paper_jet/ && rm paper/borrador_jet.md paper/suplemento_jet.md'
echo "Listo: resúmenes en analisis/jet/*_resumen.txt, figuras en figuras/jet/, PDF en paper_jet/borrador_jet.pdf"
