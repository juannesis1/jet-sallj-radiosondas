# Data sources

All data are public and are not included because of their size.

| Dataset | Period | Source | Local folder | How |
|---|---|---|---|---|
| IGRA v2 radiosondes (data and station list) | period of record, 31 stations | https://doi.org/10.7289/V5X63K0Q (https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/) | `data/igra_sa/` (`<station>-data.txt.zip`), `data/igra/igra2-station-list.txt` | download; the list of stations used is `data/igra_sa/estaciones.txt` (included) |
| ERA5 pressure levels (12 levels, 1000–700 hPa, 00 and 12 UTC, nearest grid point) and 850-hPa fields | Oct–Mar 1979–2025 | Copernicus Climate Data Store, https://doi.org/10.24381/cds.bd0915c6 (free account; the CDS API key stays in the user's `~/.cdsapirc`) | `data/era5/` | `analisis/jet/02_descarga_era5_perfiles.py` |
| CHIRPS v2.0 daily precipitation, 0.25° | Oct 1997–Mar 2025 | Climate Hazards Center via the IRI Data Library | `data/chirps/` | `analisis/jet/18a_descarga_chirps_diario.sh` |
| NOAA Integrated Surface Database (full hourly reports), four airports | 1980–2025 | https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database | `data/isd_full/` | one CSV per station and year |
| ONI | — | https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt | `data/indices/oni.txt` | text file |
| PDO (ERSST v5) | — | https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat | `data/indices/pdo_ersst.dat` | text file |
| SAM (Marshall, 2003) | — | https://legacy.bas.ac.uk/met/gjma/ | `data/indices/sam_marshall.txt` | text file |
