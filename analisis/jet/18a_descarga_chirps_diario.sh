#!/bin/sh
# Paper 2, paso 18a: CHIRPS v2.0 diario 0.25° (Funk et al. 2015), 40-20°S, 65-50°W, oct-mar 1997/98-2024/25,
# recortado en el IRI Data Library. Un archivo por temporada en data/chirps_diario/.
cd "$(dirname "$0")/../.." || exit 1
for y in $(seq 1997 2024); do
  f="data/chirps_diario/chirps_${y}-$((y+1)).nc"
  [ -s "$f" ] && continue
  curl -s -m 600 --retry 5 -o "$f.tmp" "https://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p25/.prcp/X/-65/-50/RANGEEDGES/Y/-40/-20/RANGEEDGES/T/(1%20Oct%20${y})/(31%20Mar%20$((y+1)))/RANGEEDGES/data.nc" && mv "$f.tmp" "$f" && echo "ok $f"
done
