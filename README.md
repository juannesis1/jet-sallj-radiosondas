# Has the South American low-level jet changed? Radiosonde evidence and the limits set by the observing system, 1980–2025

Analysis code, intermediate results, figures and manuscript sources for the paper of the same title (Nesis, J.; submitted to the
International Journal of Climatology).

The paper asks whether the frequency of the South American low-level jet has changed in 1980–2025, using IGRA v2 radiosondes
(31 stations), ERA5, CHIRPS and surface observations, and how much of an apparent change comes from archive sources, wind-measurement
breaks, vertical resolution and sounding noise.

## Reproducing the results
1. Download the public data listed in `DATOS.md` (not included because of their size).
2. Create the environment: `python -m venv .venv && .venv/bin/pip install -r requirements.txt`.
3. Run `./REPRODUCIR_JET.sh` (steps 01–37; `./REPRODUCIR_JET.sh desde 22` restarts from a step). It regenerates every number
   (`analisis/jet/*_resumen.txt`), the figures (`figuras/jet/`) and the manuscript PDFs (`paper_jet/`).

Each step is a script in `analisis/jet/` whose docstring states what it computes; the summaries committed here are the outputs of the
run used in the paper.

## Contents
- `analisis/jet/`: scripts 01–37 and their outputs (`*_resumen.txt`, CSV, parquet).
- `analisis/estadistica.py`, `analisis/figuras_paper.py`, `analisis/borrador_html.py`: shared helpers.
- `paper_jet/`: manuscript source (`borrador.md`), PDFs, references and the submission package.
- `figuras/jet/`: figures.

## License and citation
Archived release: https://doi.org/10.5281/zenodo.23196720

Code under the MIT license (`LICENSE`). Please cite the archived release (see `CITATION.cff`).
