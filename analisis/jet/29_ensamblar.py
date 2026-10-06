"""Paper 2, paso 29: ensambla el manuscrito: inserta la Tabla 1 (28_tabla_estaciones.csv) y las referencias (30_referencias.py) en
paper_jet/borrador.md y escribe paper/borrador_jet.md, que es lo que convierte borrador_html.py en PDF.
Marcadores del borrador: [[TABLA1]], [[TABLA2]] (quiebres, 22_quiebres.csv) y [[REFERENCIAS]].
"""
import os

import numpy as np
import pandas as pd


def main():
    t = pd.read_csv("analisis/jet/28_tabla_estaciones.csv").fillna("–")
    t.columns = ["IGRA ID", "Station", "Lat. (°)", "Lon. (°)", "Elev. (m)", "12-UTC record", "12-UTC soundings", "Levels 1980–95",
                 "Levels 2005–25", "Used in"]
    md = ["| " + " | ".join(t.columns) + " |", "|" + "|".join("---" for _ in t.columns) + "|"]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in t.itertuples(index=False)]
    q = pd.read_csv("analisis/jet/22_quiebres.csv")
    q["Decision"] = np.where(q.regional, "regional, not adjusted", "adjusted")
    q = q.sort_values(["nivel", "estacion", "fecha"], ascending=[False, True, True])
    t2 = pd.DataFrame({"Level (hPa)": q.nivel, "Station": q.estacion, "Date": q.fecha, "Step (m s⁻¹)": q.paso_m_s.round(2),
                       "Control (m s⁻¹)": q.control_m_s.round(2), "Control stations": q.n_control, "Decision": q.Decision})
    md2 = ["| " + " | ".join(t2.columns) + " |", "|" + "|".join("---" for _ in t2.columns) + "|"]
    md2 += ["| " + " | ".join(str(v) for v in r) + " |" for r in t2.itertuples(index=False)]
    s = open("paper_jet/borrador.md").read()
    s = s.replace("[[TABLA2]]", "\n".join(md2))
    s = s.replace("[[TABLA1]]", "\n".join(md))
    s = s.replace("[[REFERENCIAS]]", open("paper_jet/referencias.md").read().strip())
    # texto principal y material de apoyo (Supporting Information) por separado
    cut = s.index("\n## Supplementary Material")
    principal, apoyo = s[:cut].rstrip() + "\n", s[cut:]
    titulo = s.split("\n")[0].lstrip("# ").strip()
    apoyo = "# Supporting Information for: " + titulo + "\n\n**Juan Nesis**\n" + apoyo.replace("## Supplementary Material\n", "", 1)
    open("paper/borrador_jet.md", "w").write(principal)
    open("paper/suplemento_jet.md", "w").write(apoyo)
    print("ensamblado", len(principal.split()), "palabras (texto principal) y", len(apoyo.split()), "(material de apoyo)")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
