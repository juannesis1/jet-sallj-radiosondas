"""Paper 2, paso 30: lista de referencias con formato uniforme y autores completos, armada desde Crossref a partir de los DOI.

Cada DOI se consulta en la API de Crossref (título, autores, revista, volumen, páginas o número de artículo, año); el resultado se guarda en
analisis/jet/30_referencias.json (caché, para no depender de la red al regenerar) y en paper_jet/referencias.md.
Uso: python analisis/jet/30_referencias.py
"""
import json
import os
import re
import urllib.parse
import urllib.request

DOIS = [
    "10.1002/joc.3370060607", "10.1029/2004JD004536", "10.1175/1525-7541(2002)003<0630:THCOTL>2.0.CO;2",
    "10.1175/1520-0477-38.5.283", "10.1175/1520-0493(1968)096<0833:COTLLJ>2.0.CO;2", "10.1175/JCLI3594.1", "10.1175/2008BAMS2603.1",
    "10.1038/sdata.2015.66", "10.1127/0941-2948/2008/0298", "10.1175/JCLI4050.1", "10.1175/JCLI-D-11-00668.1",
    "10.1016/S0022-1694(97)00125-X", "10.1002/qj.3803", "10.1111/j.2153-3490.1967.tb01473.x", "10.1038/s41612-019-0077-5",
    "10.1038/s41612-023-00501-4", "10.1175/1520-0442(2004)017<2261:COTLJE>2.0.CO;2", "10.1175/2008JCLI2263.1", "10.1029/2018JD029634",
    "10.1038/s41612-024-00852-6", "10.1175/MWR-D-17-0237.1", "10.1029/2001JD001315", "10.1175/MWR3305.1", "10.1002/jame.20013", "10.1175/MWR-D-21-0161.1", "10.1175/MWR3317.1",
    "10.1175/1520-0493(2003)131<2361:TNALAS>2.0.CO;2", "10.1175/BAMS-87-1-63", "10.1175/BAMS-D-15-00267.1",
    "10.1111/j.2517-6161.1995.tb02031.x", "10.1080/01621459.1968.10480934", "10.2307/1907187", "10.1007/BF01068419", "10.1177/1948550617697177",
    "10.1175/1520-0477(1997)078<1069:APICOW>2.0.CO;2", "10.1175/1520-0442(2003)016<4134:TITSAM>2.0.CO;2",
]
EXTRA = ["Omega Navigation System (VLF hyperbolic radio-navigation, closed 30 September 1997). jproc.ca/hyperbolic/omega.html (accessed 6 October 2026)."]


def consulta(doi):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/:()<>;")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "paper-jet (mailto:juannesiss@gmail.com)"}), timeout=30) as r:
        return json.load(r)["message"]


def formato(m):
    aut = []
    for a in m.get("author", []):
        ini = " ".join(f"{p[0]}." for p in re.split(r"[\s-]+", a.get("given", "")) if p)
        fam = a.get("family", "")
        fam = fam.title() if fam.isupper() else fam
        aut.append(f"{fam}, {ini}")
    autores = aut[0] if len(aut) == 1 else ", ".join(aut[:-1]) + ", and " + aut[-1]
    año = m["issued"]["date-parts"][0][0]
    titulo = re.sub(r"\s+", " ", m["title"][0]).rstrip(".")
    titulo = titulo.capitalize() if titulo.isupper() else titulo
    rev = m["container-title"][0] if m.get("container-title") else ""
    vol = m.get("volume", "")
    pag = m.get("page") or m.get("article-number", "")
    pag = pag.replace("-", "–") if pag else ""
    resto = f"*{rev}*" + (f", **{vol}**" if vol else "") + (f", {pag}" if pag else "")
    return f"{autores} ({año}). {titulo}. {resto}. https://doi.org/{m['DOI']}"


def main():
    ruta = "analisis/jet/30_referencias.json"
    cache = json.load(open(ruta)) if os.path.exists(ruta) else {}
    for d in DOIS:
        if d not in cache:
            cache[d] = consulta(d)
    json.dump(cache, open(ruta, "w"), ensure_ascii=False, indent=1)
    refs = [formato(cache[d]) for d in DOIS] + EXTRA
    refs = sorted(refs, key=lambda x: re.sub(r"[^a-z]", "", x.lower()))
    open("paper_jet/referencias.md", "w").write("\n\n".join(refs) + "\n")
    print(f"{len(refs)} referencias")
    print("\n".join(refs[:6]))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
