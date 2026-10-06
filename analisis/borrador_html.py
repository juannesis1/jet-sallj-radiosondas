"""Convierte paper/manuscrito.md (o el nombre dado como argumento, p. ej. suplemento) en PDF con las figuras incrustadas.
El HTML intermedio queda en paper/.html/ (oculto); el PDF en paper/."""
import sys
NOMBRE = sys.argv[1] if len(sys.argv) > 1 else "manuscrito"
import base64
import re

import os

import markdown

os.makedirs("paper/.html", exist_ok=True)
s = open(f"paper/{NOMBRE}.md").read()


def incrustar(m):
    b = base64.b64encode(open("paper/" + m.group(2), "rb").read()).decode()
    return f"![{m.group(1)}](data:image/png;base64,{b})"


s = re.sub(r"!\[([^\]]*)\]\((\.\./figuras/[^)]+)\)", incrustar, s)
css = ("body{font-family:Georgia,serif;max-width:820px;margin:2rem auto;padding:0 16px;line-height:1.55;"
       "color:#222;background:#fff}img{max-width:100%;margin:.5rem 0 1.5rem}h1{font-size:1.6rem}"
       "h2{margin-top:2.2rem;border-bottom:1px solid #ccc}h3{margin-top:1.6rem}")
html = markdown.markdown(s, extensions=["tables"])
open(f"paper/.html/{NOMBRE}.html", "w").write(
    '<!doctype html><html lang="en"><head><meta charset="utf-8">'
    f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{NOMBRE}</title>'
    f"<style>{css}</style></head><body>{html}</body></html>")
print("ok")

# PDF (Chrome sin interfaz): se abre en Vista Previa sin depender del navegador
import os
import subprocess

chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if os.path.exists(chrome):
    raiz = os.path.abspath("paper")
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={raiz}/{NOMBRE}.pdf", f"file://{raiz}/.html/{NOMBRE}.html"],
                   capture_output=True, timeout=180)
    print("pdf ok")
