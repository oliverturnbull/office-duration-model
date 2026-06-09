"""Convert docs/survival_analysis.md to docs/survival_analysis.pdf.

Usage:
    python3 docs/build_pdf.py

Requires:
    pip install weasyprint markdown
"""

import pathlib
import sys

try:
    import markdown
    from weasyprint import HTML, CSS
except ImportError as exc:
    print(f"Error: {exc}")
    print("Install dependencies with:  pip install weasyprint markdown")
    sys.exit(1)

HERE = pathlib.Path(__file__).parent
MD_PATH = HERE / "survival_analysis.md"
PDF_PATH = HERE / "survival_analysis.pdf"

md_text = MD_PATH.read_text(encoding="utf-8")

extensions = ["tables", "fenced_code", "footnotes", "toc", "attr_list"]
body_html = markdown.markdown(md_text, extensions=extensions)

full_html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Survival Analysis Guide</title></head>
<body>{body_html}</body>
</html>"""

page_css = CSS(string="""
    @page {
        margin: 2.5cm;
        size: A4;
    }
    body {
        font-family: Georgia, "Liberation Serif", serif;
        font-size: 11pt;
        line-height: 1.65;
        color: #1a1a1a;
    }
    h1 {
        font-family: Arial, "Liberation Sans", sans-serif;
        font-size: 20pt;
        color: #1a2c45;
        border-bottom: 2px solid #1a2c45;
        padding-bottom: 6pt;
        margin-top: 0;
    }
    h2 {
        font-family: Arial, "Liberation Sans", sans-serif;
        font-size: 14pt;
        color: #1a2c45;
        border-bottom: 1px solid #ccc;
        padding-bottom: 3pt;
        margin-top: 28pt;
    }
    h3 {
        font-family: Arial, "Liberation Sans", sans-serif;
        font-size: 11.5pt;
        color: #2c3e50;
        margin-top: 18pt;
    }
    h4 {
        font-size: 10.5pt;
        color: #444;
        margin-top: 14pt;
    }
    code {
        font-family: "FreeMono", "DejaVu Sans Mono", "Courier New", monospace;
        font-size: 9pt;
        background: #f4f6f8;
        padding: 1pt 4pt;
        border-radius: 3pt;
        color: #2c3e50;
    }
    pre {
        background: #f4f6f8;
        border-left: 3px solid #1a2c45;
        padding: 10pt 14pt;
        margin: 10pt 0;
        page-break-inside: avoid;
    }
    pre code {
        background: none;
        padding: 0;
        font-size: 9pt;
        line-height: 1.5;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 12pt 0;
        font-size: 10pt;
        page-break-inside: avoid;
    }
    th {
        background: #1a2c45;
        color: white;
        padding: 5pt 8pt;
        text-align: left;
        font-family: Arial, sans-serif;
        font-weight: bold;
    }
    td {
        border: 1pt solid #ccc;
        padding: 5pt 8pt;
        vertical-align: top;
    }
    tr:nth-child(even) td {
        background: #f9f9f9;
    }
    blockquote {
        border-left: 3px solid #aaa;
        margin-left: 0;
        margin-right: 0;
        padding-left: 14pt;
        color: #555;
        font-style: italic;
    }
    hr {
        border: none;
        border-top: 1px solid #ddd;
        margin: 20pt 0;
    }
    p { margin: 8pt 0; }
    ul, ol { margin: 6pt 0; padding-left: 20pt; }
    li { margin: 3pt 0; }
    a { color: #1a5276; }
    em { font-style: italic; }
    strong { font-weight: bold; }
""")

HTML(string=full_html).write_pdf(str(PDF_PATH), stylesheets=[page_css])
print(f"Written: {PDF_PATH}")
