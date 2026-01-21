#!/usr/bin/env python3

# pip install python-docx
# pandoc inp.md -o out.pdf --pdf-engine=xelatex

import os
import zipfile
from pathlib import Path
from docx import Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph
from docx.shared import Inches
import base64

def extract_images_from_docx(docx_path, output_dir):
    """Wyciąga obrazki z pliku .docx i zapisuje je w katalogu."""
    images = {}
    
    # Otwórz .docx jako archiwum ZIP
    with zipfile.ZipFile(docx_path, 'r') as zip_ref:
        # Znajdź wszystkie pliki obrazków
        image_files = [f for f in zip_ref.namelist() if f.startswith('word/media/')]
        
        for img_file in image_files:
            # Wyciągnij nazwę pliku
            img_name = os.path.basename(img_file)
            
            # Zapisz obrazek
            img_data = zip_ref.read(img_file)
            img_path = os.path.join(output_dir, img_name)
            
            with open(img_path, 'wb') as f:
                f.write(img_data)
            
            images[img_name] = img_path
    
    return images

def get_paragraph_style_off01(paragraph):
    """Określa styl paragrafu (nagłówek, lista, itp.)."""
    if paragraph.style.name.startswith('Heading'):
        level = paragraph.style.name.replace('Heading ', '')
        try:
            return 'heading', int(level)
        except:
            return 'normal', 0
    elif paragraph.style.name.startswith('List'):
        return 'list', 0
    else:
        return 'normal', 0


def get_paragraph_style(paragraph):
    """Określa styl paragrafu (nagłówek, lista, itp.)."""
    style_name = paragraph.style.name
    
    if style_name.startswith('Heading'):
        level = style_name.replace('Heading ', '')
        try:
            return 'heading', int(level)
        except:
            return 'normal', 0
    elif 'List' in style_name or paragraph.text.strip().startswith('-') or paragraph.text.strip().startswith('•'):
        return 'list', 0
    else:
        return 'normal', 0


def format_text_off(run):
    """Formatuje tekst zgodnie z jego stylami (bold, italic, itp.)."""
    # text = run.text.strip()
    text = run.text

    if run.bold and run.italic:
        return f"***{text}***"
    elif run.bold:
        return f"**{text}**"
    elif run.italic:
        return f"*{text}*"
    elif run.underline:
        return f"<u>{text}</u>"
    else:
        return run.text  # bez strip()
    
def format_text(run):
    text = run.text
    
    # Sprawdź czy to subscript/superscript
    if run.font.subscript:
        return f"~{text}~"  # lub _{text}
    elif run.font.superscript:
        return f"^{text}^"
    elif run.bold and run.italic:
        return f"***{text.strip()}***"
    elif run.bold:
        return f"**{text.strip()}**"
    elif run.italic:
        return f"*{text.strip()}*"
    elif run.underline:
        return f"<u>{text.strip()}</u>"
    else:
        return text


def get_image_size(run):
    """Wyciąga rozmiar obrazka z run."""
    try:
        inline = run._element.xpath('.//wp:inline')[0]
        extent = inline.xpath('.//wp:extent')[0]
        cx = int(extent.get('cx'))  # szerokość w EMU (English Metric Units)
        cy = int(extent.get('cy'))  # wysokość w EMU
        
        # Konwersja EMU na cale (914400 EMU = 1 cal)
        width_inches = cx / 914400
        height_inches = cy / 914400
        
        return width_inches, height_inches
    except:
        return None, None


def process_paragraph(paragraph, images_map, images_dir):
        
    # W process_paragraph, dla tego konkretnego tekstu:
    # if 'transformer_block' in paragraph.text:
    #     print(f"\n=== MATH PARAGRAPH ===")
    #     print(f"Full text: {repr(paragraph.text)}")
    #     print(f"Runs: {len(paragraph.runs)}")
    #     for i, run in enumerate(paragraph.runs):
    #         print(f"  Run {i}: text={repr(run.text)}, bold={run.bold}, italic={run.italic}")

    """Przetwarza pojedynczy paragraf."""
    style_type, level = get_paragraph_style(paragraph)

    # if paragraph.text.strip().startswith('-') or paragraph.text.strip().startswith('•'):
    #     print(f"DEBUG: style={paragraph.style.name}, indent={paragraph.paragraph_format.left_indent}, text={paragraph.text[:50]}")

    # Sprawdź wcięcie dla list
    indent_level = 0
    if paragraph.paragraph_format.left_indent:
        # Przelicz wcięcie na poziom (każde ~1.27cm = 457200 EMU to jeden poziom)
        indent_emu = paragraph.paragraph_format.left_indent
        indent_level = int(indent_emu / 457200)
    
    # Sprawdź czy paragraf zawiera obrazki
    has_images = False
    for run in paragraph.runs:
        if 'graphicData' in run._element.xml:
            has_images = True
            break
    
    # Jeśli paragraf zawiera obrazki
    if has_images:
        result = []
        processed_images = set()  # Dodaj to
        for run in paragraph.runs:
            # Sprawdź czy run zawiera obrazek
            if 'graphicData' in run._element.xml:
                width, height = get_image_size(run)
                for rel_id in run._element.xpath('.//a:blip/@r:embed'):
                    rel = paragraph.part.rels[rel_id]
                    if "image" in rel.target_ref:
                        img_name = os.path.basename(rel.target_ref)
                        if img_name in images_map and img_name not in processed_images:
                            if width:
                                # Konwertuj na cm dla LaTeX
                                width_cm = width * 2.54
                                max_width_cm = 16  # Dodaj to
                                if width_cm > max_width_cm:
                                    # result.append(f"\n![image]({Path(images_dir).name}/{img_name}){{width=\\textwidth}}\n")
                                    # result.append(f"\n![]({Path(images_dir).name}/{img_name}){{width=\\textwidth}}\n")
                                    # result.append(f"\n\\begin{{center}}![]({Path(images_dir).name}/{img_name}){{width=\\textwidth}}\\end{{center}}\n")
                                    result.append(f"\n\\begin{{center}}\\includegraphics[width=\\textwidth]{{{Path(images_dir).name}/{img_name}}}\\end{{center}}\n")
                                else:
                                    # result.append(f"\n![image]({Path(images_dir).name}/{img_name}){{width={width_cm:.1f}cm}}\n")
                                    # result.append(f"\n![]({Path(images_dir).name}/{img_name}){{width={width_cm:.1f}cm}}\n")
                                    # result.append(f"\n\\begin{{center}}![]({Path(images_dir).name}/{img_name}){{width={width_cm:.1f}cm}}\\end{{center}}\n")
                                    result.append(f"\n\\begin{{center}}\\includegraphics[width={width_cm:.1f}cm]{{{Path(images_dir).name}/{img_name}}}\\end{{center}}\n")
                                                                        
                            else:
                                result.append(f"\n![image]({Path(images_dir).name}/{img_name})\n")
                            processed_images.add(img_name)

            else:
                result.append(format_text(run))
        return ''.join(result)

    # Zwykły tekst
    text = ''.join([format_text(run) for run in paragraph.runs])

    if not text.strip():
        return ''

    # Sprawdź czy paragraf ma numerację/bullet
    is_list = False
    list_level = 0

    num_pr = paragraph._element.xpath('.//w:numPr')
    if num_pr:
        is_list = True
        ilvl = paragraph._element.xpath('.//w:numPr/w:ilvl/@w:val')
        if ilvl:
            list_level = int(ilvl[0])

    # Jeśli to lista z bullet points
    if is_list:
        text_clean = text.strip()
        indent = '  ' * list_level
        return f"{indent}- {text_clean}\n"

    # Jeśli to nagłówek
    if style_type == 'heading':
        return f"{'#' * level} {text.strip()}\n"

    # Jeśli ma wcięcie (ale nie jest listą)
    if paragraph.paragraph_format.left_indent and paragraph.paragraph_format.left_indent > 457200:
        return f"    {text.strip()}\n"

    # Zwykły tekst
    return f"{text}\n"



def process_table(table):
    """Przetwarza tabelę i konwertuje na markdown."""
    result = []
    
    for i, row in enumerate(table.rows):
        cells = [cell.text.strip() for cell in row.cells]
        result.append('| ' + ' | '.join(cells) + ' |')
        
        # Dodaj separator po pierwszym wierszu (nagłówek)
        if i == 0:
            result.append('| ' + ' | '.join(['---'] * len(cells)) + ' |')
    
    return '\n'.join(result) + '\n\n'

# def get_margins(doc):
#     """Odczytuje marginesy z dokumentu."""
#     section = doc.sections[0]
#     top = section.top_margin.cm
#     bottom = section.bottom_margin.cm
#     left = section.left_margin.cm
#     right = section.right_margin.cm
#     return top, bottom, left, right

def get_margins(doc):
    """Odczytuje marginesy z dokumentu i przelicza je na cm."""
    section = doc.sections[0]
    pgMar = section._sectPr.pgMar
    ns = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    
    # Odczytaj wartości z attrib i przelicz na cm
    top = round(float(pgMar.attrib.get(f'{ns}top', 1440)) / 1440 * 2.54, 2)
    bottom = round(float(pgMar.attrib.get(f'{ns}bottom', 1440)) / 1440 * 2.54, 2)
    left = round(float(pgMar.attrib.get(f'{ns}left', 1814)) / 1440 * 2.54, 2)
    right = round(float(pgMar.attrib.get(f'{ns}right', 1814)) / 1440 * 2.54, 2)
    
    return top, bottom, left, right


def docx_to_markdown(docx_path, output_md_path=None, images_dir=None):
    """
    Konwertuje plik .docx na markdown z obrazkami.
    
    Args:
        docx_path: ścieżka do pliku .docx
        output_md_path: ścieżka do pliku wyjściowego .md (opcjonalne)
        images_dir: katalog na obrazki (opcjonalne)
    """
    docx_path = Path(docx_path)
    
    # Ustaw domyślne ścieżki
    if output_md_path is None:
        output_md_path = docx_path.with_suffix('.md')
    
    # if images_dir is None:
    #     images_dir = docx_path.parent / f"{docx_path.stem}_images"
    
    if images_dir is None:
        images_dir = Path(f"{docx_path.stem}_images")  # Relatywna ścieżka


    # Utwórz katalog na obrazki
    os.makedirs(images_dir, exist_ok=True)
    
    # Wyciągnij obrazki
    print(f"Wyciągam obrazki do: {images_dir}")
    images_map = extract_images_from_docx(docx_path, images_dir)
    print(f"Znaleziono {len(images_map)} obrazków")
    
    # Wczytaj dokument
    doc = Document(docx_path)
    
    top, bottom, left, right = get_margins(doc)

#     markdown_header = """---
# geometry: margin=1cm
# header-includes:
#   - \\usepackage{fontspec}
#   - \\usepackage{adjustbox}
#   - \\setmainfont{DejaVu Serif}
#   - \\setmonofont{DejaVu Sans Mono}
#   - \\usepackage{unicode-math}
#   - \\setmathfont{Latin Modern Math}
# ---
# """

    markdown_header = f"""---
geometry:
  - top={top:.1f}cm
  - bottom={bottom:.1f}cm
  - left={left:.1f}cm
  - right={right:.1f}cm
header-includes:
  - \\usepackage{{fontspec}}
  - \\usepackage{{adjustbox}}
  - \\setmainfont{{Consolas}}
  - \\setmonofont{{Consolas}}
  - \\usepackage{{unicode-math}}
  - \\setmathfont{{Latin Modern Math}}
---
"""

    print(markdown_header)


    markdown_content = []
    
    # Przetwórz wszystkie elementy dokumentu
    for element in doc.element.body:
        if isinstance(element, CT_P):
            # Paragraf
            paragraph = Paragraph(element, doc)
            md_text = process_paragraph(paragraph, images_map, images_dir)
            if md_text:
                markdown_content.append(md_text)
        
        elif isinstance(element, CT_Tbl):
            # Tabela
            table = Table(element, doc)
            md_table = process_table(table)
            markdown_content.append(md_table)
    
    # Zapisz markdown
    # with open(output_md_path, 'w', encoding='utf-8') as f:
        # f.write('\n'.join(markdown_content))
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write(markdown_header + '\n'.join(markdown_content))

    print(f"Konwersja zakończona!")
    print(f"Plik markdown: {output_md_path}")
    print(f"Obrazki: {images_dir}")
    
    return output_md_path, images_dir

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Użycie: docx2md.py <plik.docx>")
        sys.exit(1)
    
    docx_file = sys.argv[1]
    
    if not os.path.exists(docx_file):
        print(f"Błąd: Plik {docx_file} nie istnieje!")
        sys.exit(1)
    
    # Konwertuj
    md_file, img_dir = docx_to_markdown(docx_file)
    
    print(f"\n✓ Gotowe! Sprawdź plik: {md_file}")


