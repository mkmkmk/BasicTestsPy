import re
import subprocess
import sys
from pathlib import Path

# from docx.oxml.ns import qn
# from docx.oxml import parse_xml

def process_table_captions(content: str) -> str:
    """
    Numeruje podpisy tabel (Table: (\#tab:label) opis -> Tabela N. opis)
    i zamienia \@ref(tab:label) na numer tabeli.
    """
    table_counter = 0
    table_map = {}
    
    def replace_caption(match):
        nonlocal table_counter
        table_counter += 1
        label = match.group(1)
        caption_text = match.group(2).strip()
        table_map[label] = table_counter
        return f'Table: Tabela {table_counter}. {caption_text}'
    
    # \\?# obsługuje zarówno (#tab:...) jak i (\#tab:...)
    content = re.sub(
        r'Table:\s*\(\\?#(tab:[\w\-]+)\)\s*(.*)',
        replace_caption,
        content
    )
    
    def replace_ref(match):
        label = match.group(1)
        return str(table_map.get(label, '?'))
    
    # \\?@ obsługuje zarówno @ref(...) jak i \@ref(...)
    content = re.sub(r'\\?@ref\((tab:[\w\-]+)\)', replace_ref, content)
    
    return content


def apply_document_styling(docx_path: str):
    """Ustawia czcionkę i wyjustowanie na już przekonwertowanym docx,
    zamiast używać --reference-doc (który psuje tabele z formułami)."""
    from docx import Document
    from docx.shared import Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    
    doc = Document(docx_path)
    
    style = doc.styles['Normal']
    style.font.name = 'Times New Roman'
    style.font.size = Pt(12)
    
    # Justuj tylko akapity spoza tabel (doc.paragraphs pomija te w tabelach)
    count = 0
    for para in doc.paragraphs:
        if para.text.strip():
            para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            count += 1
    
    doc.save(docx_path)
    print(f"✓ Zastosowano styl dokumentu ({count} akapitów wyjustowano)")

 

def strip_cell_borders(table):
    """Usuwa tcBorders ze wszystkich komórek, żeby dziedziczyły z tblBorders."""
    from docx.oxml.ns import qn
    
    removed = 0
    for row in table.rows:
        for cell in row.cells:
            tcPr = cell._tc.tcPr
            if tcPr is None:
                continue
            tcBorders = tcPr.find(qn('w:tcBorders'))
            if tcBorders is not None:
                tcPr.remove(tcBorders)
                removed += 1
    
    print(f"[debug] Usunięto tcBorders z {removed} komórek")
    

def debug_table_xml(docx_path: str):
    """Diagnostyka: pokazuje surowy XML tabeli przed/po modyfikacji."""
    from docx import Document
    import docx
    
    print(f"python-docx wersja: {docx.__version__ if hasattr(docx, '__version__') else 'nieznana'}")
    
    doc = Document(docx_path)
    print(f"Liczba tabel w dokumencie: {len(doc.tables)}")
    
    if not doc.tables:
        print("⚠ BRAK TABEL - pandoc mógł nie utworzyć prawdziwej tabeli docx")
        print("Sprawdź czy w markdown tabela ma poprawny format (nagłówek + | --- |)")
        return
    
    for idx, table in enumerate(doc.tables):
        print(f"\n--- Tabela {idx} ---")
        print(f"Styl: {table.style.name if table.style else 'brak'}")
        print(f"Liczba wierszy: {len(table.rows)}, kolumn: {len(table.columns)}")
        
        tblPr_xml = table._tbl.tblPr.xml if table._tbl.tblPr is not None else "BRAK tblPr"
        print(f"tblPr XML:\n{tblPr_xml}")


def set_table_borders(table, val='single', sz='4', color='000000'):
    """Wymusza obramowanie tabeli, wstawiając element w poprawnym
    miejscu wg sekwencji OOXML."""
    from docx.oxml.ns import qn
    from docx.oxml import parse_xml
    
    tblPr = table._tbl.tblPr
    
    existing = tblPr.find(qn('w:tblBorders'))
    if existing is not None:
        print("  [debug] Usuwam istniejący tblBorders")
        tblPr.remove(existing)
    
    ns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    edges = ('top', 'left', 'bottom', 'right', 'insideH', 'insideV')
    edges_xml = ''.join(
        f'<w:{edge} w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        for edge in edges
    )
    tblBorders = parse_xml(f'<w:tblBorders {ns}>{edges_xml}</w:tblBorders>')
    
    successors = ('w:tblLayout', 'w:tblCellMar', 'w:tblLook')
    inserted = False
    for tag in successors:
        sibling = tblPr.find(qn(tag))
        if sibling is not None:
            sibling.addprevious(tblBorders)
            print(f"  [debug] Wstawiono tblBorders przed {tag}")
            inserted = True
            break
    
    if not inserted:
        tblPr.append(tblBorders)
        print("  [debug] Wstawiono tblBorders na końcu (append)")
    
    # print(f"  [debug] tblPr po zmianie:\n{tblPr.xml}")


def fix_tables_style(docx_path: str, style_name: str = 'Table Grid'):
    """Nakłada styl + wymusza obramowanie na wszystkich tabelach w DOCX."""
    try:
        from docx import Document
        import hashlib
        
        # Sprawdź hash pliku przed zapisem
        with open(docx_path, 'rb') as f:
            hash_before = hashlib.md5(f.read()).hexdigest()
        
        doc = Document(docx_path)
        count = 0
        
        print(f"\n[debug] Znaleziono {len(doc.tables)} tabel w dokumencie")
        
        for table in doc.tables:
            print(f"\n[debug] Przetwarzam tabelę, obecny styl: {table.style.name if table.style else 'brak'}")
            
            try:
                table.style = style_name
                print(f"[debug] Ustawiono styl '{style_name}' OK")
            except KeyError as e:
                print(f"[debug] KeyError przy ustawianiu stylu: {e}")
            
            set_table_borders(table)
            strip_cell_borders(table)
            
            count += 1
        
        doc.save(docx_path)
        
        # Sprawdź hash po zapisie
        with open(docx_path, 'rb') as f:
            hash_after = hashlib.md5(f.read()).hexdigest()
        
        print(f"\n[debug] Hash przed: {hash_before}")
        print(f"[debug] Hash po:    {hash_after}")
        print(f"[debug] Plik {'ZMIENIONY' if hash_before != hash_after else 'BEZ ZMIAN!!'}")
        
        print(f"✓ Ustawiono obramowanie dla {count} tabel")
        
    except ImportError:
        print("⚠ Brak python-docx. Zainstaluj: pip install python-docx")
    except Exception as e:
        import traceback
        print(f"⚠ Błąd stylowania tabel: {e}")
        traceback.print_exc()
        

def fix_matrix_brackets(content: str) -> str:
    """
    Naprawia problem z \left[ ... \right] w macierzach/wektorach,
    które pandoc źle konwertuje do OMML (zamienia ] na )).
    Zamienia \left[ \begin{array}...\end{array} \right] na \begin{bmatrix}...\end{bmatrix}.
    Dla pozostałych przypadków \left[ ... \right] usuwa auto-sizing.
    """
    # Przypadek 1: macierz w array -> bmatrix (najpewniejsze renderowanie w OMML)
    content = re.sub(
        r'\\left\[\s*\\begin\{array\}\{[^}]*\}(.*?)\\end\{array\}\s*\\right\]',
        r'\\begin{bmatrix}\1\\end{bmatrix}',
        content,
        flags=re.DOTALL
    )

    # Przypadek 2: samodzielny \left[ ... \right] (np. wektor bez array)
    # zamień na zwykłe nawiasy kwadratowe bez auto-sizingu
    content = re.sub(r'\\left\[', '[', content)
    content = re.sub(r'\\right\]', ']', content)

    return content



def preprocess_markdown(md_content: str) -> tuple:
    """
    Przetwarza markdown: numeruje wzory i obrazki, zamienia odnośniki.
    Zwraca: (processed_content, equation_map)
    """
    # md_content = fix_table_pipe_escaping(md_content)
    md_content = process_table_captions(md_content)
    equation_counter = 0
    figure_counter = 0
    equation_map = {}
    
    # Zamień bloki equation na $$ i usuń \label
    def replace_equation(match):
        nonlocal equation_counter
        equation_counter += 1
        
        content = match.group(1)
        
        # Wyciągnij i zapisz label
        label_match = re.search(r'\\label\{(.*?)\}', content)
        if label_match:
            label = label_match.group(1)
            equation_map[label] = equation_counter
            content = re.sub(r'\s*\\label\{.*?\}\s*', '', content)
        
        return f'\n$$\n{content.strip()}\n$$\n\n'
    
    content = re.sub(
        r'\\begin\{equation\}(.*?)\\end\{equation\}',
        replace_equation,
        md_content,
        flags=re.DOTALL
    )
    
    # Zamień \eqref{label} na (numer)
    for label, num in equation_map.items():
        content = re.sub(
            rf'\\eqref\{{{re.escape(label)}\}}',
            f'({num})',
            content
        )

    # Dodaj "Fig. N." do podpisów obrazków
    def replace_figure(match):
        nonlocal figure_counter
        figure_counter += 1
        caption = match.group(1).replace('\n', ' ').strip()
        path = match.group(2).strip()
        return f'![Fig. {figure_counter}. {caption}]({path})'

    content = re.sub(
        r'!\[(.*?)\]\((.*?)\)',
        replace_figure,
        content,
        flags=re.DOTALL
    )
    content = fix_matrix_brackets(content)

    return content, equation_map


def add_equation_numbers_to_docx(docx_path: str):
    """Dodaje numery wzorów do DOCX po konwersji."""
    try:
        from docx import Document
        from docx.shared import Pt
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        
        doc = Document(docx_path)
        eq_counter = 0
        
        i = 0
        while i < len(doc.paragraphs):
            para = doc.paragraphs[i]
            
            # Sprawdź czy akapit zawiera wzór display (bez tekstu)
            if para._element.xpath('.//m:oMath') and len(para.text.strip()) == 0:
                eq_counter += 1
                
                # Wycentruj wzór
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                
                # Dodaj nowy akapit z numerem po prawej
                if i + 1 < len(doc.paragraphs):
                    new_para = doc.paragraphs[i+1].insert_paragraph_before()
                else:
                    new_para = doc.add_paragraph()
                
                new_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                run = new_para.add_run(f'({eq_counter})')
                run.font.size = Pt(11)
                
                i += 2  # Pomiń nowo dodany akapit
            else:
                i += 1
        
        doc.save(docx_path)
        print(f"✓ Dodano {eq_counter} numerów wzorów")
        
    except ImportError:
        print("⚠ Brak python-docx. Zainstaluj: pip install python-docx")
    except Exception as e:
        print(f"⚠ Błąd postprocessingu: {e}")



def convert_md_to_docx(input_md: str, output_docx: str = None, 
                       math_method: str = 'omml',
                       reference_doc: str = None,
                       create_template: bool = True):
    """Konwertuje Markdown na DOCX z numeracją wzorów i obrazków."""
    input_path = Path(input_md)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Plik {input_md} nie istnieje")
    
    # Wczytaj i preprocessuj
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    processed_content, eq_map = preprocess_markdown(content)
    
    print(f"Znaleziono {len(eq_map)} wzorów")
    
    # Zapisz do tymczasowego pliku
    temp_file = input_path.with_suffix('.tmp.md')
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(processed_content)
    
    if output_docx is None:
        output_docx = input_path.with_suffix('.docx')
    
    output_path = Path(output_docx)
    
    # Komenda Pandoc
    cmd = [
        'pandoc',
        str(temp_file),
        '-o', str(output_path),
        '--from', 'markdown+tex_math_dollars',
        '--to', 'docx',
        '--standalone',
        '--citeproc',
        '--resource-path', f"{input_path.parent}:.",
        '--highlight-style', 'tango',
        '--dpi', '300',
    ]
    
    if math_method == 'webtex':
        cmd.extend(['--webtex=https://latex.codecogs.com/png.latex?'])
    
    if reference_doc and Path(reference_doc).exists():
        cmd.extend(['--reference-doc', reference_doc])
    
    try:
        print(f"Konwertuję {input_path.name} -> {output_path.name}")
        print(f"Metoda wzorów: {math_method}\n")
        print("CMD:", ' '.join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if result.stderr:
            print(f"Pandoc: {result.stderr}")
        
        add_equation_numbers_to_docx(str(output_path))
        fix_tables_style(str(output_path))
        apply_document_styling(str(output_path))
        
        # Usuń plik tymczasowy
        temp_file.unlink()
        
        return output_path
        
    except Exception as e:
        if temp_file.exists():
            temp_file.unlink()
        raise


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Użycie: python md2docx.py <plik.md> [output.docx] [opcje]")
        print("\nOpcje:")
        print("  --math omml|webtex    Metoda renderowania wzorów (domyślnie: omml)")
        print("  --template plik.docx  Użyj własnego szablonu Word")
        print("  --no-template         Nie twórz automatycznego szablonu")
        print("\nPrzykłady:")
        print("  python md2docx.py article.md")
        print("  python md2docx.py article.md output.docx --math webtex")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not Path(input_file).exists():
        print(f"✗ Błąd: Plik {input_file} nie istnieje!")
        sys.exit(1)
    
    # Parsuj argumenty
    output_file = None
    math_method = 'omml'
    reference_doc = None
    create_template = True
    
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg == '--math' and i + 1 < len(sys.argv):
            math_method = sys.argv[i + 1]
            i += 2
        elif arg == '--template' and i + 1 < len(sys.argv):
            reference_doc = sys.argv[i + 1]
            i += 2
        elif arg == '--no-template':
            create_template = False
            i += 1
        elif not arg.startswith('--'):
            output_file = arg
            i += 1
        else:
            i += 1
    
    # Konwertuj
    try:
        result = convert_md_to_docx(
            input_file, 
            output_file, 
            math_method=math_method,
            reference_doc=reference_doc,
            create_template=create_template
        )
        print(f"\n✓ Gotowe! Plik: {result}")
    except Exception as e:
        print(f"\n✗ Błąd: {e}")
        sys.exit(1)
