#!/usr/bin/env python3
"""
yaml_chat_to_md.py

Konwertuje pliki YAML z zapisanymi rozmowami czatu (format: session_id + conversation)
na pliki Markdown.

Obsługuje:
- pojedynczy plik wejściowy -> pojedynczy plik .md
- wiele plików wejściowych (np. *.yaml) -> wiele plików .md (ta sama nazwa, inne rozszerzenie)
- wiele dokumentów YAML w jednym pliku (oddzielonych `---`)

Użycie:
    python yaml_chat_to_md.py rozmowa.yaml
    python yaml_chat_to_md.py rozmowa.yaml -o rozmowa.md
    python yaml_chat_to_md.py *.yaml --outdir md/
"""

import argparse
import sys
from pathlib import Path

import yaml
import re

ROLE_LABELS = {
    "user": "Użytkownik",
    "assistant": "Asystent",
    "system": "System",
}


def role_label(role: str) -> str:
    return ROLE_LABELS.get(role, role.capitalize())


def render_conversation(doc: dict) -> str:
    """Zamienia jeden dokument YAML (jedna rozmowa) na tekst Markdown."""
    session_id = doc.get("session_id", "brak-id")
    conversation = doc.get("conversation", [])

    lines = []
    lines.append(f"# Rozmowa `{session_id}`\n")

    for i, turn in enumerate(conversation, start=1):
        role = turn.get("role", "unknown")
        content = turn.get("content", "")

        # zamień escapowane \$ na zwykłe $ (wymagane do renderowania wzorów inline)
        content = re.sub(r"\\\$", "$", content)

        # content jest już markdownem (tak jak go wyświetla Gradio),
        # więc po prostu go wklejamy, otoczonego nagłówkiem sekcji.
        lines.append(f"## {role_label(role)}\n")
        lines.append(content.rstrip() + "\n")

    return "\n".join(lines) + "\n"


def convert_file(path: Path, outdir: Path | None, output: Path | None) -> None:
    text = path.read_text(encoding="utf-8")

    # obsługa wielu dokumentów YAML w jednym pliku (---)
    docs = list(yaml.safe_load_all(text))
    docs = [d for d in docs if d]  # pomiń puste

    if not docs:
        print(f"[pominięto] brak danych w {path}", file=sys.stderr)
        return

    if output is not None:
        # pojedynczy plik wyjściowy, wszystkie dokumenty sklejone
        md = "\n---\n\n".join(render_conversation(d) for d in docs)
        output.write_text(md, encoding="utf-8")
        print(f"zapisano: {output}")
        return

    # domyślnie: jeden plik .md per plik wejściowy (ew. w outdir)
    target_dir = outdir if outdir is not None else path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    if len(docs) == 1:
        md = render_conversation(docs[0])
        out_path = target_dir / (path.stem + ".md")
        out_path.write_text(md, encoding="utf-8")
        print(f"zapisano: {out_path}")
    else:
        # jeśli wiele dokumentów w jednym pliku -> numerujemy
        for idx, d in enumerate(docs, start=1):
            md = render_conversation(d)
            out_path = target_dir / f"{path.stem}_{idx}.md"
            out_path.write_text(md, encoding="utf-8")
            print(f"zapisano: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Konwersja rozmów YAML -> Markdown")
    parser.add_argument("inputs", nargs="+", type=Path, help="pliki .yaml do konwersji")
    parser.add_argument("-o", "--output", type=Path, default=None,
                         help="plik wyjściowy .md (tylko gdy podano jeden plik wejściowy)")
    parser.add_argument("--outdir", type=Path, default=None,
                         help="katalog docelowy dla plików .md (przy wielu wejściach)")
    args = parser.parse_args()

    if args.output is not None and len(args.inputs) > 1:
        parser.error("--output można użyć tylko z jednym plikiem wejściowym")

    for path in args.inputs:
        if not path.exists():
            print(f"[błąd] nie znaleziono pliku: {path}", file=sys.stderr)
            continue
        convert_file(path, args.outdir, args.output)


if __name__ == "__main__":
    main()

