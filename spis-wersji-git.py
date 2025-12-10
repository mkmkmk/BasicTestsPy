#!/usr/bin/env python3
import sys
import csv
import subprocess
from pathlib import Path

def get_git_commits(subdir_path, min_lines=10):
    """Pobiera commity z tagami lub z min_lines zmian dla podkatalogu"""
    
    # Sprawdź czy to repozytorium git
    try:
        subprocess.run(['git', 'rev-parse', '--git-dir'], 
                      cwd=".", check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print(f"Błąd: {subdir_path} nie jest w repozytorium git")
        sys.exit(1)
    
    commits = []
    subdir_name = Path(subdir_path).name
    print(f"DEBUG: Szukam commitów dla podkatalogu: {subdir_name}")
    print(f"DEBUG: Próg linii zmian: {min_lines}")
    
    # Format: hash|date|author|subject
    git_cmd = [
        'git', 'log', '--oneline', '--pretty=format:%H|%ad|%an|%s',
        '--date=short', '--', subdir_path
    ]
    
    print(f"DEBUG: Komenda git: {' '.join(git_cmd)}")
    
    try:
        result = subprocess.run(git_cmd, cwd=".", 
                              capture_output=True, text=True, check=True)
        
        all_commits = result.stdout.strip().split('\n')
        print(f"DEBUG: Znaleziono {len(all_commits)} commitów w historii")
        
        if not result.stdout.strip():
            print("DEBUG: Brak commitów w historii dla tego podkatalogu")
            return commits
        
        for i, line in enumerate(all_commits):
            if not line:
                continue
                
            print(f"DEBUG: Przetwarzam commit {i+1}/{len(all_commits)}")
            
            try:
                hash_full, date, author, subject = line.split('|', 3)
                hash_short = hash_full[:8]
                print(f"DEBUG: Commit {hash_short}: {subject[:50]}...")
                
                # Sprawdź czy commit ma tag
                tag_cmd = ['git', 'tag', '--points-at', hash_full]
                tag_result = subprocess.run(tag_cmd, cwd=".", 
                                          capture_output=True, text=True)
                has_tag = bool(tag_result.stdout.strip())
                if has_tag:
                    print(f"DEBUG: Commit {hash_short} ma tag: {tag_result.stdout.strip()}")
                
                # Sprawdź liczbę zmian
                stat_cmd = ['git', 'show', '--stat', '--format=', hash_full, '--', subdir_path]
                stat_result = subprocess.run(stat_cmd, cwd=".", 
                                           capture_output=True, text=True)
                
                # Policz linie zmian (dodane + usunięte)
                lines_changed = 0
                for stat_line in stat_result.stdout.split('\n'):
                    if '|' in stat_line and ('+' in stat_line or '-' in stat_line):
                        # Wyciągnij liczbę zmian z linii typu: "file.py | 15 +++++-----"
                        parts = stat_line.split('|')
                        if len(parts) > 1:
                            changes_part = parts[1].strip()
                            # Znajdź pierwszą liczbę
                            import re
                            match = re.search(r'\d+', changes_part)
                            if match:
                                lines_changed += int(match.group())
                
                print(f"DEBUG: Commit {hash_short} zmienił {lines_changed} linii")
                
                hash_display = hash_short
                if has_tag:
                    tag_names = tag_result.stdout.strip().split('\n')[0]  # pierwszy tag
                    hash_display = f"{hash_short}/{tag_names}"
    
                # Dodaj commit jeśli ma tag lub przekracza próg zmian
                if has_tag or lines_changed >= min_lines:
                    print(f"DEBUG: ✓ Commit {hash_short} spełnia kryteria (tag: {has_tag}, linie: {lines_changed})")
                    commits.append({
                        'nazwa': subdir_name,
                        'data': date,
                        'hash': hash_display,
                        'opis': subject,
                        'autor': author
                    })
                else:
                    print(f"DEBUG: ✗ Commit {hash_short} nie spełnia kryteriów (tag: {has_tag}, linie: {lines_changed})")
                    
            except ValueError as e:
                print(f"DEBUG: Błąd parsowania linii: {line[:100]}... - {e}")
                continue
                
    except subprocess.CalledProcessError as e:
        print(f"Błąd git: {e}")
        sys.exit(1)
    
    print(f"DEBUG: Końcowo znaleziono {len(commits)} commitów spełniających kryteria")
    return commits

def main():
    if len(sys.argv) < 2:
        print("Użycie: python3 script.py <ścieżka_do_podkatalogu> [min_linii_zmian]")
        print("Przykład: python3 script.py ./SigLib 20")
        sys.exit(1)
    
    subdir_path = sys.argv[1]
    min_lines = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    if not Path(subdir_path).exists():
        print(f"Błąd: Katalog {subdir_path} nie istnieje")
        sys.exit(1)
    
    commits = get_git_commits(subdir_path, min_lines)
    
    if not commits:
        print("Nie znaleziono commitów spełniających kryteria")
        return
    
    # Zapisz do CSV
    output_file = f"{Path(subdir_path).name}_commits.csv"
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Nazwa', 'Data', 'Hash', 'Opis', 'Autor']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for commit in commits:
            writer.writerow({
                'Nazwa': commit['nazwa'],
                'Data': commit['data'], 
                'Hash': commit['hash'],
                'Opis': commit['opis'],
                'Autor': commit['autor']
            })
    
    print(f"Zapisano {len(commits)} commitów do {output_file}")
    
    # Pokaż też na ekranie
    print("\nZnalezione commity:")
    for commit in commits:
        print(f"{commit['nazwa']}\t{commit['data']}\t{commit['hash']}\t{commit['opis']}\t{commit['autor']}")

if __name__ == "__main__":
    main()
