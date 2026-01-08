#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import tempfile
import shutil
from datetime import datetime
import sys
import re
import csv

# Konfiguracja ścieżek (do dostosowania)
REPO1_PATH = "/home/mkrej/dyskE/MojePrg/SymuLBNP/SymuLBNP.git"
REPO2_PATH = "/home/mkrej/dyskE/MojePrg/SymuLBNP/SymuLBNP-MD"
COMMITS1_FILE = "/home/mkrej/dyskE/MojePrg/SymuLBNP/repo-svn/commits1.txt"
COMMITS2_FILE = "/home/mkrej/dyskE/MojePrg/SymuLBNP/repo-svn/commits2.txt"
SVN_REPO = "/home/mkrej/dyskE/MojePrg/SymuLBNP/repo-svn/svn_target"
SVN_WORKING_COPY = "/home/mkrej/dyskE/MojePrg/SymuLBNP/repo-svn/svn_working"
SUBDIR1 = "SymuLBNP.git"
SUBDIR2 = "SymuLBNP-Frontend"
CSV_OUTPUT = "/home/mkrej/dyskE/MojePrg/SymuLBNP/repo-svn/git_to_svn_mapping.csv"

# Katalogi do pominięcia podczas kopiowania
SKIP_DIRS_REPO1 = ['.git', 'node_modules', '__pycache__', '.vscode', 'build', 'dist', 'bin' , 'obj', ".vscode", 'AvaDbgDbViewer', 'Ava', 'MultiChartLib', 'SigChartsLib', 'Win', 'Legacy', 'Periph', 'AvaSymuAllDevDbgGui', 'AvaSymuDbgGui', 'AvaPhysioTestGui']
SKIP_DIRS_REPO2 = ['.git', 'target', '.idea', 'logs', 'temp', 'bin' , 'obj', ".vscode"]

def run_command(cmd, cwd=None, capture_output=True):
    """Wykonuje komendę shell i zwraca wynik"""
    try:
        if isinstance(cmd, str):
            result = subprocess.run(cmd, shell=True, cwd=cwd, 
                                  capture_output=capture_output, 
                                  text=True, check=True)
        else:
            result = subprocess.run(cmd, cwd=cwd, 
                                  capture_output=capture_output, 
                                  text=True, check=True)
        return result.stdout.strip() if capture_output else None
    except subprocess.CalledProcessError as e:
        print(f"Błąd wykonania komendy: {cmd}")
        print(f"Kod błędu: {e.returncode}")
        if capture_output and hasattr(e, 'stderr') and e.stderr:
            print(f"Stderr: {e.stderr}")
        raise

def escape_commit_message(message):
    """Escape'uje komentarz dla bezpiecznego użycia w shell"""
    message = message.replace('"', '\\"')
    message = message.replace('`', '\\`')
    message = message.replace('$', '\\$')
    message = message.replace('\\', '\\\\')
    message = re.sub(r'[\x00-\x1f\x7f]', '', message)
    return message

def validate_sha(repo_path, sha):
    """Sprawdza czy SHA istnieje w repo"""
    try:
        run_command(['git', 'cat-file', '-e', sha], cwd=repo_path)
        return True
    except:
        return False

def should_skip_item(item_name, skip_dirs):
    """Sprawdza czy katalog/plik powinien być pominięty"""
    return item_name in skip_dirs

def copy_with_exclusions(src_dir, dst_dir, skip_dirs):
    """Kopiuje zawartość katalogu z wykluczeniami"""
    if not os.path.exists(src_dir):
        return
    
    for item in os.listdir(src_dir):
        if should_skip_item(item, skip_dirs):
            continue
            
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(dst_dir, item)
        
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, ignore=lambda dir, files: 
                           [f for f in files if should_skip_item(f, skip_dirs)])
        else:
            shutil.copy2(src_path, dst_path)

def read_commits_file(filepath):
    """Czyta plik z commitami w formacie sha;komentarz"""
    commits = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(';', 1)
                if len(parts) != 2:
                    print(f"Ostrzeżenie: Błąd w linii {line_num} pliku {filepath}: {line}")
                    continue
                
                sha, comment = parts
                sha = sha.strip()
                comment = comment.strip()
                
                if not re.match(r'^[a-fA-F0-9]{7,40}$', sha):
                    print(f"Ostrzeżenie: Nieprawidłowy SHA w linii {line_num}: {sha}")
                    continue
                
                commits.append((sha, comment))
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku: {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Błąd czytania pliku {filepath}: {e}")
        sys.exit(1)
    
    return commits

def get_commit_timestamp(repo_path, sha):
    """Pobiera timestamp commita z repo git"""
    timestamp_str = run_command(['git', 'show', '-s', '--format=%ct', sha], cwd=repo_path)
    return int(timestamp_str)

def get_commit_date_iso(repo_path, sha):
    """Pobiera datę commita w formacie ISO dla SVN"""
    return run_command(['git', 'show', '-s', '--format=%ci', sha], cwd=repo_path)

def get_commit_author(repo_path, sha):
    """Pobiera autora commita z repo git"""
    return run_command(['git', 'show', '-s', '--format=%an <%ae>', sha], cwd=repo_path)

def checkout_commit(repo_path, sha, target_dir, skip_dirs):
    """Przywraca stan plików z danego commita do katalogu docelowego"""
    # Usuń zawartość katalogu docelowego (zachowaj .svn)
    if os.path.exists(target_dir):
        for item in os.listdir(target_dir):
            if item == '.svn':
                continue
            item_path = os.path.join(target_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
    else:
        os.makedirs(target_dir)
    
    # Użyj git archive do wyeksportowania stanu z commita
    try:
        with tempfile.TemporaryDirectory() as temp_extract_dir:
            archive_cmd = f"git archive --format=tar {sha} | tar -x -C '{temp_extract_dir}'"
            run_command(archive_cmd, cwd=repo_path, capture_output=False)
            
            # Kopiuj z wykluczeniami
            copy_with_exclusions(temp_extract_dir, target_dir, skip_dirs)
            
    except Exception as e:
        print(f"   Ostrzeżenie: git archive nie powiódł się dla {sha}, używam fallback")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                run_command(['git', 'clone', repo_path, temp_dir + '/temp_repo'])
                temp_repo = temp_dir + '/temp_repo'
                run_command(['git', 'checkout', sha], cwd=temp_repo)
                copy_with_exclusions(temp_repo, target_dir, skip_dirs)
        except Exception as e2:
            print(f"   Błąd: Nie można przywrócić stanu commita {sha}: {e2}")
            raise

def create_svn_repo():
    """Tworzy nowe repozytorium SVN"""
    if os.path.exists(SVN_REPO):
        print(f"Usuwam istniejące repo SVN: {SVN_REPO}")
        shutil.rmtree(SVN_REPO)
    
    print(f"Tworzę nowe repo SVN: {SVN_REPO}")
    run_command(['svnadmin', 'create', SVN_REPO])
    # Utwórz hook dla zmiany właściwości rewizji
    hook_path = os.path.join(SVN_REPO, 'hooks', 'pre-revprop-change')
    with open(hook_path, 'w') as f:
        f.write('''#!/bin/sh
REPOS="$1"
REV="$2"
USER="$3"
PROPNAME="$4"
ACTION="$5"

if [ "$ACTION" = "M" -a "$PROPNAME" = "svn:log" ]; then exit 0; fi
if [ "$ACTION" = "M" -a "$PROPNAME" = "svn:date" ]; then exit 0; fi
if [ "$ACTION" = "M" -a "$PROPNAME" = "svn:author" ]; then exit 0; fi

echo "Changing revision properties other than svn:log, svn:date, svn:author is prohibited" >&2
exit 1
''')
    os.chmod(hook_path, 0o755)    

    if os.path.exists(SVN_WORKING_COPY):
        shutil.rmtree(SVN_WORKING_COPY)
    
    svn_url = f"file://{os.path.abspath(SVN_REPO)}"
    run_command(['svn', 'checkout', svn_url, SVN_WORKING_COPY])
    
    subdir1_path = os.path.join(SVN_WORKING_COPY, SUBDIR1)
    subdir2_path = os.path.join(SVN_WORKING_COPY, SUBDIR2)
    os.makedirs(subdir1_path, exist_ok=True)
    os.makedirs(subdir2_path, exist_ok=True)
    
    run_command(['svn', 'add', SUBDIR1, SUBDIR2], cwd=SVN_WORKING_COPY)
    run_command(['svn', 'commit', '-m', 'Utworzenie struktury katalogów'], cwd=SVN_WORKING_COPY)

def convert_git_date_to_svn(git_date):
    """Konwertuje datę z git do formatu SVN"""
    return run_command(['date', '-d', git_date, '+%Y-%m-%dT%H:%M:%S.000000Z'])

def svn_commit_with_date(message, date_iso, author, working_copy_path):
    """Wykonuje commit SVN z określoną datą i autorem"""
    run_command(['svn', 'update'], cwd=working_copy_path)
    run_command(['svn', 'add', '--force', '.'], cwd=working_copy_path)
    
    try:
        status_output = run_command(['svn', 'status'], cwd=working_copy_path)
        for line in status_output.split('\n'):
            line = line.strip()
            if line.startswith('!'):
                missing_file = line[1:].strip()
                if missing_file:
                    try:
                        run_command(['svn', 'remove', missing_file], cwd=working_copy_path)
                    except:
                        print(f"   Ostrzeżenie: Nie można usunąć {missing_file}")
    except:
        pass
    
    try:
        status_output = run_command(['svn', 'status'], cwd=working_copy_path)
        changes = [line for line in status_output.split('\n') 
                  if line.strip() and not line.startswith('?') and line.strip() != '']
        
        if not changes:
            print(f"   Brak zmian do commitowania, pomijam...")
            return
    except:
        pass
    
    safe_message = escape_commit_message(message)
    
    try:
        run_command(['svn', 'commit', '-m', safe_message], cwd=working_copy_path)
        
        info_output = run_command(['svn', 'info'], cwd=working_copy_path)
        revision = None
        for line in info_output.split('\n'):
            if line.startswith('Revision:'):
                revision = line.split(':')[1].strip()
                break
        
        if revision:
            svn_url = f"file://{os.path.abspath(SVN_REPO)}"
            svn_date = convert_git_date_to_svn(date_iso)

            run_command(['svn', 'propset', '--revprop', f'-r{revision}', 
                        'svn:date', svn_date, svn_url])

            run_command(['svn', 'propset', '--revprop', f'-r{revision}', 
                        'svn:author', author, svn_url])

            return revision

    except Exception as e:
        print(f"   Ostrzeżenie: Błąd podczas commitowania: {e}")

    return None


def write_csv_mapping(csv_path, mapping_data):
    """Zapisuje mapowanie Git SHA -> SVN revision do pliku CSV"""
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Git SHA', 'SVN Revision', 'Repository', 'Comment'])
            for row in mapping_data:
                writer.writerow(row)
        print(f"\n   Zapisano mapowanie do pliku: {csv_path}")
    except Exception as e:
        print(f"\n   Błąd zapisu pliku CSV: {e}")


def main():
    os.environ['LC_ALL'] = 'C.utf8'
    os.environ['LANG'] = 'C.utf8'
    print("=== Git to SVN Migration Script ===")
    print(f"Katalogi pomijane w repo1: {SKIP_DIRS_REPO1}")
    print(f"Katalogi pomijane w repo2: {SKIP_DIRS_REPO2}")
    
    paths_to_check = [
        (REPO1_PATH, "REPO1"),
        (REPO2_PATH, "REPO2"), 
        (COMMITS1_FILE, "COMMITS1"),
        (COMMITS2_FILE, "COMMITS2")
    ]
    
    for path, name in paths_to_check:
        if not os.path.exists(path):
            print(f"Błąd: Nie znaleziono {name}: {path}")
            sys.exit(1)
    
    print("1. Czytam pliki z commitami...")
    commits1 = read_commits_file(COMMITS1_FILE)
    commits2 = read_commits_file(COMMITS2_FILE)
    
    print(f"   Repo1: {len(commits1)} commitów")
    print(f"   Repo2: {len(commits2)} commitów")
    
    print("2. Walidacja i pobieranie metadanych commitów...")
    all_commits = []
    
    # Przetwarzaj commity z repo1
    for sha, comment in commits1:
        try:
            if not validate_sha(REPO1_PATH, sha):
                print(f"   Ostrzeżenie: SHA {sha} nie istnieje w repo1, pomijam")
                continue
                
            timestamp = get_commit_timestamp(REPO1_PATH, sha)
            date_iso = get_commit_date_iso(REPO1_PATH, sha)
            author = get_commit_author(REPO1_PATH, sha)
            all_commits.append((timestamp, sha, comment, REPO1_PATH, SUBDIR1, date_iso, author, SKIP_DIRS_REPO1))
        except Exception as e:
            print(f"   Ostrzeżenie: Błąd dla commita {sha} z repo1: {e}")
            continue
    
    # Przetwarzaj commity z repo2
    for sha, comment in commits2:
        try:
            if not validate_sha(REPO2_PATH, sha):
                print(f"   Ostrzeżenie: SHA {sha} nie istnieje w repo2, pomijam")
                continue
                
            timestamp = get_commit_timestamp(REPO2_PATH, sha)
            date_iso = get_commit_date_iso(REPO2_PATH, sha)
            author = get_commit_author(REPO2_PATH, sha)
            all_commits.append((timestamp, sha, comment, REPO2_PATH, SUBDIR2, date_iso, author, SKIP_DIRS_REPO2))
        except Exception as e:
            print(f"   Ostrzeżenie: Błąd dla commita {sha} z repo2: {e}")
            continue
    
    if not all_commits:
        print("Błąd: Brak prawidłowych commitów do przetworzenia!")
        sys.exit(1)
    
    print("3. Sortuję commity chronologicznie...")
    all_commits.sort(key=lambda x: x[0])  # sortuj po timestamp
    
    print(f"   Łącznie: {len(all_commits)} commitów do przetworzenia")
    
    print("4. Tworzę repozytorium SVN...")
    try:
        create_svn_repo()
    except Exception as e:
        print(f"Błąd tworzenia repo SVN: {e}")
        sys.exit(1)
    
    print("5. Przetwarzam commity...")
    successful_commits = 0
    csv_mapping = []

    for i, (timestamp, sha, comment, repo_path, subdir, date_iso, author, skip_dirs) in enumerate(all_commits, 1):
        print(f"   [{i}/{len(all_commits)}] {sha[:8]} by {author} - {comment[:50]}...")
        
        try:
            # Przywróć stan z commita
            target_dir = os.path.join(SVN_WORKING_COPY, subdir)
            checkout_commit(repo_path, sha, target_dir, skip_dirs)
            
            # Commit do SVN z autorem
            svn_revision = svn_commit_with_date(comment, date_iso, author, SVN_WORKING_COPY)

            if svn_revision:
                repo_name = "Repo1" if repo_path == REPO1_PATH else "Repo2"
                csv_mapping.append([sha, svn_revision, repo_name, comment])
                successful_commits += 1

            successful_commits += 1
            
        except Exception as e:
            print(f"   Błąd przetwarzania commita {sha}: {e}")
            continue

    write_csv_mapping(CSV_OUTPUT, csv_mapping)

    print("6. Gotowe!")
    print(f"   Pomyślnie przetworzono: {successful_commits}/{len(all_commits)} commitów")
    print(f"   SVN repo: {SVN_REPO}")
    print(f"   Working copy: {SVN_WORKING_COPY}")

if __name__ == "__main__":
    main()

