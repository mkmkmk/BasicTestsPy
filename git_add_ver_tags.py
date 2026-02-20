#!/usr/bin/env python3
"""
Skrypt do dodawania tagów Git na podstawie pliku CSV z wersjami i SHA commitów.
Sprawdza duplikaty i konflikty przed dodaniem tagów.

# Dry-run dla Repo1
python add_git_ver_tags.py git_to_svn_mapping-wersje.csv Repo1 --dry-run

# Tagowanie Repo2
python add_git_ver_tags.py git_to_svn_mapping-wersje.csv Repo2

"""

import csv
import subprocess
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import argparse

def read_csv_off1(filename: str) -> List[Tuple[str, str]]:
    """Wczytuje CSV i zwraca listę (wersja, sha)"""
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if len(row) < 2:
                continue
            
            # Parsowanie: "1.22.1/2025-11-03" -> "1.22.1"
            version_part = row[0].strip().strip('"')
            version = version_part.split('/')[0]
            
            # Parsowanie: "185/7a14955d" -> "7a14955d"
            sha_part = row[1].strip().strip('"')
            sha = sha_part.split('/')[1] if '/' in sha_part else sha_part
            
            data.append((version, sha))
    
    return data


def read_csv_off2(filename: str, target_repo: str) -> List[Tuple[str, str]]:
    """Wczytuje CSV i zwraca listę (wersja, sha) dla wybranego repo"""
    data = []
    current_version = None
    last_commit_for_version = {}  # {version: (sha, date)}
    
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)  # Pomiń nagłówek
        
        for row in reader:
            if len(row) == 0 or not row[0].strip():
                continue
            
            line = row[0].strip()
            
            # Linia z wersją: "--- 1.4.0"
            if line.startswith('---'):
                if current_version and current_version in last_commit_for_version:
                    sha, _ = last_commit_for_version[current_version]
                    data.append((current_version, sha))
                
                current_version = line.replace('---', '').strip()
                last_commit_for_version[current_version] = None
                continue
            
            # Linia z commitem
            if len(row) >= 4:
                sha = row[0].strip()
                repo = row[2].strip()
                date = row[3].strip()
                
                # Tylko commity z wybranego repo
                if repo == target_repo and current_version:
                    # Zapisz lub zaktualizuj ostatni commit dla tej wersji
                    if current_version not in last_commit_for_version or last_commit_for_version[current_version] is None:
                        last_commit_for_version[current_version] = (sha, date)
                    else:
                        # Porównaj daty i weź późniejszy
                        _, prev_date = last_commit_for_version[current_version]
                        if date > prev_date:
                            last_commit_for_version[current_version] = (sha, date)
        
        # Dodaj ostatnią wersję jeśli była
        if current_version and current_version in last_commit_for_version and last_commit_for_version[current_version]:
            sha, _ = last_commit_for_version[current_version]
            data.append((current_version, sha))
    
    return data


def read_csv_off3(filename: str, target_repo: str) -> List[Tuple[str, str]]:
    """Wczytuje CSV i zwraca listę (wersja, sha) dla wybranego repo"""
    data = []
    current_version = None
    last_commit_for_version = {}  # {version: (sha, date)}
    
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)  # Pomiń nagłówek
        
        for row in reader:
            if len(row) == 0 or not row[0].strip():
                continue
            
            line = row[0].strip()
            
            # Linia z wersją: "--- 1.4.0"
            if line.startswith('---'):
                # Zapisz poprzednią wersję jeśli była
                if current_version and last_commit_for_version.get(current_version):
                    sha, _ = last_commit_for_version[current_version]
                    data.append((current_version, sha))
                
                current_version = line.replace('---', '').strip()
                last_commit_for_version[current_version] = None
                continue
            
            # Linia z commitem
            if len(row) >= 4:
                sha = row[0].strip()
                repo = row[2].strip()
                date = row[3].strip()
                
                # Tylko commity z wybranego repo
                if repo == target_repo and current_version:
                    # Zapisz lub zaktualizuj ostatni commit dla tej wersji
                    prev = last_commit_for_version.get(current_version)
                    if prev is None:
                        last_commit_for_version[current_version] = (sha, date)
                    else:
                        # Porównaj daty i weź późniejszy
                        _, prev_date = prev
                        if date > prev_date:
                            last_commit_for_version[current_version] = (sha, date)
        
        # Dodaj ostatnią wersję jeśli była
        if current_version and last_commit_for_version.get(current_version):
            sha, _ = last_commit_for_version[current_version]
            data.append((current_version, sha))
    
    return data

def read_csv(filename: str, target_repo: str) -> List[Tuple[str, str]]:
    """Wczytuje CSV i zwraca listę (wersja, sha) dla wybranego repo"""
    data = []
    current_commits = []  # Commity przed następną wersją
    
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)  # Pomiń nagłówek
        
        for row in reader:
            if len(row) == 0 or not row[0].strip():
                continue
            
            line = row[0].strip()
            
            # Linia z wersją: "--- 1.4.0"
            if line.startswith('---'):
                version = line.replace('---', '').strip()
                
                # Znajdź ostatni commit z target_repo przed tą linią
                for sha, date, repo in reversed(current_commits):
                    if repo == target_repo:
                        data.append((version, sha))
                        break
                
                current_commits = []  # Reset dla następnej wersji
                continue
            
            # Linia z commitem
            if len(row) >= 4:
                sha = row[0].strip()
                repo = row[2].strip()
                date = row[3].strip()
                current_commits.append((sha, date, repo))
    
    return data
    
def check_conflicts(data: List[Tuple[str, str]]) -> Tuple[bool, Dict[str, Set[str]]]:
    """
    Sprawdza czy jedna wersja nie ma przypisanych różnych SHA.
    Zwraca (czy_są_konflikty, słownik_konfliktów)
    """
    version_to_shas = defaultdict(set)
    
    for version, sha in data:
        version_to_shas[version].add(sha)
    
    conflicts = {v: shas for v, shas in version_to_shas.items() if len(shas) > 1}
    
    return len(conflicts) > 0, conflicts

def remove_duplicates(data: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Usuwa duplikaty (ta sama wersja + SHA)"""
    seen = set()
    unique = []
    
    for version, sha in data:
        key = (version, sha)
        if key not in seen:
            seen.add(key)
            unique.append((version, sha))
    
    return unique

def verify_commit_exists(sha: str) -> bool:
    """Sprawdza czy commit o danym SHA istnieje w repo"""
    try:
        result = subprocess.run(
            ['git', 'cat-file', '-t', sha],
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode == 0 and result.stdout.strip() == 'commit'
    except Exception:
        return False

def tag_exists(tag: str) -> bool:
    """Sprawdza czy tag już istnieje"""
    try:
        result = subprocess.run(
            ['git', 'tag', '-l', tag],
            capture_output=True,
            text=True,
            check=False
        )
        return bool(result.stdout.strip())
    except Exception:
        return False

def get_tag_commit(tag: str) -> str:
    """Zwraca SHA commita na który wskazuje tag"""
    try:
        result = subprocess.run(
            ['git', 'rev-list', '-n', '1', tag],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return ""
    except Exception:
        return ""

def create_tag(version: str, sha: str, force: bool = False, dry_run: bool = False) -> bool:
    """Tworzy tag dla danego commita"""
    tag = f"v{version}"
    
    if dry_run:
        return True  # W trybie dry-run zawsze sukces
    
    cmd = ['git', 'tag']
    if force:
        cmd.append('-f')
    cmd.extend([tag, sha])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Błąd przy tworzeniu tagu {tag}: {e}")
        return False

def main():
    # Sprawdź argumenty
    # dry_run = '--dry-run' in sys.argv or '-n' in sys.argv
    # filename = 'tags-1.csv'
    
    parser = argparse.ArgumentParser(description='Dodaje tagi Git z pliku CSV')
    parser.add_argument('csv_file', help='Ścieżka do pliku CSV')
    parser.add_argument('repo', choices=['Repo1', 'Repo2'], help='Które repo tagować')
    parser.add_argument('--dry-run', '-n', action='store_true', help='Tryb pustego przebiegu')
    args = parser.parse_args()

    filename = args.csv_file
    target_repo = args.repo
    dry_run = args.dry_run
        
    if dry_run:
        print(f"🔍 TRYB PUSTEGO PRZEBIEGU (dry-run) - repo: {target_repo}\n")
    else:
        print(f"🏷️  Tagowanie repo: {target_repo}\n")
                
        
    print("🔍 Wczytuję dane z CSV...")
    try:
        data = read_csv(filename, target_repo)
    except FileNotFoundError:
        print(f"❌ Nie znaleziono pliku {filename}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Błąd przy czytaniu CSV: {e}")
        sys.exit(1)
    
    print(f"✅ Wczytano {len(data)} wpisów")
    
    # Sprawdzenie konfliktów
    print("\n🔍 Sprawdzam konflikty (jedna wersja -> różne SHA)...")
    has_conflicts, conflicts = check_conflicts(data)
    
    if has_conflicts:
        print("❌ ZNALEZIONO KONFLIKTY! Jedna wersja ma przypisane różne SHA:")
        for version, shas in conflicts.items():
            print(f"   v{version} -> {', '.join(shas)}")
        print("\n⚠️  Popraw plik CSV przed kontynuacją!")
        sys.exit(1)
    
    print("✅ Brak konfliktów")
    
    # Usunięcie duplikatów
    print("\n🔍 Usuwam duplikaty...")
    unique_data = remove_duplicates(data)
    removed = len(data) - len(unique_data)
    if removed > 0:
        print(f"✅ Usunięto {removed} duplikatów")
    else:
        print("✅ Brak duplikatów")
    
    print(f"\n📋 Do przetworzenia: {len(unique_data)} unikalnych tagów")
    
    # Weryfikacja commitów
    print("\n🔍 Sprawdzam czy commity istnieją...")
    missing_commits = []
    for version, sha in unique_data:
        if not verify_commit_exists(sha):
            missing_commits.append((version, sha))
    
    if missing_commits:
        print(f"❌ Nie znaleziono {len(missing_commits)} commitów:")
        for version, sha in missing_commits:
            print(f"   v{version} -> {sha}")
        # if len(missing_commits) > 10:
        #    print(f"   ... i {len(missing_commits) - 10} więcej")
        
        if not dry_run:
            response = input("\n⚠️  Kontynuować mimo brakujących commitów? (tak/nie): ")
            if response.lower() not in ['tak', 't', 'yes', 'y']:
                sys.exit(1)
        
        # Usuń brakujące commity z listy
        unique_data = [(v, s) for v, s in unique_data if (v, s) not in missing_commits]
    else:
        print("✅ Wszystkie commity istnieją")
    
    # Sprawdzenie istniejących tagów i co się zmieni
    print("\n🔍 Sprawdzam istniejące tagi...")
    new_tags = []
    overwrite_tags = []
    unchanged_tags = []
    
    for version, sha in unique_data:
        tag = f"v{version}"
        if tag_exists(tag):
            existing_sha = get_tag_commit(tag)
            if existing_sha == sha:
                unchanged_tags.append((version, sha))
            else:
                overwrite_tags.append((version, sha, existing_sha))
        else:
            new_tags.append((version, sha))
    
    # Podsumowanie zmian
    print("\n" + "="*70)
    print("📊 CO ZOSTANIE ZMIENIONE:")
    print("="*70)
    
    if new_tags:
        print(f"\n✨ NOWE TAGI ({len(new_tags)}):")
        for version, sha in new_tags:
            print(f"   v{version} -> {sha}")
        # if len(new_tags) > 10:
        #    print(f"   ... i {len(new_tags) - 10} więcej")
    
    if overwrite_tags:
        print(f"\n🔄 NADPISANE TAGI ({len(overwrite_tags)}):")
        for version, new_sha, old_sha in overwrite_tags:
            print(f"   v{version}: {old_sha[:8]} -> {new_sha[:8]}")
        # if len(overwrite_tags) > 10:
        #    print(f"   ... i {len(overwrite_tags) - 10} więcej")
    
    if unchanged_tags:
        print(f"\n⏭️  POMINIĘTE (już istnieją z tym samym SHA): {len(unchanged_tags)}")
    
    print("\n" + "="*70)
    print("📈 PODSUMOWANIE:")
    print(f"   • Nowe tagi:      {len(new_tags)}")
    print(f"   • Do nadpisania:  {len(overwrite_tags)}")
    print(f"   • Bez zmian:      {len(unchanged_tags)}")
    print(f"   • RAZEM:          {len(unique_data)}")
    print("="*70)
    
    if dry_run:
        print("\n✅ KONIEC TRYBU PUSTEGO PRZEBIEGU")
        print("💡 Aby wykonać zmiany, uruchom bez --dry-run:")
        print("   python add_git_tags.py")
        sys.exit(0)
    
    # Pytanie o nadpisywanie
    force = False
    if overwrite_tags:
        response = input("\n⚠️  Nadpisać istniejące tagi? (tak/nie): ")
        force = response.lower() in ['tak', 't', 'yes', 'y']
        if not force:
            print("ℹ️  Istniejące tagi zostaną pominięte")
            # Usuń tagi do nadpisania z listy
            unique_data = [(v, s) for v, s in unique_data if (v, s) not in [(v, s) for v, s, _ in overwrite_tags]]
    
    if not new_tags and not (overwrite_tags and force):
        print("\n✅ Nic do zrobienia!")
        sys.exit(0)
    
    response = input("\n❓ Rozpocząć tagowanie? (tak/nie): ")
    if response.lower() not in ['tak', 't', 'yes', 'y']:
        print("❌ Anulowano")
        sys.exit(0)
    
    # Tworzenie tagów
    print("\n🏷️  Tworzę tagi...")
    success_count = 0
    failed = []
    
    for version, sha in unique_data:
        tag = f"v{version}"
        
        # Pomiń jeśli tag istnieje i nie nadpisujemy
        if not force and tag_exists(tag):
            continue
        
        if create_tag(version, sha, force, dry_run=False):
            success_count += 1
            print(f"✅ {tag} -> {sha}")
        else:
            failed.append((version, sha))
            print(f"❌ {tag} -> {sha}")
    
    # Podsumowanie końcowe
    print("\n" + "="*60)
    print("🎉 ZAKOŃCZONO!")
    print(f"   ✅ Sukces: {success_count}")
    if failed:
        print(f"   ❌ Błędy: {len(failed)}")
        print("\nNieudane tagi:")
        for version, sha in failed:
            print(f"   v{version} -> {sha}")
    print("="*60)
    
    if success_count > 0:
        print("\n💡 Aby wypchnąć tagi do zdalnego repo:")
        print("   git push origin --tags")
        if force:
            print("   (użyj --force jeśli nadpisywałeś istniejące)")

if __name__ == '__main__':
    main()


