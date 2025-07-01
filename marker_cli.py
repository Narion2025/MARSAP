#!/usr/bin/env python3
"""
Marker CLI - Command Line Interface für den Marker-Detektor
Ermöglicht die Analyse von Texten direkt über die Kommandozeile
"""

import argparse
import sys
import os
from pathlib import Path
import json
import yaml
from typing import List, Optional
from colorama import init, Fore, Back, Style

# Importiere den Marker Matcher
from marker_matcher import MarkerMatcher

# Importiere CoSD Module
try:
    from cosd import CoSDAnalyzer
    COSD_AVAILABLE = True
except ImportError:
    COSD_AVAILABLE = False

# Initialisiere colorama für farbige Ausgabe
init(autoreset=True)


class MarkerCLI:
    """Command Line Interface für Marker-Analyse"""
    
    def __init__(self):
        self.matcher = MarkerMatcher()
        self.cosd_analyzer = None
        
        # Initialisiere CoSD wenn verfügbar
        if COSD_AVAILABLE:
            try:
                # Verwende erweiterte Marker-Datei falls vorhanden
                marker_path = Path("../the_mind_system/hive_mind/config/markers/Merged_Marker_Set.yaml")
                if not marker_path.exists():
                    marker_path = None
                self.cosd_analyzer = CoSDAnalyzer(
                    marker_data_path=str(marker_path) if marker_path else None,
                    base_matcher=self.matcher
                )
            except Exception as e:
                print(f"{Fore.YELLOW}Warnung: CoSD-Modul konnte nicht initialisiert werden: {e}{Style.RESET_ALL}")
                self.cosd_analyzer = None
        
    def analyze_text(self, text: str, verbose: bool = False):
        """Analysiert einen einzelnen Text"""
        result = self.matcher.analyze_text(text)
        self._print_result(result, verbose)
        
    def analyze_file(self, file_path: str, verbose: bool = False):
        """Analysiert eine Textdatei"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            print(f"\n{Fore.CYAN}Analysiere Datei: {file_path}{Style.RESET_ALL}")
            print("-" * 60)
            
            result = self.matcher.analyze_text(text)
            self._print_result(result, verbose)
            
        except FileNotFoundError:
            print(f"{Fore.RED}Fehler: Datei '{file_path}' nicht gefunden!{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Fehler beim Lesen der Datei: {e}{Style.RESET_ALL}")
    
    def analyze_directory(self, dir_path: str, pattern: str = "*.txt", verbose: bool = False):
        """Analysiert alle Textdateien in einem Verzeichnis"""
        path = Path(dir_path)
        
        if not path.is_dir():
            print(f"{Fore.RED}Fehler: '{dir_path}' ist kein Verzeichnis!{Style.RESET_ALL}")
            return
        
        files = list(path.glob(pattern))
        
        if not files:
            print(f"{Fore.YELLOW}Keine Dateien mit Pattern '{pattern}' gefunden!{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.CYAN}Analysiere {len(files)} Dateien in: {dir_path}{Style.RESET_ALL}")
        print("=" * 60)
        
        total_stats = {
            'files_analyzed': 0,
            'total_markers': 0,
            'risk_levels': {'green': 0, 'yellow': 0, 'blinking': 0, 'red': 0}
        }
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                result = self.matcher.analyze_text(text)
                
                print(f"\n{Fore.BLUE}Datei: {file_path.name}{Style.RESET_ALL}")
                self._print_result(result, verbose=False)  # Kompakte Ausgabe für Batch
                
                # Statistiken sammeln
                total_stats['files_analyzed'] += 1
                total_stats['total_markers'] += len(result.gefundene_marker)
                total_stats['risk_levels'][result.risk_level] += 1
                
            except Exception as e:
                print(f"{Fore.RED}Fehler bei {file_path.name}: {e}{Style.RESET_ALL}")
        
        # Zusammenfassung
        self._print_batch_summary(total_stats)
    
    def _print_result(self, result, verbose: bool = False):
        """Gibt das Analyse-Ergebnis formatiert aus"""
        # Risk-Level mit Farbe
        risk_colors = {
            'green': Fore.GREEN,
            'yellow': Fore.YELLOW,
            'blinking': Fore.MAGENTA,
            'red': Fore.RED
        }
        
        risk_color = risk_colors.get(result.risk_level, Fore.WHITE)
        
        print(f"\n{risk_color}Risk-Level: {result.risk_level.upper()}{Style.RESET_ALL}")
        print(f"Gefundene Marker: {len(result.gefundene_marker)}")
        
        if result.gefundene_marker:
            print("\nGefundene Muster:")
            
            # Gruppiere nach Marker-Namen
            marker_groups = {}
            for match in result.gefundene_marker:
                if match.marker_name not in marker_groups:
                    marker_groups[match.marker_name] = []
                marker_groups[match.marker_name].append(match)
            
            for marker_name, matches in marker_groups.items():
                print(f"\n  {Fore.CYAN}{marker_name}{Style.RESET_ALL} ({len(matches)}x)")
                
                if verbose:
                    # Zeige alle Treffer
                    for i, match in enumerate(matches[:5]):  # Max 5 Beispiele
                        print(f"    - '{match.matched_text[:60]}...'")
                    if len(matches) > 5:
                        print(f"    ... und {len(matches) - 5} weitere")
                else:
                    # Zeige nur erstes Beispiel
                    print(f"    - '{matches[0].matched_text[:60]}...'")
        
        print(f"\n{Fore.CYAN}Zusammenfassung:{Style.RESET_ALL}")
        print(result.summary)
    
    def _print_batch_summary(self, stats: dict):
        """Gibt eine Zusammenfassung für Batch-Analysen aus"""
        print("\n" + "=" * 60)
        print(f"{Fore.CYAN}ZUSAMMENFASSUNG{Style.RESET_ALL}")
        print("=" * 60)
        
        print(f"Analysierte Dateien: {stats['files_analyzed']}")
        print(f"Gefundene Marker gesamt: {stats['total_markers']}")
        print(f"Durchschnitt pro Datei: {stats['total_markers'] / stats['files_analyzed']:.1f}")
        
        print("\nRisk-Level-Verteilung:")
        for level, count in stats['risk_levels'].items():
            percentage = (count / stats['files_analyzed']) * 100
            print(f"  - {level}: {count} ({percentage:.1f}%)")
    
    def list_markers(self, category: Optional[str] = None):
        """Listet alle verfügbaren Marker auf"""
        markers = self.matcher.markers
        
        if category:
            markers = {k: v for k, v in markers.items() 
                      if v.get('kategorie', '').lower() == category.lower()}
        
        print(f"\n{Fore.CYAN}Verfügbare Marker ({len(markers)}){Style.RESET_ALL}")
        print("=" * 60)
        
        # Gruppiere nach Kategorie
        by_category = {}
        for marker_name, marker_data in markers.items():
            cat = marker_data.get('kategorie', 'UNCATEGORIZED')
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append((marker_name, marker_data))
        
        for cat, marker_list in sorted(by_category.items()):
            print(f"\n{Fore.YELLOW}{cat}:{Style.RESET_ALL}")
            
            for name, data in sorted(marker_list):
                desc = data.get('beschreibung', 'Keine Beschreibung')[:60]
                if len(desc) == 60:
                    desc += "..."
                print(f"  - {name}: {desc}")
    
    def export_results(self, text: str, output_file: str, format: str = 'json'):
        """Exportiert Analyse-Ergebnisse in eine Datei"""
        result = self.matcher.analyze_text(text)
        result_dict = result.to_dict()
        
        try:
            if format == 'json':
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result_dict, f, ensure_ascii=False, indent=2)
            elif format == 'yaml':
                with open(output_file, 'w', encoding='utf-8') as f:
                    yaml.dump(result_dict, f, allow_unicode=True)
            else:
                print(f"{Fore.RED}Unbekanntes Format: {format}{Style.RESET_ALL}")
                return
            
            print(f"{Fore.GREEN}Ergebnisse exportiert nach: {output_file}{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}Fehler beim Export: {e}{Style.RESET_ALL}")
    
    def analyze_cosd(self, texts: List[str], verbose: bool = False, export_file: Optional[str] = None):
        """Analysiert eine Textsequenz mit CoSD"""
        if not self.cosd_analyzer:
            print(f"{Fore.RED}CoSD-Modul nicht verfügbar!{Style.RESET_ALL}")
            return
        
        if len(texts) < 2:
            print(f"{Fore.YELLOW}CoSD-Analyse benötigt mindestens 2 Texte!{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.CYAN}CoSD-Analyse für {len(texts)} Texte{Style.RESET_ALL}")
        print("=" * 60)
        
        try:
            # Führe CoSD-Analyse durch
            result = self.cosd_analyzer.analyze_drift(texts)
            
            # Ausgabe
            self._print_cosd_result(result, verbose)
            
            # Export falls gewünscht
            if export_file:
                self._export_cosd_result(result, export_file)
                
        except Exception as e:
            print(f"{Fore.RED}Fehler bei CoSD-Analyse: {e}{Style.RESET_ALL}")
    
    def _print_cosd_result(self, result, verbose: bool = False):
        """Gibt CoSD-Analyse-Ergebnis formatiert aus"""
        # Risk Assessment
        risk = result.risk_assessment
        risk_colors = {
            'low': Fore.GREEN,
            'medium': Fore.YELLOW,
            'high': Fore.MAGENTA,
            'critical': Fore.RED
        }
        risk_color = risk_colors.get(risk['risk_level'], Fore.WHITE)
        
        print(f"\n{risk_color}CoSD Risk-Level: {risk['risk_level'].upper()}{Style.RESET_ALL}")
        print(f"Gesamt-Risiko-Score: {risk['total_risk_score']:.2f}")
        
        # Drift-Metriken
        print(f"\n{Fore.CYAN}Drift-Metriken:{Style.RESET_ALL}")
        print(f"  - Durchschnittliche Geschwindigkeit: {result.drift_velocity['average_velocity']:.3f}")
        print(f"  - Pfadlänge: {result.drift_path['path_length']:.3f}")
        print(f"  - Krümmung: {result.drift_path['curvature']:.3f}")
        
        # Resonanzmuster
        if result.resonance_patterns:
            print(f"\n{Fore.CYAN}Resonanzmuster:{Style.RESET_ALL}")
            chains = [p for p in result.resonance_patterns if p['type'] == 'resonance_chain']
            strong = [p for p in result.resonance_patterns if p['type'] == 'strong_resonance']
            print(f"  - Starke Resonanzen: {len(strong)}")
            print(f"  - Resonanzketten: {len(chains)}")
            
            if verbose and chains:
                print("  Resonanzketten:")
                for chain in chains[:3]:
                    print(f"    - Länge {chain['chain_length']}, Dichte {chain['density']:.2f}")
        
        # Emergente Cluster
        if result.emergent_clusters:
            print(f"\n{Fore.CYAN}Emergente Cluster:{Style.RESET_ALL}")
            for cluster in result.emergent_clusters[:5]:
                print(f"  - {len(cluster.marker_names)} Marker, Kohäsion: {cluster.cohesion_score:.2f}")
                if verbose:
                    print(f"    Marker: {', '.join(list(cluster.marker_names)[:5])}")
        
        # Empfehlungen
        print(f"\n{Fore.CYAN}Empfehlungen:{Style.RESET_ALL}")
        for rec in risk['recommendations']:
            print(f"  • {rec}")
    
    def _export_cosd_result(self, result, output_file: str):
        """Exportiert CoSD-Ergebnis in Datei"""
        try:
            result_dict = result.to_dict()
            
            if output_file.endswith('.yaml'):
                with open(output_file, 'w', encoding='utf-8') as f:
                    yaml.dump(result_dict, f, allow_unicode=True)
            else:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result_dict, f, ensure_ascii=False, indent=2)
            
            print(f"{Fore.GREEN}CoSD-Ergebnis exportiert nach: {output_file}{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}Fehler beim Export: {e}{Style.RESET_ALL}")


def main():
    """Hauptfunktion für CLI"""
    parser = argparse.ArgumentParser(
        description='Marker-basierte Textanalyse für psychologische Muster',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Text direkt analysieren
  python marker_cli.py -t "Das hast du dir nur eingebildet."
  
  # Datei analysieren
  python marker_cli.py -f chat_log.txt
  
  # Verzeichnis analysieren
  python marker_cli.py -d ./chats --pattern "*.txt"
  
  # Alle Marker auflisten
  python marker_cli.py --list-markers
  
  # Ergebnis exportieren
  python marker_cli.py -t "Text..." --export result.json
  
  # CoSD-Analyse durchführen
  python marker_cli.py --cosd-analyze -f text1.txt text2.txt text3.txt
        """
    )
    
    # Analyse-Optionen
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('-t', '--text', help='Text direkt analysieren')
    input_group.add_argument('-f', '--file', help='Textdatei analysieren')
    input_group.add_argument('-d', '--directory', help='Alle Dateien in Verzeichnis analysieren')
    
    # Weitere Optionen
    parser.add_argument('-p', '--pattern', default='*.txt', 
                       help='Datei-Pattern für Verzeichnis-Analyse (Standard: *.txt)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Ausführliche Ausgabe mit allen Treffern')
    parser.add_argument('--list-markers', action='store_true',
                       help='Liste alle verfügbaren Marker auf')
    parser.add_argument('--category', help='Filtere Marker nach Kategorie')
    parser.add_argument('--export', help='Exportiere Ergebnisse in Datei')
    parser.add_argument('--format', choices=['json', 'yaml'], default='json',
                       help='Export-Format (Standard: json)')
    
    # CoSD-spezifische Optionen
    parser.add_argument('--cosd-analyze', action='store_true',
                       help='Führe CoSD-Analyse für eine Textsequenz durch')
    parser.add_argument('--drift-timeframe', type=int, default=5,
                       help='Zeitabstand zwischen Texten in Minuten (Standard: 5)')
    parser.add_argument('--resonance-threshold', type=float, default=0.7,
                       help='Schwellenwert für Resonanz-Erkennung (Standard: 0.7)')
    parser.add_argument('files', nargs='*',
                       help='Dateien für CoSD-Analyse (bei --cosd-analyze)')
    
    args = parser.parse_args()
    
    # Erstelle CLI-Instanz
    cli = MarkerCLI()
    
    # Führe gewünschte Aktion aus
    if args.list_markers:
        cli.list_markers(args.category)
    elif args.cosd_analyze:
        # CoSD-Analyse
        if not args.files:
            print(f"{Fore.RED}Fehler: CoSD-Analyse benötigt Dateien!{Style.RESET_ALL}")
            parser.print_help()
            sys.exit(1)
        
        # Lade Texte aus Dateien
        texts = []
        for file_path in args.files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
            except Exception as e:
                print(f"{Fore.RED}Fehler beim Lesen von {file_path}: {e}{Style.RESET_ALL}")
                sys.exit(1)
        
        # Konfiguriere CoSD-Analyzer falls vorhanden
        if cli.cosd_analyzer:
            cli.cosd_analyzer.config['resonance_threshold'] = args.resonance_threshold
        
        # Führe Analyse durch
        cli.analyze_cosd(texts, args.verbose, args.export)
        
    elif args.text:
        if args.export:
            cli.export_results(args.text, args.export, args.format)
        else:
            cli.analyze_text(args.text, args.verbose)
    elif args.file:
        cli.analyze_file(args.file, args.verbose)
    elif args.directory:
        cli.analyze_directory(args.directory, args.pattern, args.verbose)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main() 