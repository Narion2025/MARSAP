# 🔍 Marker Master System

## Semantisch-psychologischer Resonanz- und Manipulations-Detektor

Ein modulares System zur Erkennung psychologischer Kommunikationsmuster, manipulativer Techniken und emotionaler Dynamiken in Texten.

## 📋 Übersicht

Das Marker Master System analysiert Texte auf Basis eines umfangreichen Katalogs psychologischer Marker. Es erkennt subtile Manipulationstechniken, emotionale Muster und Kommunikationsdynamiken.

### Kernkomponenten

1. **marker_master_export.yaml/json** - Zentrale Marker-Datenbank mit 72+ Mustern
2. **marker_matcher.py** - Core-Engine für Pattern-Matching und semantische Analyse
3. **marker_api.py** - REST-API für Integration in andere Systeme
4. **marker_cli.py** - Command-Line-Interface für direkte Nutzung

## 🚀 Installation

```bash
# 1. Dependencies installieren
pip install -r requirements.txt

# 2. System testen
python3 marker_matcher.py
```

## 💻 Verwendung

### CLI - Kommandozeile

```bash
# Text direkt analysieren
python3 marker_cli.py -t "Das hast du dir nur eingebildet."

# Datei analysieren
python3 marker_cli.py -f chat_log.txt

# Verzeichnis mit allen Chats analysieren
python3 marker_cli.py -d ./chats --pattern "*.txt"

# Alle verfügbaren Marker auflisten
python3 marker_cli.py --list-markers

# Ergebnis exportieren
python3 marker_cli.py -t "Dein Text..." --export result.json
```

### Python API

```python
from marker_matcher import MarkerMatcher

# Matcher initialisieren
matcher = MarkerMatcher()

# Text analysieren
result = matcher.analyze_text("Ich habe nie gesagt, dass ich mitkomme.")

# Ergebnis auswerten
print(f"Risk-Level: {result.risk_level}")
print(f"Gefundene Marker: {len(result.gefundene_marker)}")

for match in result.gefundene_marker:
    print(f"- {match.marker_name}: {match.matched_text}")
```

### REST API

```bash
# Server starten
python3 marker_api.py

# Text analysieren
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Du bist zu empfindlich."}'

# Batch-Analyse
curl -X POST http://localhost:5000/analyze_batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Text 1", "Text 2"]}'

# Alle Marker abrufen
curl http://localhost:5000/markers
```

## 🎯 Risk-Level System

Das System bewertet Texte mit einem vierstufigen Risiko-System:

- 🟢 **Grün**: Kein oder nur unkritischer Marker
- 🟡 **Gelb**: 1-2 moderate Marker, erste Drift erkennbar
- 🟠 **Blinkend**: 3+ Marker oder ein Hochrisiko-Marker, klare Drift/Manipulation
- 🔴 **Rot**: Hochrisiko-Kombination, massive Drift/Manipulation

## 📊 Erkannte Muster

### Manipulationstechniken
- **GASLIGHTING**: Realitätsverzerrung und Selbstzweifel induzieren
- **LOVE_BOMBING**: Überwältigende Zuneigung als Manipulationsmittel
- **BLAME_SHIFT**: Verantwortung auf andere verschieben
- **SILENT_TREATMENT**: Schweigen als Bestrafung

### Emotionale Dynamiken
- **AMBIVALENCE**: Hin- und hergerissen zwischen Optionen
- **ESCALATION**: Konflikteskalation
- **TERMINATION**: Beziehungsabbruch-Signale
- **AROUSAL**: Emotionale Erregung

### Beziehungsmuster
- **PROJECTIVE_IDENTIFICATION**: Eigene Gefühle auf andere projizieren
- **DRAMA_TRIANGLE**: Opfer-Täter-Retter-Dynamik
- **ISOLATION**: Soziale Isolation fördern
- **COMPARISON_GHOST**: Vergleiche mit Ex-Partnern

## 🔧 Erweiterung

### Neue Marker hinzufügen

1. Bearbeite `marker_master_export.yaml`:

```yaml
- marker: NEUER_MARKER
  beschreibung: "Beschreibung des Musters"
  beispiele:
    - "Beispielsatz 1"
    - "Beispielsatz 2"
  kategorie: MANIPULATION
  tags: [neu, custom]
  risk_score: 3
```

2. Regeneriere die Master-Datei:

```bash
python3 create_marker_master.py
```

### Semantische Detektoren

Für komplexere Muster können Python-Detektoren im Ordner `SEMANTIC_DETECTORS_PYTHO` hinzugefügt werden.

## 📈 Statistiken

Das System enthält aktuell:
- 72 eindeutige Marker
- 1000+ Beispiel-Patterns
- 20 semantische Detektoren
- 4 Risiko-Stufen

## 🛠️ Troubleshooting

### Fehlende Dependencies
```bash
pip install -r requirements.txt
```

### Marker werden nicht erkannt
- Prüfe ob der Text die exakten Beispiele enthält
- Verwende `--verbose` für detaillierte Ausgabe
- Überprüfe die Groß-/Kleinschreibung

### Performance-Probleme
- Reduziere die Anzahl der Marker
- Nutze Batch-Processing für viele Dateien
- Deaktiviere semantische Detektoren für Speed

## 📝 Lizenz

Dieses System ist für Forschungs- und Bildungszwecke gedacht. Bei kommerzieller Nutzung bitte Rücksprache halten.

## 🤝 Beitragen

Neue Marker, Verbesserungen und Fehlerkorrekturen sind willkommen! 

1. Fork das Repository
2. Erstelle einen Feature-Branch
3. Committe deine Änderungen
4. Push zum Branch
5. Erstelle einen Pull Request

## ⚠️ Disclaimer

Dieses System ist ein Hilfsmittel zur Textanalyse und ersetzt keine professionelle psychologische Beratung. Die Ergebnisse sollten immer im Kontext interpretiert werden. 