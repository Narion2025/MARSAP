# MARSAP - Marker Analysis & Recognition System for Adaptive Patterns

## 🔍 Semantisch-psychologischer Resonanz- und Manipulations-Detektor

MARSAP ist ein fortschrittliches System zur Erkennung psychologischer Kommunikationsmuster, manipulativer Techniken und emotionaler Dynamiken in Texten. Es nutzt einen umfangreichen Katalog von 72+ Markern zur Live-Analyse von Kommunikationsmustern.

## 🎯 Features

- **72 psychologische Marker** für Manipulation, emotionale Dynamik und Beziehungsmuster
- **Vierstufiges Risk-Level-System** (🟢 Grün, 🟡 Gelb, 🟠 Blinkend, 🔴 Rot)
- **Mehrere Interfaces**: CLI, Python-API und REST-API
- **Real-time Analyse** für Chat-Monitoring
- **Batch-Processing** für Archiv-Scans
- **Modulare Erweiterbarkeit** für neue Marker

## 🚀 Quick Start

```bash
# 1. Repository klonen
git clone https://github.com/Narion2025/MARSAP.git
cd MARSAP

# 2. Dependencies installieren
pip install -r requirements.txt

# 3. System testen
python3 marker_matcher.py

# 4. Text analysieren
python3 marker_cli.py -t "Das hast du dir nur eingebildet."
```

## 💻 Verwendung

### Command Line Interface
```bash
# Einzeltext analysieren
python3 marker_cli.py -t "Zu analysierender Text"

# Datei analysieren
python3 marker_cli.py -f chat_log.txt

# Alle Marker auflisten
python3 marker_cli.py --list-markers
```

### Python API
```python
from marker_matcher import MarkerMatcher

matcher = MarkerMatcher()
result = matcher.analyze_text("Dein Text hier...")
print(f"Risk-Level: {result.risk_level}")
```

## 📊 Erkannte Muster

### Manipulationstechniken
- **GASLIGHTING** - Realitätsverzerrung und Selbstzweifel
- **LOVE_BOMBING** - Überwältigende Zuneigung als Manipulation  
- **BLAME_SHIFT** - Verantwortung auf andere verschieben

### Emotionale Dynamiken
- **AMBIVALENCE** - Hin- und hergerissen zwischen Optionen
- **ESCALATION** - Konflikteskalation

## 🔧 Erweiterte Nutzung

Siehe [MARKER_SYSTEM_README.md](MARKER_SYSTEM_README.md) für detaillierte Dokumentation.

## ⚠️ Disclaimer

MARSAP ist ein Hilfsmittel zur Textanalyse und ersetzt keine professionelle psychologische Beratung.

---

**MARSAP** - *Marker Analysis & Recognition System for Adaptive Patterns*
