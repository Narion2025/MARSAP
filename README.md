# MARSAP - Marker Analysis & Recognition System for Adaptive Patterns

## 🔍 Semantisch-psychologischer Resonanz- und Manipulations-Detektor mit CoSD-Integration

MARSAP ist ein fortschrittliches System zur Erkennung psychologischer Kommunikationsmuster, manipulativer Techniken und emotionaler Dynamiken in Texten. Es nutzt einen umfangreichen Katalog von 72+ Markern zur Live-Analyse von Kommunikationsmustern.

**NEU: CoSD-Modul (Co-emergent Semantic Drift)** - Analysiert semantische Drift und emergente Bedeutungsmuster in Textsequenzen.

## 🎯 Features

- **72 psychologische Marker** für Manipulation, emotionale Dynamik und Beziehungsmuster
- **Vierstufiges Risk-Level-System** (🟢 Grün, 🟡 Gelb, 🟠 Blinkend, 🔴 Rot)
- **Mehrere Interfaces**: CLI, Python-API und REST-API
- **Real-time Analyse** für Chat-Monitoring
- **Batch-Processing** für Archiv-Scans
- **Modulare Erweiterbarkeit** für neue Marker
- **CoSD-Analyse** für semantische Drift-Erkennung und Resonanzmuster
- **Emergenz-Detektion** für neue Bedeutungsmuster

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

### Docker

```bash
# Container bauen und starten
docker compose up --build

# API im Browser testen
curl http://localhost:5000/health
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

### CoSD-Analyse (NEU)
```bash
# Textsequenz analysieren
python3 marker_cli.py --cosd-analyze text1.txt text2.txt text3.txt

# Mit erweiterten Optionen
python3 marker_cli.py --cosd-analyze --resonance-threshold 0.8 --export cosd_result.json text*.txt
```

```python
# Python API für CoSD
from cosd import CoSDAnalyzer

analyzer = CoSDAnalyzer()
result = analyzer.analyze_drift(["Text 1", "Text 2", "Text 3"])
print(f"Drift-Geschwindigkeit: {result.drift_velocity['average_velocity']:.3f}")
print(f"Risk-Level: {result.risk_assessment['risk_level']}")
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
