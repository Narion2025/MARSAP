# 🚀 MARSAP - Einfache Anleitung

## Was ist MARSAP?

MARSAP ist ein System, das **psychologische Marker** in Texten erkennt. Es kann manipulatives Verhalten, emotionale Dynamiken und Beziehungsmuster identifizieren.

## 🎯 Was Sie damit machen können:

1. **Chat-Nachrichten analysieren** - Erkennt manipulatives Verhalten
2. **Beziehungstexte prüfen** - Identifiziert toxische Dynamiken  
3. **Eigene Kommunikation reflektieren** - Versteht emotionale Muster
4. **Risiko-Bewertung** - Gibt Warnstufen aus

## 🚀 Schnellstart

### 1. System testen
```bash
python3 test_system.py
```

### 2. Einzelnen Text analysieren
```bash
python3 marker_cli.py --text "Ey, Alter, das reicht jetzt. Ich bin hier raus."
```

### 3. API-Server starten
```bash
python3 marker_api.py --port 5001
```

### 4. Über Web-Interface nutzen
Öffnen Sie: http://localhost:5001

## 📊 Risk-Level System

- 🟢 **Grün**: Keine kritischen Marker
- 🟡 **Gelb**: 1-2 moderate Marker (Vorsicht)
- 🟠 **Blinkend**: 3+ Marker (klare Manipulation)
- 🔴 **Rot**: Hochrisiko-Kombination (massive Manipulation)

## 🔍 Erkannte Muster

### Manipulationstechniken
- **ABBRUCHMARKER**: Gesprächsabbruch als Druckmittel
- **AMBIVALENZMARKER**: Hin- und hergerissen sein
- **GASLIGHTING**: Realitätsverzerrung
- **LOVE_BOMBING**: Überwältigende Zuneigung

### Emotionale Dynamiken
- **ESCALATION**: Konflikteskalation
- **TERMINATION**: Beziehungsabbruch-Signale
- **AROUSAL**: Emotionale Erregung

## 💡 Praktische Beispiele

### Beispiel 1: Abruchmarker
**Text**: "Ey, Alter, das reicht jetzt. Ich bin hier raus."
**Ergebnis**: 🟡 Gelb - ABBRUCHMARKER erkannt

### Beispiel 2: Ambivalenz
**Text**: "Ich bin hin- und hergerissen zwischen Bleiben und Gehen"
**Ergebnis**: 🟡 Gelb - AMBIVALENZMARKER erkannt

### Beispiel 3: Kombination
**Text**: "Du bist zu empfindlich. Ich hab keinen Bock mehr."
**Ergebnis**: 🟠 Blinkend - Mehrere Marker erkannt

## 🛠️ Drei Wege MARSAP zu nutzen

### 1. Python-API (Programmierer)
```python
from marker_matcher import MarkerMatcher

matcher = MarkerMatcher()
result = matcher.analyze_text("Dein Text hier")
print(f"Risk-Level: {result.risk_level}")
```

### 2. Kommandozeile (Schnelltest)
```bash
python3 marker_cli.py --text "Text zum Analysieren"
```

### 3. Web-API (Integration)
```bash
curl -X POST http://localhost:5001/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Text zum Analysieren"}'
```

## 📁 Dateien verstehen

- `marker_matcher.py` - Hauptlogik
- `marker_cli.py` - Kommandozeilen-Tool
- `marker_api.py` - Web-Server
- `marker_master_export.yaml` - Alle Marker-Definitionen
- `test_system.py` - Test-Skript

## 🔧 Probleme lösen

### Port 5000 belegt
```bash
python3 marker_api.py --port 5001
```

### Marker werden nicht erkannt
- Verwende exakte Beispiele aus der Dokumentation
- Prüfe Groß-/Kleinschreibung
- Nutze `--verbose` für Details

### Fehlende Dependencies
```bash
pip install -r requirements.txt
```

## 🎯 Nächste Schritte

1. **Testen Sie das System** mit `python3 test_system.py`
2. **Analysieren Sie eigene Texte** mit der CLI
3. **Starten Sie den API-Server** für Web-Integration
4. **Schauen Sie sich die Marker an** in `marker_master_export.yaml`

## 💭 Wichtiger Hinweis

MARSAP ist ein **Hilfsmittel zur Textanalyse** und ersetzt keine professionelle psychologische Beratung. Die Ergebnisse sollten immer im Kontext interpretiert werden.

## 🆘 Hilfe

- Schauen Sie in `README.md` für detaillierte Dokumentation
- Testen Sie mit `python3 test_system.py`
- Prüfen Sie die Logs für Fehlermeldungen

---

**Viel Erfolg mit MARSAP! 🚀** 