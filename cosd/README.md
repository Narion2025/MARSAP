# CoSD - Co-emergent Semantic Drift Module

## Überblick

Das CoSD-Modul erweitert MARSAP um die Fähigkeit, semantische Drift und emergente Bedeutungsmuster in Textsequenzen zu analysieren. Es erkennt, wie sich die Bedeutung und emotionale Färbung von Kommunikation über Zeit verändert.

## Kernkonzepte

### Semantische Drift
- **Definition**: Graduelle Verschiebung von Bedeutung und emotionaler Färbung in einer Kommunikationssequenz
- **Messung**: Vektorbasierte Analyse im mehrdimensionalen semantischen Raum
- **Anwendung**: Erkennung von Manipulationsmustern und emotionaler Eskalation

### Resonanzmuster
- **Starke Kopplung**: Semantisch ähnliche Texte mit hoher Übereinstimmung
- **Resonanzketten**: Sequenzen von gekoppelten Bedeutungsmustern
- **Phasenausrichtung**: Synchronisation von emotionalen oder thematischen Mustern

### Emergente Cluster
- **Neue Kombinationen**: Erkennung bisher nicht beobachteter Marker-Kombinationen
- **Kohäsion**: Maß für die Zusammengehörigkeit von Markern in einem Cluster
- **Zeitliche Stabilität**: Persistenz von Mustern über mehrere Texte

## Architektur

```
cosd/
├── cost_vector_math.py      # Mathematische Grundoperationen
├── drift_analyzer.py        # Hauptanalyse-Logik
├── semantic_marker_interface.py  # Integration mit Marker-System
└── test_cosd.py            # Unit-Tests
```

### Module im Detail

#### cost_vector_math.py
- `CoSDVector`: Datenstruktur für semantische Vektoren
- `VectorOperations`: Distanz- und Ähnlichkeitsberechnungen
- `calculate_drift_velocity()`: Drift-Geschwindigkeitsberechnung
- `calculate_resonance_coupling()`: Resonanz-Analyse
- `cluster_vectors()`: Semantisches Clustering

#### drift_analyzer.py
- `CoSDAnalyzer`: Hauptklasse für Drift-Analyse
- `DriftAnalysisResult`: Ergebnis-Datenstruktur
- Risikobewertung und Empfehlungsgenerierung

#### semantic_marker_interface.py
- `MarkerVectorizer`: Konvertiert Marker zu Vektoren
- `SemanticCluster`: Repräsentiert Marker-Gruppen
- `CoSDMarkerMatcher`: Integration mit bestehendem System

## Verwendung

### CLI
```bash
# Grundlegende Analyse
python marker_cli.py --cosd-analyze file1.txt file2.txt file3.txt

# Mit Konfiguration
python marker_cli.py --cosd-analyze --resonance-threshold 0.8 \
                    --drift-timeframe 10 --export result.json *.txt
```

### Python API
```python
from cosd import CoSDAnalyzer

# Initialisierung
analyzer = CoSDAnalyzer(
    marker_data_path="path/to/markers.yaml",
    config={
        'resonance_threshold': 0.8,
        'cluster_distance_threshold': 0.5
    }
)

# Analyse
texts = ["Text 1", "Text 2", "Text 3"]
result = analyzer.analyze_drift(texts)

# Ergebnisse
print(f"Risk-Level: {result.risk_assessment['risk_level']}")
print(f"Drift-Geschwindigkeit: {result.drift_velocity['average_velocity']}")
print(f"Emergente Cluster: {len(result.emergent_clusters)}")
```

### REST API
```bash
# Status prüfen
curl http://localhost:5000/api/cosd/status

# Analyse durchführen
curl -X POST http://localhost:5000/api/cosd/analyze \
     -H "Content-Type: application/json" \
     -d '{
       "texts": ["Text 1", "Text 2", "Text 3"],
       "config": {
         "resonance_threshold": 0.8
       }
     }'
```

## Metriken und Interpretation

### Risk-Level
- **low** (grün): Stabile Kommunikation, geringe Drift
- **medium** (gelb): Moderate Verschiebungen erkennbar
- **high** (orange): Signifikante semantische Drift
- **critical** (rot): Extreme Verschiebungen, mögliche Manipulation

### Drift-Velocity
- **< 0.1**: Sehr langsame Veränderung
- **0.1 - 0.3**: Normale Konversationsdynamik
- **0.3 - 0.5**: Beschleunigte Verschiebung
- **> 0.5**: Rapide semantische Veränderung

### Resonanz-Patterns
- **Einzelresonanzen**: Punktuelle Übereinstimmungen
- **Resonanzketten**: Systematische Muster über Zeit
- **Kohäsion > 0.8**: Starke thematische Bindung

## Konfiguration

```python
config = {
    'min_text_length': 10,              # Minimale Textlänge
    'resonance_threshold': 0.7,         # Schwelle für Resonanz
    'cluster_distance_threshold': 0.5,  # Max. Distanz für Cluster
    'drift_velocity_window': 3,         # Fenster für Geschwindigkeit
    'emergence_detection_sensitivity': 0.8,  # Emergenz-Sensitivität
    'risk_thresholds': {
        'low': 0.3,
        'medium': 0.6,
        'high': 0.85
    }
}
```

## Technische Details

### Vektorisierung
- Verwendet Token-basierte Repräsentation aus Marker-Definitionen
- Dimensionalität abhängig von Anzahl einzigartiger Tokens
- Normalisierte Vektoren für Vergleichbarkeit

### Distanzmetriken
- **Kosinus-Ähnlichkeit**: Hauptmetrik für semantische Nähe
- **Euklidische Distanz**: Für Pfadberechnungen
- **Manhattan-Distanz**: Alternative Metrik

### Clustering
- Hierarchisches Clustering mit konfigurierbarer Distanzschwelle
- Unterstützt sowohl euklidische als auch Kosinus-Distanz

## Tests

```bash
# Unit-Tests ausführen
cd MARSAP
python -m cosd.test_cosd
```

## Abhängigkeiten

- numpy >= 1.24.0
- spacy >= 3.6.0
- Bestehende MARSAP-Installation

## Limitationen

- Benötigt mindestens 2 Texte für Drift-Analyse
- Performance abhängig von Textlänge und Marker-Anzahl
- Emergenz-Detektion basiert auf historischen Daten

## Weiterentwicklung

Mögliche Erweiterungen:
- Integration mit Spacy für erweiterte NLP-Features
- Machine Learning für Mustervorhersage
- Echtzeit-Streaming-Analyse
- Visualisierung der Drift-Pfade 