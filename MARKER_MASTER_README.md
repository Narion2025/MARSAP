# Marker Master Export

Diese Dateien enthalten das vollständige Marker-Masterset für den semantisch-psychologischen Resonanz- und Manipulations-Detektor.

## Verwendung

### Import in Python:
```python
import yaml
with open('marker_master_export.yaml', 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)
    
markers = data['markers']
```

### Struktur eines Markers:
- `marker`: Name/ID des Markers
- `beschreibung`: Klartext-Beschreibung
- `beispiele`: Liste typischer Formulierungen
- `kategorie`: Thematische Einordnung
- `tags`: Klassifikations-Tags
- `risk_score`: Risiko-Gewichtung (1-5)
- `semantics_detector`: Optional - Python-Detektor-Datei

### Risiko-Level:
- **Grün**: Kein oder nur unkritischer Marker
- **Gelb**: 1-2 moderate Marker, erste Drift erkennbar
- **Blinkend**: 3+ Marker oder ein Hochrisiko-Marker
- **Rot**: Hochrisiko-Kombination, massive Manipulation

Generiert am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
