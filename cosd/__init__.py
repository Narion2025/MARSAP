#!/usr/bin/env python3
"""
CoSD - Co-emergent Semantic Drift Module für MARSAP

Dieses Modul implementiert die Analyse von semantischer Drift und Marker-Clustern
nach dem Co-emergenten Semantic Drift (CoSD)-Verfahren.

Hauptkomponenten:
- drift_analyzer: Kernfunktionalität für Drift-Analyse
- cost_vector_math: Mathematische Operationen für Vektor-Berechnungen
- semantic_marker_interface: Integration mit bestehendem Marker-System

Verwendung:
    from marsap.cosd import CoSDAnalyzer
    
    analyzer = CoSDAnalyzer()
    result = analyzer.analyze_drift(text_sequence, markers)
"""

__version__ = "1.0.0"
__author__ = "MARSAP CoSD Integration"

# Zentrale Imports für einfachen Zugriff
from .drift_analyzer import CoSDAnalyzer, DriftAnalysisResult
from .cost_vector_math import CoSDVector, calculate_drift_velocity
from .semantic_marker_interface import CoSDMarkerMatcher, SemanticCluster

__all__ = [
    "CoSDAnalyzer",
    "DriftAnalysisResult", 
    "CoSDVector",
    "calculate_drift_velocity",
    "CoSDMarkerMatcher",
    "SemanticCluster"
] 