#!/usr/bin/env python3
"""
CoSD Vector Mathematics - Mathematische Operationen für semantische Drift-Berechnungen

Dieses Modul implementiert alle mathematischen Grundoperationen für die
Co-emergente Semantic Drift (CoSD) Analyse, einschließlich Vektor-Berechnungen,
Distanzmetriken und Resonanz-Algorithmen.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class CoSDVector:
    """
    Repräsentiert einen semantischen Vektor im CoSD-Raum.
    
    Attributes:
        dimensions: Numpy-Array der Vektor-Dimensionen
        timestamp: Zeitstempel der Vektor-Erstellung
        marker_weights: Dict mit Marker-Namen und deren Gewichtungen
        metadata: Zusätzliche Metadaten
    """
    dimensions: np.ndarray
    timestamp: datetime
    marker_weights: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validierung und Normalisierung nach Initialisierung."""
        if not isinstance(self.dimensions, np.ndarray):
            self.dimensions = np.array(self.dimensions, dtype=np.float64)
    
    @property
    def magnitude(self) -> float:
        """Berechnet die Magnitude (Länge) des Vektors."""
        return np.linalg.norm(self.dimensions)
    
    def normalize(self) -> 'CoSDVector':
        """Gibt einen normalisierten Vektor zurück (Länge = 1)."""
        mag = self.magnitude
        if mag == 0:
            return self
        normalized_dims = self.dimensions / mag
        return CoSDVector(
            dimensions=normalized_dims,
            timestamp=self.timestamp,
            marker_weights=self.marker_weights.copy(),
            metadata=self.metadata.copy()
        )
    
    def dot_product(self, other: 'CoSDVector') -> float:
        """Berechnet das Skalarprodukt mit einem anderen Vektor."""
        return np.dot(self.dimensions, other.dimensions)


class VectorOperations:
    """Sammlung von Vektor-Operationen für CoSD-Berechnungen."""
    
    @staticmethod
    def cosine_similarity(vec1: CoSDVector, vec2: CoSDVector) -> float:
        """
        Berechnet die Kosinus-Ähnlichkeit zwischen zwei Vektoren.
        
        Args:
            vec1: Erster Vektor
            vec2: Zweiter Vektor
            
        Returns:
            float: Ähnlichkeit zwischen -1 und 1 (1 = identisch)
        """
        if vec1.magnitude == 0 or vec2.magnitude == 0:
            return 0.0
        
        return vec1.dot_product(vec2) / (vec1.magnitude * vec2.magnitude)
    
    @staticmethod
    def euclidean_distance(vec1: CoSDVector, vec2: CoSDVector) -> float:
        """
        Berechnet die euklidische Distanz zwischen zwei Vektoren.
        
        Args:
            vec1: Erster Vektor
            vec2: Zweiter Vektor
            
        Returns:
            float: Euklidische Distanz (>= 0)
        """
        return np.linalg.norm(vec1.dimensions - vec2.dimensions)
    
    @staticmethod
    def manhattan_distance(vec1: CoSDVector, vec2: CoSDVector) -> float:
        """
        Berechnet die Manhattan-Distanz zwischen zwei Vektoren.
        
        Args:
            vec1: Erster Vektor
            vec2: Zweiter Vektor
            
        Returns:
            float: Manhattan-Distanz (>= 0)
        """
        return np.sum(np.abs(vec1.dimensions - vec2.dimensions))


def calculate_drift_velocity(
    vectors: List[CoSDVector],
    time_window: Optional[int] = None
) -> Dict[str, Union[float, List[float]]]:
    """
    Berechnet die Drift-Geschwindigkeit über eine Sequenz von Vektoren.
    
    Args:
        vectors: Liste von CoSDVector-Objekten in chronologischer Reihenfolge
        time_window: Optionales Zeitfenster für gleitende Durchschnitte
        
    Returns:
        Dict mit Drift-Metriken:
        - average_velocity: Durchschnittliche Geschwindigkeit
        - instantaneous_velocities: Liste von Momentangeschwindigkeiten
        - acceleration: Beschleunigung der semantischen Drift
        - total_distance: Gesamtstrecke der semantischen Bewegung
    """
    if len(vectors) < 2:
        return {
            'average_velocity': 0.0,
            'instantaneous_velocities': [],
            'acceleration': 0.0,
            'total_distance': 0.0
        }
    
    velocities = []
    total_distance = 0.0
    
    # Berechne Momentangeschwindigkeiten
    for i in range(1, len(vectors)):
        dist = VectorOperations.euclidean_distance(vectors[i-1], vectors[i])
        time_delta = (vectors[i].timestamp - vectors[i-1].timestamp).total_seconds()
        
        if time_delta > 0:
            velocity = dist / time_delta
            velocities.append(velocity)
            total_distance += dist
    
    # Berechne Durchschnitt
    avg_velocity = np.mean(velocities) if velocities else 0.0
    
    # Berechne Beschleunigung (Änderung der Geschwindigkeit)
    acceleration = 0.0
    if len(velocities) >= 2:
        velocity_changes = np.diff(velocities)
        acceleration = np.mean(velocity_changes)
    
    return {
        'average_velocity': float(avg_velocity),
        'instantaneous_velocities': velocities,
        'acceleration': float(acceleration),
        'total_distance': float(total_distance)
    }


def calculate_resonance_coupling(
    vec1: CoSDVector,
    vec2: CoSDVector,
    coupling_threshold: float = 0.7
) -> Dict[str, float]:
    """
    Berechnet die Resonanz-Kopplung zwischen zwei semantischen Vektoren.
    
    Args:
        vec1: Erster Vektor
        vec2: Zweiter Vektor
        coupling_threshold: Schwellenwert für starke Kopplung
        
    Returns:
        Dict mit Kopplungsmetriken:
        - coupling_strength: Stärke der Kopplung (0-1)
        - phase_alignment: Phasen-Ausrichtung
        - resonance_factor: Resonanz-Faktor
        - is_strongly_coupled: Boolean für starke Kopplung
    """
    similarity = VectorOperations.cosine_similarity(vec1, vec2)
    
    # Berechne Phasen-Ausrichtung basierend auf Winkel
    angle = np.arccos(np.clip(similarity, -1.0, 1.0))
    phase_alignment = 1.0 - (angle / np.pi)
    
    # Resonanz-Faktor berücksichtigt auch Magnitude-Verhältnis
    magnitude_ratio = min(vec1.magnitude, vec2.magnitude) / max(vec1.magnitude, vec2.magnitude)
    resonance_factor = phase_alignment * magnitude_ratio
    
    # Kopplungsstärke kombiniert mehrere Faktoren
    coupling_strength = (similarity + phase_alignment + resonance_factor) / 3.0
    
    return {
        'coupling_strength': float(coupling_strength),
        'phase_alignment': float(phase_alignment),
        'resonance_factor': float(resonance_factor),
        'is_strongly_coupled': coupling_strength >= coupling_threshold
    }


def cluster_vectors(
    vectors: List[CoSDVector],
    distance_threshold: float = 0.5,
    method: str = 'euclidean'
) -> List[List[int]]:
    """
    Clustert Vektoren basierend auf semantischer Ähnlichkeit.
    
    Args:
        vectors: Liste von CoSDVector-Objekten
        distance_threshold: Maximale Distanz für Cluster-Zugehörigkeit
        method: Distanzmetrik ('euclidean' oder 'cosine')
        
    Returns:
        Liste von Listen mit Vektor-Indizes pro Cluster
    """
    if not vectors:
        return []
    
    # Initialisiere Cluster
    clusters = []
    assigned = set()
    
    # Wähle Distanzfunktion
    if method == 'euclidean':
        dist_func = VectorOperations.euclidean_distance
    else:  # cosine
        dist_func = lambda v1, v2: 1.0 - VectorOperations.cosine_similarity(v1, v2)
    
    # Einfacher hierarchischer Clustering-Algorithmus
    for i, vec in enumerate(vectors):
        if i in assigned:
            continue
            
        # Starte neuen Cluster
        cluster = [i]
        assigned.add(i)
        
        # Füge ähnliche Vektoren hinzu
        for j in range(i + 1, len(vectors)):
            if j in assigned:
                continue
                
            # Prüfe Distanz zu allen Cluster-Mitgliedern
            max_dist = max(dist_func(vec, vectors[idx]) for idx in cluster)
            
            if max_dist <= distance_threshold:
                cluster.append(j)
                assigned.add(j)
        
        clusters.append(cluster)
    
    return clusters


def calculate_semantic_drift_path(
    vectors: List[CoSDVector]
) -> Dict[str, any]:
    """
    Berechnet den vollständigen semantischen Drift-Pfad.
    
    Args:
        vectors: Chronologisch geordnete Liste von Vektoren
        
    Returns:
        Dict mit Pfad-Metriken:
        - path_length: Gesamtlänge des Pfades
        - curvature: Krümmung des Pfades
        - drift_direction: Hauptrichtung der Drift
        - stability_zones: Bereiche mit geringer Bewegung
    """
    if len(vectors) < 2:
        return {
            'path_length': 0.0,
            'curvature': 0.0,
            'drift_direction': np.zeros(vectors[0].dimensions.shape) if vectors else None,
            'stability_zones': []
        }
    
    # Berechne Pfadlänge
    path_length = sum(
        VectorOperations.euclidean_distance(vectors[i], vectors[i+1])
        for i in range(len(vectors) - 1)
    )
    
    # Berechne Drift-Richtung (Endpunkt - Startpunkt)
    drift_direction = vectors[-1].dimensions - vectors[0].dimensions
    
    # Berechne Krümmung (Abweichung von gerader Linie)
    direct_distance = VectorOperations.euclidean_distance(vectors[0], vectors[-1])
    curvature = (path_length / direct_distance - 1.0) if direct_distance > 0 else 0.0
    
    # Identifiziere Stabilitätszonen (geringe Bewegung)
    stability_zones = []
    stability_threshold = 0.1  # Anpassbar
    
    for i in range(1, len(vectors)):
        dist = VectorOperations.euclidean_distance(vectors[i-1], vectors[i])
        if dist < stability_threshold:
            if not stability_zones or stability_zones[-1]['end'] != i-1:
                stability_zones.append({'start': i-1, 'end': i})
            else:
                stability_zones[-1]['end'] = i
    
    return {
        'path_length': float(path_length),
        'curvature': float(curvature),
        'drift_direction': drift_direction.tolist() if drift_direction is not None else None,
        'stability_zones': stability_zones
    } 