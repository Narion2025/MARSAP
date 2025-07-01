#!/usr/bin/env python3
"""
Semantic Marker Interface - Integration zwischen CoSD und MARSAP Marker-System

Dieses Modul stellt die Schnittstelle zwischen dem CoSD-Analysesystem
und dem bestehenden MARSAP-Marker-System her.
"""

import yaml
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging

from .cost_vector_math import CoSDVector, VectorOperations

logger = logging.getLogger(__name__)


@dataclass
class SemanticCluster:
    """
    Repräsentiert eine Gruppe semantisch verwandter Marker.
    
    Attributes:
        cluster_id: Eindeutige ID des Clusters
        marker_names: Set von Marker-Namen im Cluster
        centroid: Zentrum des Clusters als CoSDVector
        cohesion_score: Kohäsionswert des Clusters (0-1)
        emergence_timestamp: Zeitpunkt der Cluster-Entstehung
    """
    cluster_id: str
    marker_names: Set[str]
    centroid: Optional[CoSDVector] = None
    cohesion_score: float = 0.0
    emergence_timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, any] = field(default_factory=dict)
    
    def add_marker(self, marker_name: str):
        """Fügt einen Marker zum Cluster hinzu."""
        self.marker_names.add(marker_name)
    
    def calculate_cohesion(self, marker_vectors: Dict[str, CoSDVector]) -> float:
        """
        Berechnet die Kohäsion des Clusters basierend auf Marker-Vektoren.
        
        Args:
            marker_vectors: Dict mit Marker-Namen und deren Vektoren
            
        Returns:
            float: Kohäsionswert zwischen 0 und 1
        """
        if len(self.marker_names) < 2:
            return 1.0
        
        similarities = []
        marker_list = list(self.marker_names)
        
        for i in range(len(marker_list)):
            for j in range(i + 1, len(marker_list)):
                if marker_list[i] in marker_vectors and marker_list[j] in marker_vectors:
                    sim = VectorOperations.cosine_similarity(
                        marker_vectors[marker_list[i]],
                        marker_vectors[marker_list[j]]
                    )
                    similarities.append(sim)
        
        self.cohesion_score = np.mean(similarities) if similarities else 0.0
        return self.cohesion_score


class MarkerVectorizer:
    """Konvertiert Marker-Definitionen in CoSD-Vektoren."""
    
    def __init__(self, marker_data_path: Optional[str] = None):
        """
        Initialisiert den Vectorizer.
        
        Args:
            marker_data_path: Pfad zur Marker-Datei (yaml)
        """
        self.marker_data = {}
        self.token_index = {}
        self.dimension_count = 0
        
        if marker_data_path:
            self.load_marker_data(marker_data_path)
    
    def load_marker_data(self, file_path: str):
        """
        Lädt Marker-Daten aus einer YAML-Datei.
        
        Args:
            file_path: Pfad zur YAML-Datei
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.marker_data = yaml.safe_load(f)
            
            # Erstelle Token-Index
            self._build_token_index()
            logger.info(f"Loaded marker data with {self.dimension_count} unique tokens")
            
        except Exception as e:
            logger.error(f"Error loading marker data: {e}")
            raise
    
    def _build_token_index(self):
        """Erstellt einen Index aller einzigartigen Tokens."""
        all_tokens = set()
        
        # Sammle alle Tokens aus verschiedenen Marker-Kategorien
        for category, markers in self.marker_data.items():
            if isinstance(markers, dict):
                for marker_name, marker_info in markers.items():
                    if isinstance(marker_info, dict) and 'tokens' in marker_info:
                        all_tokens.update(marker_info['tokens'])
        
        # Erstelle Index
        self.token_index = {token: idx for idx, token in enumerate(sorted(all_tokens))}
        self.dimension_count = len(self.token_index)
    
    def vectorize_marker(self, marker_category: str, marker_name: str) -> Optional[CoSDVector]:
        """
        Konvertiert einen spezifischen Marker in einen CoSDVector.
        
        Args:
            marker_category: Kategorie des Markers (z.B. 'Architecture_Markers')
            marker_name: Name des Markers (z.B. 'root_signal')
            
        Returns:
            CoSDVector oder None wenn Marker nicht gefunden
        """
        if marker_category not in self.marker_data:
            return None
            
        markers = self.marker_data[marker_category]
        if marker_name not in markers:
            return None
        
        marker_info = markers[marker_name]
        
        # Erstelle Vektor mit Nullen
        vector = np.zeros(self.dimension_count)
        
        # Setze Werte basierend auf Tokens
        if 'tokens' in marker_info:
            weight = marker_info.get('weight', 1.0)
            for token in marker_info['tokens']:
                if token in self.token_index:
                    vector[self.token_index[token]] = weight
        
        return CoSDVector(
            dimensions=vector,
            timestamp=datetime.now(),
            marker_weights={marker_name: marker_info.get('weight', 1.0)},
            metadata={'category': marker_category, 'name': marker_name}
        )
    
    def vectorize_text_with_markers(
        self,
        text: str,
        active_markers: List[Tuple[str, str]]
    ) -> CoSDVector:
        """
        Erstellt einen Vektor basierend auf Text und aktiven Markern.
        
        Args:
            text: Zu analysierender Text
            active_markers: Liste von (category, marker_name) Tupeln
            
        Returns:
            Kombinierter CoSDVector
        """
        # Initialisiere Vektor
        combined_vector = np.zeros(self.dimension_count)
        marker_weights = {}
        
        # Kombiniere Marker-Vektoren
        for category, marker_name in active_markers:
            marker_vec = self.vectorize_marker(category, marker_name)
            if marker_vec:
                combined_vector += marker_vec.dimensions
                marker_weights[f"{category}.{marker_name}"] = marker_vec.marker_weights.get(marker_name, 1.0)
        
        # Normalisiere wenn nötig
        if np.any(combined_vector):
            combined_vector = combined_vector / np.linalg.norm(combined_vector)
        
        return CoSDVector(
            dimensions=combined_vector,
            timestamp=datetime.now(),
            marker_weights=marker_weights,
            metadata={'text_preview': text[:100], 'marker_count': len(active_markers)}
        )


class CoSDMarkerMatcher:
    """
    Erweitert den bestehenden MarkerMatcher um CoSD-Funktionalität.
    """
    
    def __init__(self, base_matcher=None, marker_data_path: Optional[str] = None):
        """
        Initialisiert den CoSD-erweiterten Matcher.
        
        Args:
            base_matcher: Instanz des Original-MarkerMatchers
            marker_data_path: Pfad zur erweiterten Marker-Datei
        """
        self.base_matcher = base_matcher
        self.vectorizer = MarkerVectorizer(marker_data_path)
        self.cluster_cache = {}
        
    def analyze_with_cosd(self, text: str) -> Dict[str, any]:
        """
        Analysiert Text mit Standard-Markern und CoSD-Erweiterungen.
        
        Args:
            text: Zu analysierender Text
            
        Returns:
            Dict mit Standard-Analyse und CoSD-Metriken
        """
        # Führe Standard-Analyse durch wenn base_matcher vorhanden
        base_result = None
        if self.base_matcher:
            base_result = self.base_matcher.analyze_text(text)
        
        # Extrahiere aktive Marker aus base_result
        active_markers = []
        if base_result:
            for match in base_result.gefundene_marker:
                # Versuche Kategorie aus Tags oder Metadata zu extrahieren
                category = self._infer_category(match)
                if category:
                    active_markers.append((category, match.marker_name))
        
        # Erstelle CoSD-Vektor
        text_vector = self.vectorizer.vectorize_text_with_markers(text, active_markers)
        
        # Identifiziere semantische Cluster
        clusters = self._identify_clusters(active_markers)
        
        # Berechne CoSD-spezifische Metriken
        cosd_metrics = {
            'vector_magnitude': float(text_vector.magnitude),
            'dominant_dimensions': self._get_dominant_dimensions(text_vector, n=5),
            'semantic_clusters': [
                {
                    'id': cluster.cluster_id,
                    'markers': list(cluster.marker_names),
                    'cohesion': cluster.cohesion_score
                }
                for cluster in clusters
            ],
            'marker_diversity': self._calculate_marker_diversity(active_markers)
        }
        
        # Kombiniere Ergebnisse
        result = {
            'base_analysis': base_result.to_dict() if base_result else None,
            'cosd_analysis': cosd_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def _infer_category(self, marker_match) -> Optional[str]:
        """
        Versucht die Kategorie eines Markers zu inferieren.
        
        Args:
            marker_match: MarkerMatch-Objekt
            
        Returns:
            Kategorie-String oder None
        """
        # Mapping von bekannten Tags zu Kategorien
        tag_to_category = {
            'manipulation': 'Architecture_Markers',
            'emotion': 'Emotion_Valence',
            'spiral': 'Spiral_Dynamics',
            'archetype': 'Archetypes',
            'meta': 'Meta_Semantik'
        }
        
        # Prüfe Tags
        for tag in marker_match.tags:
            tag_lower = tag.lower()
            for key, category in tag_to_category.items():
                if key in tag_lower:
                    return category
        
        # Fallback auf Marker-Namen-Analyse
        marker_lower = marker_match.marker_name.lower()
        if any(color in marker_lower for color in ['beige', 'purpur', 'rot', 'blau', 'orange', 'grün', 'gelb', 'türkis']):
            return 'Spiral_Dynamics'
        
        return 'Architecture_Markers'  # Default
    
    def _identify_clusters(self, active_markers: List[Tuple[str, str]]) -> List[SemanticCluster]:
        """
        Identifiziert semantische Cluster in den aktiven Markern.
        
        Args:
            active_markers: Liste von (category, marker_name) Tupeln
            
        Returns:
            Liste von SemanticCluster-Objekten
        """
        clusters = []
        
        # Gruppiere nach Kategorie als einfache Cluster-Strategie
        category_groups = {}
        for category, marker in active_markers:
            if category not in category_groups:
                category_groups[category] = set()
            category_groups[category].add(marker)
        
        # Erstelle Cluster für jede Kategorie mit mehr als einem Marker
        for category, markers in category_groups.items():
            if len(markers) > 1:
                cluster = SemanticCluster(
                    cluster_id=f"{category}_cluster_{datetime.now().timestamp()}",
                    marker_names=markers,
                    metadata={'category': category}
                )
                
                # Berechne Kohäsion wenn möglich
                marker_vectors = {}
                for marker in markers:
                    vec = self.vectorizer.vectorize_marker(category, marker)
                    if vec:
                        marker_vectors[marker] = vec
                
                if marker_vectors:
                    cluster.calculate_cohesion(marker_vectors)
                
                clusters.append(cluster)
        
        return clusters
    
    def _get_dominant_dimensions(self, vector: CoSDVector, n: int = 5) -> List[Dict[str, any]]:
        """
        Identifiziert die dominanten Dimensionen eines Vektors.
        
        Args:
            vector: CoSDVector
            n: Anzahl der Top-Dimensionen
            
        Returns:
            Liste mit Dimension-Info Dicts
        """
        # Finde Top-N Indizes
        top_indices = np.argsort(np.abs(vector.dimensions))[-n:][::-1]
        
        # Erstelle Reverse-Mapping von Index zu Token
        index_to_token = {idx: token for token, idx in self.vectorizer.token_index.items()}
        
        dominant = []
        for idx in top_indices:
            if idx in index_to_token and vector.dimensions[idx] != 0:
                dominant.append({
                    'token': index_to_token[idx],
                    'value': float(vector.dimensions[idx]),
                    'absolute_value': float(np.abs(vector.dimensions[idx]))
                })
        
        return dominant
    
    def _calculate_marker_diversity(self, active_markers: List[Tuple[str, str]]) -> float:
        """
        Berechnet die Diversität der aktiven Marker.
        
        Args:
            active_markers: Liste von (category, marker_name) Tupeln
            
        Returns:
            Diversitätswert zwischen 0 und 1
        """
        if not active_markers:
            return 0.0
        
        # Zähle einzigartige Kategorien
        categories = set(cat for cat, _ in active_markers)
        
        # Diversität basiert auf Anzahl verschiedener Kategorien
        max_categories = len(self.vectorizer.marker_data) if self.vectorizer.marker_data else 1
        diversity = len(categories) / max_categories
        
        return min(1.0, diversity) 