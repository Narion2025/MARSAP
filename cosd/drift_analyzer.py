#!/usr/bin/env python3
"""
Drift Analyzer - Kernmodul für Co-emergente Semantic Drift (CoSD) Analyse

Dieses Modul implementiert die Hauptlogik für die Analyse semantischer Drift
in Textsequenzen und die Erkennung von emergenten Bedeutungsmustern.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import logging

from .cost_vector_math import (
    CoSDVector, VectorOperations, calculate_drift_velocity,
    calculate_resonance_coupling, cluster_vectors, calculate_semantic_drift_path
)
from .semantic_marker_interface import (
    MarkerVectorizer, SemanticCluster, CoSDMarkerMatcher
)

logger = logging.getLogger(__name__)


@dataclass
class DriftAnalysisResult:
    """
    Ergebnis einer CoSD-Analyse.
    
    Attributes:
        text_sequence: Analysierte Textsequenz
        drift_vectors: Liste der berechneten Drift-Vektoren
        drift_velocity: Drift-Geschwindigkeitsmetriken
        drift_path: Pfad-Metriken der semantischen Bewegung
        resonance_patterns: Erkannte Resonanzmuster
        emergent_clusters: Neu entstehende semantische Cluster
        risk_assessment: Risikobewertung basierend auf Drift-Mustern
        timestamp: Zeitstempel der Analyse
    """
    text_sequence: List[str]
    drift_vectors: List[CoSDVector]
    drift_velocity: Dict[str, Union[float, List[float]]]
    drift_path: Dict[str, any]
    resonance_patterns: List[Dict[str, any]]
    emergent_clusters: List[SemanticCluster]
    risk_assessment: Dict[str, any]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, any]:
        """Konvertiert das Ergebnis in ein serialisierbares Dict."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'text_count': len(self.text_sequence),
            'drift_velocity': self.drift_velocity,
            'drift_path': self.drift_path,
            'resonance_patterns': self.resonance_patterns,
            'emergent_clusters': [
                {
                    'id': cluster.cluster_id,
                    'markers': list(cluster.marker_names),
                    'cohesion': cluster.cohesion_score,
                    'emergence_time': cluster.emergence_timestamp.isoformat()
                }
                for cluster in self.emergent_clusters
            ],
            'risk_assessment': self.risk_assessment,
            'metadata': self.metadata
        } 

class CoSDAnalyzer:
    """
    Hauptklasse für Co-emergente Semantic Drift Analyse.
    
    Diese Klasse koordiniert die verschiedenen Analyse-Komponenten
    und bietet eine einheitliche Schnittstelle für CoSD-Analysen.
    """
    
    def __init__(
        self,
        marker_data_path: Optional[str] = None,
        base_matcher=None,
        config: Optional[Dict[str, any]] = None
    ):
        """
        Initialisiert den CoSD-Analyzer.
        
        Args:
            marker_data_path: Pfad zur erweiterten Marker-Datei
            base_matcher: Optionale Instanz des Original-MarkerMatchers
            config: Optionale Konfigurationsparameter
        """
        self.config = config or self._default_config()
        self.cosd_matcher = CoSDMarkerMatcher(base_matcher, marker_data_path)
        self.vectorizer = self.cosd_matcher.vectorizer
        
        # Cache für Performance
        self.vector_cache = {}
        self.cluster_history = []
        
    def _default_config(self) -> Dict[str, any]:
        """Gibt die Standard-Konfiguration zurück."""
        return {
            'min_text_length': 10,
            'resonance_threshold': 0.7,
            'cluster_distance_threshold': 0.5,
            'drift_velocity_window': 3,
            'emergence_detection_sensitivity': 0.8,
            'risk_thresholds': {
                'low': 0.3,
                'medium': 0.6,
                'high': 0.85
            }
        }
    
    def analyze_drift(
        self,
        text_sequence: List[str],
        timestamps: Optional[List[datetime]] = None
    ) -> DriftAnalysisResult:
        """
        Analysiert semantische Drift in einer Textsequenz.
        
        Args:
            text_sequence: Liste von Texten in chronologischer Reihenfolge
            timestamps: Optionale Liste von Zeitstempeln für jeden Text
            
        Returns:
            DriftAnalysisResult mit allen Analyse-Metriken
        """
        if not text_sequence:
            raise ValueError("Text sequence cannot be empty")
        
        # Generiere Zeitstempel falls nicht vorhanden
        if not timestamps:
            base_time = datetime.now()
            timestamps = [
                base_time + timedelta(minutes=i * 5)
                for i in range(len(text_sequence))
            ]
        
        # Phase 1: Vektorisierung
        logger.info(f"Starting CoSD analysis for {len(text_sequence)} texts")
        drift_vectors = self._create_drift_vectors(text_sequence, timestamps)
        
        # Phase 2: Drift-Berechnung
        drift_velocity = calculate_drift_velocity(
            drift_vectors,
            self.config['drift_velocity_window']
        )
        drift_path = calculate_semantic_drift_path(drift_vectors)
        
        # Phase 3: Resonanz-Analyse
        resonance_patterns = self._analyze_resonance_patterns(drift_vectors)
        
        # Phase 4: Emergenz-Detektion
        emergent_clusters = self._detect_emergent_clusters(
            drift_vectors,
            text_sequence
        )
        
        # Phase 5: Risikobewertung
        risk_assessment = self._assess_drift_risk(
            drift_velocity,
            drift_path,
            resonance_patterns,
            emergent_clusters
        )
        
        return DriftAnalysisResult(
            text_sequence=text_sequence,
            drift_vectors=drift_vectors,
            drift_velocity=drift_velocity,
            drift_path=drift_path,
            resonance_patterns=resonance_patterns,
            emergent_clusters=emergent_clusters,
            risk_assessment=risk_assessment,
            metadata={
                'analyzer_config': self.config,
                'total_drift_distance': drift_path['path_length']
            }
        )
    
    def _create_drift_vectors(
        self,
        text_sequence: List[str],
        timestamps: List[datetime]
    ) -> List[CoSDVector]:
        """
        Erstellt Drift-Vektoren für eine Textsequenz.
        
        Args:
            text_sequence: Liste von Texten
            timestamps: Liste von Zeitstempeln
            
        Returns:
            Liste von CoSDVector-Objekten
        """
        vectors = []
        
        for i, (text, timestamp) in enumerate(zip(text_sequence, timestamps)):
            # Prüfe Cache
            cache_key = hash(text)
            if cache_key in self.vector_cache:
                vector = self.vector_cache[cache_key]
                # Aktualisiere Timestamp
                vector.timestamp = timestamp
            else:
                # Analysiere Text mit CoSD-Matcher
                analysis = self.cosd_matcher.analyze_with_cosd(text)
                
                # Extrahiere aktive Marker
                active_markers = []
                if analysis['base_analysis']:
                    for match in analysis['base_analysis']['gefundene_marker']:
                        # Inferiere Kategorie (vereinfacht)
                        category = self._infer_marker_category(match)
                        marker_name = match['marker']
                        active_markers.append((category, marker_name))
                
                # Erstelle Vektor
                vector = self.vectorizer.vectorize_text_with_markers(
                    text,
                    active_markers
                )
                vector.timestamp = timestamp
                
                # Cache für Performance
                self.vector_cache[cache_key] = vector
            
            vectors.append(vector)
        
        return vectors
    
    def _analyze_resonance_patterns(
        self,
        vectors: List[CoSDVector]
    ) -> List[Dict[str, any]]:
        """
        Analysiert Resonanzmuster zwischen Vektoren.
        
        Args:
            vectors: Liste von CoSDVector-Objekten
            
        Returns:
            Liste von Resonanzmustern
        """
        patterns = []
        threshold = self.config['resonance_threshold']
        
        # Analysiere Paare mit zeitlichem Abstand
        for i in range(len(vectors)):
            for j in range(i + 1, min(i + 5, len(vectors))):  # Max 5 Schritte voraus
                coupling = calculate_resonance_coupling(
                    vectors[i],
                    vectors[j],
                    threshold
                )
                
                if coupling['is_strongly_coupled']:
                    pattern = {
                        'type': 'strong_resonance',
                        'indices': [i, j],
                        'time_delta': (vectors[j].timestamp - vectors[i].timestamp).total_seconds(),
                        'coupling_strength': coupling['coupling_strength'],
                        'phase_alignment': coupling['phase_alignment'],
                        'resonance_factor': coupling['resonance_factor']
                    }
                    patterns.append(pattern)
        
        # Identifiziere Resonanz-Ketten
        chains = self._identify_resonance_chains(patterns, len(vectors))
        patterns.extend(chains)
        
        return patterns
    
    def _identify_resonance_chains(
        self,
        patterns: List[Dict[str, any]],
        vector_count: int
    ) -> List[Dict[str, any]]:
        """
        Identifiziert Ketten von Resonanzmustern.
        
        Args:
            patterns: Liste von einzelnen Resonanzmustern
            vector_count: Anzahl der Vektoren
            
        Returns:
            Liste von Resonanzketten
        """
        chains = []
        
        # Erstelle Adjazenzliste
        connections = defaultdict(set)
        for pattern in patterns:
            i, j = pattern['indices']
            connections[i].add(j)
            connections[j].add(i)
        
        # Finde zusammenhängende Komponenten
        visited = set()
        for start in range(vector_count):
            if start in visited or start not in connections:
                continue
            
            # BFS für zusammenhängende Komponente
            chain = set()
            queue = [start]
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                
                visited.add(current)
                chain.add(current)
                
                for neighbor in connections[current]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            
            if len(chain) >= 3:  # Mindestens 3 verbundene Punkte
                chains.append({
                    'type': 'resonance_chain',
                    'indices': sorted(list(chain)),
                    'chain_length': len(chain),
                    'density': len([p for p in patterns if all(idx in chain for idx in p['indices'])]) / (len(chain) * (len(chain) - 1) / 2)
                })
        
        return chains
    
    def _detect_emergent_clusters(
        self,
        vectors: List[CoSDVector],
        text_sequence: List[str]
    ) -> List[SemanticCluster]:
        """
        Detektiert emergente semantische Cluster.
        
        Args:
            vectors: Liste von CoSDVector-Objekten
            text_sequence: Originale Textsequenz
            
        Returns:
            Liste von emergenten SemanticCluster-Objekten
        """
        emergent_clusters = []
        
        # Cluster Vektoren
        cluster_indices = cluster_vectors(
            vectors,
            self.config['cluster_distance_threshold'],
            method='cosine'
        )
        
        # Analysiere jeden Cluster auf Emergenz
        for cluster_idx_list in cluster_indices:
            if len(cluster_idx_list) < 2:
                continue
            
            # Extrahiere Marker für diesen Cluster
            cluster_markers = set()
            for idx in cluster_idx_list:
                vector = vectors[idx]
                cluster_markers.update(vector.marker_weights.keys())
            
            # Prüfe auf Emergenz (neue Kombinationen)
            is_emergent = self._check_emergence(cluster_markers, cluster_idx_list)
            
            if is_emergent:
                cluster = SemanticCluster(
                    cluster_id=f"emergent_{datetime.now().timestamp()}",
                    marker_names=cluster_markers,
                    emergence_timestamp=vectors[min(cluster_idx_list)].timestamp,
                    metadata={
                        'indices': cluster_idx_list,
                        'emergence_score': self._calculate_emergence_score(
                            cluster_markers,
                            cluster_idx_list,
                            vectors
                        )
                    }
                )
                
                # Berechne Zentroid
                cluster_vecs = [vectors[i] for i in cluster_idx_list]
                centroid_dims = np.mean([v.dimensions for v in cluster_vecs], axis=0)
                cluster.centroid = CoSDVector(
                    dimensions=centroid_dims,
                    timestamp=datetime.now()
                )
                
                emergent_clusters.append(cluster)
        
        # Aktualisiere Cluster-Historie
        self.cluster_history.extend(emergent_clusters)
        
        return emergent_clusters
    
    def _check_emergence(
        self,
        marker_set: set,
        indices: List[int]
    ) -> bool:
        """
        Prüft ob eine Marker-Kombination emergent ist.
        
        Args:
            marker_set: Set von Marker-Namen
            indices: Indizes der beteiligten Vektoren
            
        Returns:
            True wenn emergent, sonst False
        """
        # Ein Cluster ist emergent wenn:
        # 1. Er eine neue Kombination von Markern darstellt
        # 2. Die Kombination über Zeit stabil ist
        
        # Prüfe gegen Historie
        for hist_cluster in self.cluster_history:
            if marker_set == hist_cluster.marker_names:
                return False  # Bereits bekannt
        
        # Prüfe Stabilität (mindestens 2 aufeinanderfolgende Vorkommen)
        consecutive = 0
        for i in range(len(indices) - 1):
            if indices[i+1] - indices[i] <= 2:  # Max 2 Schritte Abstand
                consecutive += 1
        
        return consecutive >= 1 and len(marker_set) >= 2
    
    def _calculate_emergence_score(
        self,
        marker_set: set,
        indices: List[int],
        vectors: List[CoSDVector]
    ) -> float:
        """
        Berechnet einen Emergenz-Score für einen Cluster.
        
        Args:
            marker_set: Set von Marker-Namen
            indices: Indizes der beteiligten Vektoren
            vectors: Alle Vektoren
            
        Returns:
            Emergenz-Score zwischen 0 und 1
        """
        # Faktoren für Emergenz-Score:
        # 1. Neuartigkeit der Kombination
        # 2. Kohäsion des Clusters
        # 3. Zeitliche Persistenz
        
        # Neuartigkeit (vereinfacht: je mehr Marker, desto neuartiger)
        novelty = min(1.0, len(marker_set) / 5.0)
        
        # Kohäsion
        cluster_vecs = [vectors[i] for i in indices]
        similarities = []
        for i in range(len(cluster_vecs)):
            for j in range(i+1, len(cluster_vecs)):
                sim = VectorOperations.cosine_similarity(cluster_vecs[i], cluster_vecs[j])
                similarities.append(sim)
        cohesion = np.mean(similarities) if similarities else 0.0
        
        # Persistenz
        time_span = (vectors[max(indices)].timestamp - vectors[min(indices)].timestamp).total_seconds()
        persistence = min(1.0, time_span / (3600 * 24))  # Normalisiert auf 24h
        
        # Kombinierter Score
        emergence_score = (novelty * 0.3 + cohesion * 0.4 + persistence * 0.3)
        
        return float(emergence_score)
    
    def _assess_drift_risk(
        self,
        drift_velocity: Dict[str, Union[float, List[float]]],
        drift_path: Dict[str, any],
        resonance_patterns: List[Dict[str, any]],
        emergent_clusters: List[SemanticCluster]
    ) -> Dict[str, any]:
        """
        Bewertet das Risiko basierend auf Drift-Mustern.
        
        Args:
            drift_velocity: Drift-Geschwindigkeitsmetriken
            drift_path: Pfad-Metriken
            resonance_patterns: Erkannte Resonanzmuster
            emergent_clusters: Emergente Cluster
            
        Returns:
            Dict mit Risikobewertung
        """
        # Risikofaktoren
        risk_factors = {}
        
        # 1. Drift-Geschwindigkeit
        avg_velocity = drift_velocity['average_velocity']
        velocity_risk = min(1.0, avg_velocity / 0.5)  # Normalisiert
        risk_factors['velocity'] = velocity_risk
        
        # 2. Pfad-Krümmung (hohe Krümmung = instabil)
        curvature = drift_path['curvature']
        curvature_risk = min(1.0, curvature)
        risk_factors['curvature'] = curvature_risk
        
        # 3. Resonanz-Dichte
        resonance_chains = [p for p in resonance_patterns if p['type'] == 'resonance_chain']
        if resonance_chains:
            max_chain_length = max(p['chain_length'] for p in resonance_chains)
            resonance_risk = min(1.0, max_chain_length / 10.0)
        else:
            resonance_risk = 0.0
        risk_factors['resonance'] = resonance_risk
        
        # 4. Emergenz-Rate
        emergence_rate = len(emergent_clusters) / max(1, len(drift_path.get('stability_zones', [])) + 1)
        emergence_risk = min(1.0, emergence_rate)
        risk_factors['emergence'] = emergence_risk
        
        # Gesamtrisiko (gewichteter Durchschnitt)
        weights = {
            'velocity': 0.3,
            'curvature': 0.2,
            'resonance': 0.3,
            'emergence': 0.2
        }
        
        total_risk = sum(
            risk_factors[factor] * weights[factor]
            for factor in weights
        )
        
        # Risiko-Level bestimmen
        thresholds = self.config['risk_thresholds']
        if total_risk < thresholds['low']:
            risk_level = 'low'
            risk_color = 'green'
        elif total_risk < thresholds['medium']:
            risk_level = 'medium'
            risk_color = 'yellow'
        elif total_risk < thresholds['high']:
            risk_level = 'high'
            risk_color = 'orange'
        else:
            risk_level = 'critical'
            risk_color = 'red'
        
        return {
            'risk_level': risk_level,
            'risk_color': risk_color,
            'total_risk_score': float(total_risk),
            'risk_factors': risk_factors,
            'recommendations': self._generate_recommendations(risk_level, risk_factors)
        }
    
    def _generate_recommendations(
        self,
        risk_level: str,
        risk_factors: Dict[str, float]
    ) -> List[str]:
        """
        Generiert Empfehlungen basierend auf Risikobewertung.
        
        Args:
            risk_level: Gesamt-Risikolevel
            risk_factors: Einzelne Risikofaktoren
            
        Returns:
            Liste von Empfehlungen
        """
        recommendations = []
        
        if risk_level in ['high', 'critical']:
            recommendations.append("Erhöhte Wachsamkeit empfohlen - signifikante semantische Drift erkannt")
        
        if risk_factors['velocity'] > 0.7:
            recommendations.append("Schnelle semantische Verschiebung - prüfen Sie auf manipulative Muster")
        
        if risk_factors['curvature'] > 0.7:
            recommendations.append("Instabile Kommunikationsmuster - achten Sie auf plötzliche Themenwechsel")
        
        if risk_factors['resonance'] > 0.7:
            recommendations.append("Starke Resonanzmuster erkannt - mögliche emotionale Verstärkung")
        
        if risk_factors['emergence'] > 0.7:
            recommendations.append("Neue Bedeutungsmuster entstehen - beobachten Sie Veränderungen im Diskurs")
        
        if not recommendations:
            recommendations.append("Kommunikation erscheint stabil - normale Beobachtung ausreichend")
        
        return recommendations
    
    def _infer_marker_category(self, marker_match: Dict[str, any]) -> str:
        """
        Inferiert die Kategorie eines Markers aus den Match-Daten.
        
        Args:
            marker_match: Dict mit Marker-Match-Informationen
            
        Returns:
            Kategorie-String
        """
        # Vereinfachte Kategorie-Inferenz
        marker_name = marker_match.get('marker', '').lower()
        
        # Prüfe bekannte Muster
        if any(emotion in marker_name for emotion in ['angst', 'freude', 'trauer', 'wut']):
            return 'Emotion_Valence'
        elif any(color in marker_name for color in ['beige', 'purpur', 'rot', 'blau', 'orange', 'grün', 'gelb', 'türkis']):
            return 'Spiral_Dynamics'
        elif any(arch in marker_name for arch in ['pippi', 'tyler', 'picard', 'clarissa', 'hermes', 'athena']):
            return 'Archetypes'
        elif any(meta in marker_name for meta in ['emergenz', 'resonanz', 'frequenz', 'spiegel']):
            return 'Meta_Semantik'
        else:
            return 'Architecture_Markers' 