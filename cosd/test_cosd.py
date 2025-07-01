#!/usr/bin/env python3
"""
Test Suite für CoSD Module - Minimale Unit-Tests ohne Demo-Daten
"""

import unittest
import numpy as np
from datetime import datetime, timedelta

from cosd.cost_vector_math import (
    CoSDVector, VectorOperations, calculate_drift_velocity,
    calculate_resonance_coupling, cluster_vectors
)
from cosd.drift_analyzer import CoSDAnalyzer, DriftAnalysisResult
from cosd.semantic_marker_interface import SemanticCluster, MarkerVectorizer


class TestCoSDVectorMath(unittest.TestCase):
    """Tests für mathematische Vektor-Operationen"""
    
    def test_vector_creation(self):
        """Test: Vektor-Erstellung und Eigenschaften"""
        dims = np.array([1.0, 2.0, 3.0])
        vec = CoSDVector(
            dimensions=dims,
            timestamp=datetime.now(),
            marker_weights={'test': 1.0}
        )
        
        self.assertEqual(vec.dimensions.shape, (3,))
        self.assertAlmostEqual(vec.magnitude, np.sqrt(14), places=5)
    
    def test_vector_normalization(self):
        """Test: Vektor-Normalisierung"""
        vec = CoSDVector(
            dimensions=np.array([3.0, 4.0]),
            timestamp=datetime.now()
        )
        
        normalized = vec.normalize()
        self.assertAlmostEqual(normalized.magnitude, 1.0, places=5)
    
    def test_cosine_similarity(self):
        """Test: Kosinus-Ähnlichkeit"""
        vec1 = CoSDVector(dimensions=np.array([1.0, 0.0]), timestamp=datetime.now())
        vec2 = CoSDVector(dimensions=np.array([0.0, 1.0]), timestamp=datetime.now())
        vec3 = CoSDVector(dimensions=np.array([1.0, 0.0]), timestamp=datetime.now())
        
        # Orthogonale Vektoren
        sim_ortho = VectorOperations.cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(sim_ortho, 0.0, places=5)
        
        # Identische Vektoren
        sim_same = VectorOperations.cosine_similarity(vec1, vec3)
        self.assertAlmostEqual(sim_same, 1.0, places=5)
    
    def test_drift_velocity(self):
        """Test: Drift-Geschwindigkeitsberechnung"""
        base_time = datetime.now()
        vectors = [
            CoSDVector(
                dimensions=np.array([i, 0, 0]),
                timestamp=base_time + timedelta(seconds=i)
            )
            for i in range(3)
        ]
        
        velocity = calculate_drift_velocity(vectors)
        
        self.assertIn('average_velocity', velocity)
        self.assertIn('total_distance', velocity)
        self.assertEqual(len(velocity['instantaneous_velocities']), 2)
    
    def test_resonance_coupling(self):
        """Test: Resonanz-Kopplungs-Berechnung"""
        vec1 = CoSDVector(dimensions=np.array([1.0, 0.0]), timestamp=datetime.now())
        vec2 = CoSDVector(dimensions=np.array([0.9, 0.1]), timestamp=datetime.now())
        
        coupling = calculate_resonance_coupling(vec1, vec2, 0.8)
        
        self.assertIn('coupling_strength', coupling)
        self.assertIn('is_strongly_coupled', coupling)
        self.assertTrue(coupling['is_strongly_coupled'])


class TestSemanticInterface(unittest.TestCase):
    """Tests für Semantic Marker Interface"""
    
    def test_semantic_cluster(self):
        """Test: Semantischer Cluster"""
        cluster = SemanticCluster(
            cluster_id="test_cluster",
            marker_names={'marker1', 'marker2'}
        )
        
        cluster.add_marker('marker3')
        self.assertEqual(len(cluster.marker_names), 3)
        self.assertIn('marker3', cluster.marker_names)
    
    def test_cluster_cohesion(self):
        """Test: Cluster-Kohäsionsberechnung"""
        cluster = SemanticCluster(
            cluster_id="test",
            marker_names={'m1', 'm2'}
        )
        
        # Mock-Vektoren
        vectors = {
            'm1': CoSDVector(dimensions=np.array([1.0, 0.0]), timestamp=datetime.now()),
            'm2': CoSDVector(dimensions=np.array([0.9, 0.1]), timestamp=datetime.now())
        }
        
        cohesion = cluster.calculate_cohesion(vectors)
        self.assertGreater(cohesion, 0.8)


class TestDriftAnalyzer(unittest.TestCase):
    """Tests für Drift Analyzer"""
    
    def test_analyzer_initialization(self):
        """Test: Analyzer-Initialisierung"""
        analyzer = CoSDAnalyzer()
        
        self.assertIsNotNone(analyzer.config)
        self.assertIn('resonance_threshold', analyzer.config)
    
    def test_minimal_drift_analysis(self):
        """Test: Minimale Drift-Analyse"""
        analyzer = CoSDAnalyzer()
        
        # Minimale Testsequenz
        texts = ["test", "analysis"]
        
        result = analyzer.analyze_drift(texts)
        
        self.assertIsInstance(result, DriftAnalysisResult)
        self.assertEqual(len(result.text_sequence), 2)
        self.assertIn('risk_level', result.risk_assessment)
    
    def test_risk_assessment(self):
        """Test: Risikobewertung"""
        analyzer = CoSDAnalyzer()
        
        # Test mit verschiedenen Metriken
        risk = analyzer._assess_drift_risk(
            drift_velocity={'average_velocity': 0.1},
            drift_path={'curvature': 0.2, 'stability_zones': []},
            resonance_patterns=[],
            emergent_clusters=[]
        )
        
        self.assertIn('risk_level', risk)
        self.assertIn('risk_factors', risk)
        self.assertIn('recommendations', risk)
        self.assertEqual(risk['risk_level'], 'low')


class TestIntegration(unittest.TestCase):
    """Integrationstests für Gesamtsystem"""
    
    def test_end_to_end_analysis(self):
        """Test: End-to-End Analyse-Pipeline"""
        analyzer = CoSDAnalyzer()
        
        # Minimal-Sequenz
        texts = [
            "alpha beta",
            "beta gamma",
            "gamma delta"
        ]
        
        result = analyzer.analyze_drift(texts)
        
        # Validiere Struktur
        self.assertEqual(len(result.drift_vectors), 3)
        self.assertIsNotNone(result.drift_velocity)
        self.assertIsNotNone(result.drift_path)
        self.assertIsInstance(result.resonance_patterns, list)
        self.assertIsInstance(result.emergent_clusters, list)
        
        # Validiere Serialisierung
        result_dict = result.to_dict()
        self.assertIn('timestamp', result_dict)
        self.assertIn('drift_velocity', result_dict)
        self.assertIn('risk_assessment', result_dict)


def run_tests():
    """Führt alle Tests aus"""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    run_tests() 