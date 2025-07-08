#!/usr/bin/env python3
"""
Einfacher Test für das CoSD-Modul
"""

import sys
import os
import pytest

# Füge das aktuelle Verzeichnis zum Python-Pfad hinzu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_cosd_import():
    """Testet den Import des CoSD-Moduls"""
    print("🔍 Teste CoSD-Import...")

    try:
        from cosd import CoSDAnalyzer  # noqa: F401
        print("✅ CoSD-Import erfolgreich!")
    except ImportError as e:
        pytest.fail(f"Import-Fehler: {e}")

def test_cosd_analyzer():
    """Testet den CoSD-Analyzer"""
    print("\n🧪 Teste CoSD-Analyzer...")
    
    try:
        from cosd import CoSDAnalyzer
        
        # Erstelle Analyzer
        analyzer = CoSDAnalyzer()
        print("✅ CoSD-Analyzer erstellt")
        
        # Teste mit einfachem Text
        test_texts = [
            "Hallo, wie geht es dir?",
            "Ey, Alter, das reicht jetzt. Ich bin hier raus.",
            "Ich bin hin- und hergerissen zwischen Bleiben und Gehen."
        ]
        
        print(f"📝 Analysiere {len(test_texts)} Test-Texte...")
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n   Text {i}: '{text[:50]}...'")
            
            try:
                result = analyzer.analyze_drift([text])
                print(f"   ✅ Analyse erfolgreich")
                
                # Verwende die richtigen Attribute
                if hasattr(result, 'drift_velocity') and result.drift_velocity:
                    avg_velocity = result.drift_velocity.get('average_velocity', 0)
                    print(f"      Drift-Velocity: {avg_velocity:.3f}")
                
                if hasattr(result, 'emergent_clusters'):
                    print(f"      Emergente Cluster: {len(result.emergent_clusters)}")
                
                if hasattr(result, 'risk_assessment'):
                    risk_level = result.risk_assessment.get('risk_level', 'unknown')
                    print(f"      Risk-Level: {risk_level}")
                
                if hasattr(result, 'resonance_patterns'):
                    print(f"      Resonanz-Muster: {len(result.resonance_patterns)}")
                    
            except Exception as e:
                pytest.fail(f"Analyse-Fehler: {e}")
        
    except Exception as e:
        pytest.fail(f"CoSD-Analyzer Fehler: {e}")

def test_chat_integration():
    """Testet die Chat-Integration"""
    print("\n💬 Teste Chat-Integration...")
    
    try:
        import requests
        
        # Teste ob Chat-Server läuft
        response = requests.get("http://localhost:5001/api/cosd/status", timeout=2)
        
        if response.status_code == 200:
            status = response.json()
            print("✅ Chat-Server läuft!")
            print(f"   CoSD verfügbar: {status.get('available', False)}")
        else:
            pytest.fail(f"Chat-Server Fehler: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        pytest.skip("Chat-Server nicht erreichbar")
    except Exception as e:
        pytest.fail(f"Chat-Integration Fehler: {e}")

if __name__ == "__main__":
    print("🚀 CoSD-Modul Test")
    print("=" * 50)
    
    # Teste Import
    import_ok = test_cosd_import()
    
    # Teste Analyzer
    analyzer_ok = False
    if import_ok:
        analyzer_ok = test_cosd_analyzer()
    
    # Teste Chat-Integration
    chat_ok = test_chat_integration()
    
    print("\n" + "=" * 50)
    print("📊 Test-Ergebnisse:")
    print(f"   Import: {'✅' if import_ok else '❌'}")
    print(f"   Analyzer: {'✅' if analyzer_ok else '❌'}")
    print(f"   Chat-Integration: {'✅' if chat_ok else '❌'}")
    
    if import_ok and analyzer_ok:
        print("\n🎉 CoSD-Modul funktioniert!")
        print("   Du kannst jetzt:")
        print("   1. Den Chat-Server starten: python3 chat_backend.py")
        print("   2. Im Browser öffnen: http://localhost:5001")
        print("   3. Chat-Nachrichten senden - CoSD läuft automatisch!")
    else:
        print("\n⚠️  Es gibt Probleme mit dem CoSD-Modul")
        print("   Prüfe die Fehlermeldungen oben") 