#!/usr/bin/env python3
"""
Einfacher Test für das MARSAP-System
"""

import requests
import json
from marker_matcher import MarkerMatcher

def test_direct_api():
    """Test der direkten Python-API"""
    print("🔍 Teste direkte Python-API...")
    
    # Matcher initialisieren
    matcher = MarkerMatcher()
    
    # Test-Texte - diese kommen tatsächlich in den Markern vor
    test_texts = [
        "Ey, Alter, das reicht jetzt. Ich bin hier raus.",  # ABBRUCHMARKER
        "Ich bin hin- und hergerissen zwischen Bleiben und Gehen – ich weiß nicht mehr weiter.",  # AMBIVALENZMARKER
        "Einerseits will ich dich unterstützen, aber andererseits bin ich mir nicht sicher, ob ich es schaffe.",  # AMBIVALENZMARKER
        "Ich hab einerseits Lust auf das Abenteuer, andererseits macht mir der Gedanke auch Angst.",  # AMBIVALENZMARKER
        "Ey, was soll das? Ich hab keinen Bock mehr, gute Nacht.",  # ABBRUCHMARKER
        "Ich liebe den Job, aber gleichzeitig zweifle ich jeden Tag daran.",  # AMBIVALENZMARKER
        "Alter, hör auf damit, ich will nicht weiter drüber reden.",  # ABBRUCHMARKER
        "Ich schwanke zwischen Enthusiasmus und Zweifel, ich kann mich nicht entscheiden."  # AMBIVALENZMARKER
    ]
    
    for text in test_texts:
        print(f"\n📝 Analysiere: '{text}'")
        result = matcher.analyze_text(text)
        
        print(f"   Risk-Level: {result.risk_level}")
        print(f"   Gefundene Marker: {len(result.gefundene_marker)}")
        
        for match in result.gefundene_marker:
            print(f"   - {match.marker_name}: '{match.matched_text}'")

def test_rest_api():
    """Test der REST-API"""
    print("\n🌐 Teste REST-API...")
    
    base_url = "http://localhost:5001"
    
    # Test einzelner Text
    test_data = {"text": "Ey, Alter, das reicht jetzt. Ich bin hier raus."}
    
    try:
        response = requests.post(f"{base_url}/analyze", 
                               json=test_data, 
                               headers={"Content-Type": "application/json"})
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API funktioniert!")
            print(f"   Risk-Level: {result['risk_level']}")
            print(f"   Marker gefunden: {len(result['gefundene_marker'])}")
            
            for marker in result['gefundene_marker']:
                print(f"   - {marker['marker_name']}: '{marker['matched_text']}'")
        else:
            print(f"❌ API-Fehler: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ API-Server nicht erreichbar. Starte ihn mit: python3 marker_api.py --port 5001")

def test_cli():
    """Test der Kommandozeile"""
    print("\n💻 Teste CLI...")
    print("Verwende: python3 marker_cli.py --text 'Ey, Alter, das reicht jetzt. Ich bin hier raus.'")

if __name__ == "__main__":
    print("🚀 MARSAP-System Test")
    print("=" * 50)
    
    # Teste direkte API
    test_direct_api()
    
    # Teste REST-API
    test_rest_api()
    
    # CLI-Hinweis
    test_cli()
    
    print("\n" + "=" * 50)
    print("✅ Test abgeschlossen!")
    print("\n📚 Nächste Schritte:")
    print("1. Analysiere eigene Texte mit der Python-API")
    print("2. Nutze die REST-API für Web-Integration")
    print("3. Verwende die CLI für Batch-Verarbeitung")
    print("4. Schau dir die Dokumentation in README.md an") 