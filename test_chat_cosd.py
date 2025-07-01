#!/usr/bin/env python3
"""
Test für CoSD-Chat-Integration
Zeigt, wie das CoSD-Tool automatisch bei Chat-Nachrichten läuft
"""

import requests
import json
import time
from datetime import datetime

def test_chat_with_cosd():
    """Testet den Chat mit automatischer CoSD-Analyse"""
    
    base_url = "http://localhost:5001"
    
    print("🚀 Teste Chat mit CoSD-Integration")
    print("=" * 50)
    
    # 1. Neue Chat-Session erstellen
    print("📝 Erstelle neue Chat-Session...")
    session_data = {
        "session_name": "CoSD Test Session",
        "config": {
            "auto_analyze": True,
            "analysis_window": 5
        }
    }
    
    try:
        response = requests.post(f"{base_url}/api/chat/sessions", json=session_data)
        if response.status_code == 200:
            session = response.json()
            session_id = session['session_id']
            print(f"✅ Session erstellt: {session_id}")
        else:
            print(f"❌ Fehler beim Erstellen der Session: {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Chat-Server nicht erreichbar. Starte ihn mit: python3 chat_backend.py")
        return
    
    # 2. Nachrichten senden mit CoSD-Analyse
    test_messages = [
        "Hallo, wie geht es dir?",
        "Ey, Alter, das reicht jetzt. Ich bin hier raus.",
        "Ich bin hin- und hergerissen zwischen Bleiben und Gehen.",
        "Du bist zu empfindlich. Ich hab keinen Bock mehr.",
        "Einerseits will ich dich unterstützen, aber andererseits bin ich mir nicht sicher."
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n💬 Sende Nachricht {i}: '{message}'")
        
        message_data = {
            "content": message,
            "sender": "user"
        }
        
        response = requests.post(f"{base_url}/api/chat/sessions/{session_id}/messages", json=message_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Nachricht gesendet")
            
            # Zeige CoSD-Analyse falls vorhanden
            if 'cosd_analysis' in result and result['cosd_analysis']:
                cosd = result['cosd_analysis']
                print(f"   🔍 CoSD-Analyse:")
                print(f"      - Status: {cosd.get('status', 'N/A')}")
                print(f"      - Drift-Score: {cosd.get('drift_score', 'N/A')}")
                print(f"      - Marker gefunden: {len(cosd.get('markers', []))}")
                
                for marker in cosd.get('markers', [])[:3]:  # Zeige nur erste 3
                    print(f"        • {marker.get('name', 'N/A')}")
        else:
            print(f"❌ Fehler beim Senden: {response.status_code}")
        
        time.sleep(1)  # Kurze Pause zwischen Nachrichten
    
    # 3. CoSD-Historie abrufen
    print(f"\n📊 Hole CoSD-Historie...")
    response = requests.get(f"{base_url}/api/chat/sessions/{session_id}/cosd/history")
    
    if response.status_code == 200:
        history = response.json()
        print(f"✅ CoSD-Historie abgerufen: {len(history)} Analysen")
        
        for i, analysis in enumerate(history[-3:], 1):  # Zeige nur letzte 3
            print(f"   Analyse {i}: {analysis.get('timestamp', 'N/A')}")
            print(f"      Drift-Score: {analysis.get('analysis', {}).get('drift_score', 'N/A')}")
    else:
        print(f"❌ Fehler beim Abrufen der Historie: {response.status_code}")
    
    # 4. Session-Export
    print(f"\n💾 Exportiere Session...")
    response = requests.get(f"{base_url}/api/chat/sessions/{session_id}/export")
    
    if response.status_code == 200:
        export = response.json()
        print(f"✅ Session exportiert")
        print(f"   Nachrichten: {len(export.get('messages', []))}")
        print(f"   CoSD-Analysen: {len(export.get('cosd_history', []))}")
    else:
        print(f"❌ Fehler beim Export: {response.status_code}")
    
    print("\n" + "=" * 50)
    print("✅ Chat mit CoSD-Integration getestet!")
    print(f"🌐 Chat-Interface: http://localhost:5001")
    print(f"📊 Session-ID: {session_id}")

def test_cosd_status():
    """Testet den CoSD-Status"""
    print("\n🔍 Teste CoSD-Status...")
    
    try:
        response = requests.get("http://localhost:5001/api/cosd/status")
        if response.status_code == 200:
            status = response.json()
            print(f"✅ CoSD Status: {status.get('status', 'N/A')}")
            print(f"   Verfügbar: {status.get('available', False)}")
            print(f"   Konfiguration: {status.get('config', {})}")
        else:
            print(f"❌ Status-Fehler: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Server nicht erreichbar")

if __name__ == "__main__":
    print("🚀 CoSD-Chat-Integration Test")
    print("=" * 50)
    
    # Teste CoSD-Status
    test_cosd_status()
    
    # Teste Chat mit CoSD
    test_chat_with_cosd()
    
    print("\n📚 Nächste Schritte:")
    print("1. Öffne http://localhost:5001 im Browser")
    print("2. Erstelle eine neue Chat-Session")
    print("3. Sende Nachrichten und schaue dir die CoSD-Analyse an")
    print("4. Die Analyse läuft automatisch bei jeder Nachricht!") 