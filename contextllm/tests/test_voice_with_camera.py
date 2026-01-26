#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV-based voice analysis + camera monitoring
Display voice analysis results on camera window
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json
import threading
import time

# Add src folder to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / 'config' / '.env'
load_dotenv(env_path)

import cv2
import numpy as np
import speech_recognition as sr
from core.voice_analyzer import VoiceAnalyzer
from core.alert_manager import get_alert_manager

# Initialize
alert_manager = get_alert_manager()
voice_analyzer = VoiceAnalyzer()
recognizer = sr.Recognizer()

# Sound alert toggle for monitor (env: MONITOR_SOUND_ALERT=1 to enable)
SOUND_ALERT_ENABLED = os.getenv('MONITOR_SOUND_ALERT', '1')

# Logging setup
LOG_DIR = Path(__file__).parent.parent / 'data' / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
SESSION_START = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f'conversation_history_{SESSION_START}.json'

# Situation guide
SITUATION_GUIDE = {
    'CRITICAL': {
        'description': '[CRITICAL] Immediate action required',
        'examples': ['Server down', 'Security breach', 'System error'],
    },
    'HIGH': {
        'description': '[HIGH] Quick response needed',
        'examples': ['Performance degradation', 'User complaint', 'Error'],
    },
    'MEDIUM': {
        'description': '[MEDIUM] Normal response',
        'examples': ['General query', 'Information', 'Other'],
    },
    'LOW': {
        'description': '[LOW] Background information',
        'examples': ['Greeting', 'General conversation', 'Note'],
    }
}


class VoiceMonitorWindow:
    """Voice monitoring + OpenCV display"""
    
    def __init__(self):
        self.width = 800
        self.height = 600
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.turn = 1
        self.listening = False
        self.last_analysis = None
        # Enable or disable system sound alert on emergency
        self.sound_alert_enabled = (SOUND_ALERT_ENABLED == '1')
        
    def create_blank_frame(self):
        """Create blank frame with dark background"""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = (40, 40, 40)
        return frame
    
    def draw_title(self, frame):
        """Draw title"""
        text = "Voice Analysis Monitor (Waiting...)"
        font_scale = 1.2
        thickness = 2
        text_size = cv2.getTextSize(text, self.font, font_scale, thickness)[0]
        x = (self.width - text_size[0]) // 2
        
        cv2.putText(frame, text, (x, 50),
                   self.font, font_scale, (100, 200, 255), thickness)
        
        cv2.line(frame, (50, 70), (self.width - 50, 70), (100, 200, 255), 2)
    
    def draw_listening_status(self, frame):
        """Display microphone status"""
        text = "Waiting for voice input... (Press 'q' to exit, 's' for sound test)"
        cv2.putText(frame, text, (50, 130),
                   self.font, 0.7, (150, 150, 150), 1)
    
    def draw_analysis(self, frame):
        """Display last analysis result"""
        if not self.last_analysis:
            return frame
        
        y = 180
        line_height = 35
        
        # Header
        cv2.putText(frame, "Latest Analysis Result:", (50, y),
                   self.font, 1.0, (200, 200, 200), 2)
        y += line_height + 10
        
        # Text input
        text = self.last_analysis.get('transcribed_text', 'Processing...')
        text = text[:60] + "..." if len(text) > 60 else text
        cv2.putText(frame, f"Input: {text}", (50, y),
                   self.font, 0.7, (200, 200, 200), 1)
        y += line_height
        
        # Situation
        situation = self.last_analysis.get('situation_type', 'Analyzing...')
        cv2.putText(frame, f"Situation: {situation}", (50, y),
                   self.font, 0.7, (150, 200, 255), 1)
        y += line_height
        
        # Priority
        priority = self.last_analysis.get('priority', 'UNKNOWN')
        priority_color = {
            'CRITICAL': (0, 0, 255),
            'HIGH': (0, 165, 255),
            'MEDIUM': (0, 255, 255),
            'LOW': (0, 255, 0)
        }.get(priority, (100, 100, 100))
        
        cv2.putText(frame, f"Priority: {priority}", (50, y),
                   self.font, 0.8, priority_color, 2)
        y += line_height
        
        # Recommended action
        action = self.last_analysis.get('action', 'Waiting')
        cv2.putText(frame, f"Action: {action[:50]}", (50, y),
                   self.font, 0.7, (200, 150, 100), 1)
        
        return frame
    
    def draw_emergency_overlay(self, frame, analysis):
        """Emergency alert overlay"""
        is_emergency = analysis.get('is_emergency', False)
        priority = analysis.get('priority', 'LOW')
        
        if not (is_emergency or priority == 'CRITICAL'):
            return frame
        
        # Red border (blinking)
        thickness = 8 if int(time.time() * 3) % 2 == 0 else 15
        cv2.rectangle(frame, (0, 0), (self.width-1, self.height-1), 
                     (0, 0, 255), thickness)
        
        # Semi-transparent red overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), 
                     (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.05, frame, 0.95, 0, frame)
        
        # Emergency text
        text_main = "EMERGENCY ALERT!"
        font_scale = 2.0
        thickness = 4
        text_size = cv2.getTextSize(text_main, self.font, font_scale, thickness)[0]
        x = (self.width - text_size[0]) // 2
        y = int(self.height * 0.4)
        
        cv2.rectangle(frame, (x - 20, y - text_size[1] - 20),
                     (x + text_size[0] + 20, y + 20),
                     (0, 0, 255), -1)
        cv2.putText(frame, text_main, (x, y),
                   self.font, font_scale, (255, 255, 255), thickness)
        
        # Emergency reason
        reason = analysis.get('emergency_reason', 'Unknown')
        reason_size = cv2.getTextSize(reason, self.font, 1.2, 2)[0]
        x_reason = (self.width - reason_size[0]) // 2
        y_reason = int(self.height * 0.65)
        
        cv2.putText(frame, reason, (x_reason, y_reason),
                   self.font, 1.2, (0, 255, 255), 2)
        
        return frame
    
    def voice_listener_thread(self):
        """Background voice listening thread"""
        print("\nVoice monitoring started (waiting quietly...)")
        
        while True:
            try:
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source, duration=1)
                    audio = recognizer.listen(source, timeout=None, phrase_time_limit=30)
                
                # Convert to text
                text = recognizer.recognize_google(audio, language='ko-KR')
                
                # Exit condition
                if text.lower() in ['quit', 'exit', 'exit program']:
                    print("Program terminated")
                    return
                
                # Print when voice detected
                print(f"\n{'='*70}")
                print(f"[VOICE DETECTED] {datetime.now().strftime('%H:%M:%S')}")
                print(f"{'='*70}")
                print(f"Input: '{text}'")
                
                # ChatGPT analysis
                print(f"Analyzing...")
                analysis = voice_analyzer.analyze_with_llm(text)
                print(f"Analysis complete\n")
                
                priority = analysis.get('priority', 'LOW')
                situation_type = analysis.get('situation_type', 'Unknown')
                action = analysis.get('action', 'Waiting')
                
                # Print results
                print(f"Situation Type: {situation_type}")
                print(f"Priority: {priority}")
                print(f"Recommended Action: {action}")
                
                # Save results
                self.last_analysis = {
                    'transcribed_text': text,
                    'situation_type': situation_type,
                    'priority': priority,
                    'action': action,
                    'is_emergency': analysis.get('is_emergency', False),
                    'emergency_reason': analysis.get('emergency_reason', '')
                }
                
                # Emergency alert (display on screen only)
                is_emergency = analysis.get('is_emergency', False)
                
                if is_emergency or priority == 'CRITICAL':
                    if self.sound_alert_enabled:
                        # Play system alert sound without Korean console messages
                        alert_manager.play_system_alert(repeat=3, delay=0.2)
                        print(f"\nEMERGENCY ALERT! (sound + screen)")
                    else:
                        print(f"\nEMERGENCY ALERT! (displaying on screen)")
                
                # Save log
                log_entry = {
                    "turn": self.turn,
                    "timestamp": datetime.now().isoformat(),
                    "transcribed_text": text,
                    "analysis": analysis
                }
                
                if LOG_FILE.exists():
                    with open(LOG_FILE, 'r', encoding='utf-8') as f:
                        history = json.load(f)
                else:
                    history = []
                
                history.append(log_entry)
                with open(LOG_FILE, 'w', encoding='utf-8') as f:
                    json.dump(history, f, ensure_ascii=False, indent=2)
                
                self.turn += 1
                print(f"Log saved\n")
            
            except sr.UnknownValueError:
                # Continue quietly if voice not recognized
                pass
            except sr.RequestError as e:
                print(f"\nVoice recognition error: {e}\n")
                break
            except KeyboardInterrupt:
                return
            except Exception as e:
                print(f"\nError: {e}\n")
    
    def run(self):
        """Main loop"""
        # Start voice listening thread
        listener_thread = threading.Thread(target=self.voice_listener_thread, daemon=True)
        listener_thread.start()
        
        print("=" * 70)
        print("Voice Analysis Monitor started")
        print("=" * 70)
        
        # OpenCV window loop
        while True:
            frame = self.create_blank_frame()
            
            # Draw UI
            self.draw_title(frame)
            self.draw_listening_status(frame)
            self.draw_analysis(frame)
            
            # Draw emergency overlay if needed
            if self.last_analysis:
                frame = self.draw_emergency_overlay(frame, self.last_analysis)
            
            # Display frame
            cv2.imshow('Voice Analysis Monitor', frame)
            
            # Key input (q to exit)
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                print("Window closed")
                break
            elif key == ord('s'):
                # manual sound test
                print("Sound test triggered")
                alert_manager.play_system_alert(repeat=1, delay=0)
        
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        monitor = VoiceMonitorWindow()
        monitor.run()
    except KeyboardInterrupt:
        print("\nProgram terminated")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
