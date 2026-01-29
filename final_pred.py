# Importing Libraries
import numpy as np
import math
import cv2

import os, sys
import traceback
import pyttsx3
from datetime import datetime
import time
from keras.models import load_model
# from cvzone.HandTrackingModule import HandDetector
from my_hand_tracker import HandDetector
from string import ascii_uppercase
import enchant
import enchant
from collections import deque, Counter
ddd=enchant.Dict("en-US") 
hd = HandDetector(maxHands=2, detectionCon=0.8)  # Changed to detect 2 hands
hd2 = HandDetector(maxHands=2, detectionCon=0.8)  # Changed to detect 2 hands
import tkinter as tk
from PIL import Image, ImageTk

offset=29


os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"


# Application :

class Application:

    def __init__(self):
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.model = load_model('cnn8grps_rad1_model.h5')
        self.speak_engine=pyttsx3.init()
        self.speak_engine.setProperty("rate",100)
        voices=self.speak_engine.getProperty("voices")
        self.speak_engine.setProperty("voice",voices[0].id)

        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0
        self.space_flag=False
        self.next_flag=True
        self.prev_char=""
        self.count=-1
        self.ten_prev_char=[]
        for i in range(10):
            self.ten_prev_char.append(" ")
        
        self.history = deque(maxlen=10)
        
        # Stability/debouncing variables - Enhanced for accuracy
        self.stability_buffer = deque(maxlen=10)  # Store last 10 detections for good smoothing
        self.last_typed_char = ""  # Last character actually typed
        self.min_stability_count = 6  # Need 6 consistent detections before typing (balanced)
        self.char_cooldown = 0  # Cooldown counter to prevent rapid duplicates
        
        # Statistics tracking
        self.session_start = None
        self.total_chars = 0
        self.total_words = 0
        self.char_frequency = {}
        
        # Templates/Quick Phrases
        self.templates = {
            "Greetings": ["Hello", "Hi", "Good morning", "Good afternoon", "Good evening"],
            "Thanks": ["Thank you", "Thanks", "You're welcome"],
            "Questions": ["How are you?", "What's your name?", "Where are you from?"],
            "Common": ["Yes", "No", "Okay", "Please", "Sorry", "Excuse me"]
        }
        
        # Voice options
        self.available_voices = voices
        self.current_voice_index = 0
        
        # Confidence tracking
        self.last_confidence = 0.0
        
        # Dark mode
        self.dark_mode = False


        for i in ascii_uppercase:
            self.ct[i] = 0
        print("Loaded model from disk")


        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("1400x800")  # Increased size for better spacing

        # --- HEADER SECTION (Top Bar) ---
        header_frame = tk.Frame(self.root)
        header_frame.place(x=0, y=0, width=1400, height=80)
        
        # Title
        self.T = tk.Label(header_frame, text="Sign Language Detection", font=("Helvetica", 24, "bold"))
        self.T.place(x=30, y=20)
        
        # Stats Display (Top Center)
        self.stats_label = tk.Label(header_frame, text="Stats: 0 chars | 0 words | 0.0 WPM", font=("Courier", 12))
        self.stats_label.place(x=500, y=30)
        
        # Dark Mode Toggle (Top Right)
        self.dark_mode_btn = tk.Button(header_frame, text="🌙", font=("Arial", 14), command=self.toggle_dark_mode)
        self.dark_mode_btn.place(x=1320, y=20)

        # --- SETTINGS BAR (Below Header) ---
        settings_frame = tk.Frame(self.root)
        settings_frame.place(x=0, y=80, width=1400, height=50)
        
        # Speed Slider
        self.speed_label = tk.Label(settings_frame, text="Speed:", font=("Arial", 11))
        self.speed_label.place(x=50, y=15)
        self.speed_slider = tk.Scale(settings_frame, from_=1, to=3, orient=tk.HORIZONTAL, length=150, command=self.adjust_speed)
        self.speed_slider.set(2)
        self.speed_slider.place(x=110, y=0)
        
        # Confidence Meter
        self.confidence_label = tk.Label(settings_frame, text="Confidence: 0%", font=("Arial", 11))
        self.confidence_label.place(x=300, y=15)
        
        # Templates
        self.template_label = tk.Label(settings_frame, text="Templates:", font=("Arial", 11))
        self.template_label.place(x=950, y=15)
        self.template_var = tk.StringVar(self.root)
        self.template_var.set("Select Phrase...")
        self.template_menu = tk.OptionMenu(settings_frame, self.template_var, 
                                            "Hello", "Hi", "Thank you", "How are you?",
                                            "Good morning", "Yes", "No",
                                            command=self.insert_template)
        self.template_menu.place(x=1050, y=10)

        # --- MAIN DISPLAY AREA (Videos) ---
        # Camera Feed
        self.panel = tk.Label(self.root)
        self.panel.place(x=150, y=150, width=500, height=480)
        
        # Skeleton View
        self.panel2 = tk.Label(self.root)
        self.panel2.place(x=750, y=150, width=400, height=400)
        


        # --- OUTPUT AREA (Bottom) ---
        # Sentence Display - Editable Text Box
        self.T3 = tk.Label(self.root, text="Sentence:", font=("Arial", 16, "bold"))
        self.T3.place(x=50, y=650)
        
        # Frame for Text + Scrollbar
        self.text_frame = tk.Frame(self.root)
        self.text_frame.place(x=180, y=650, width=900, height=50)
        
        self.scrollbar = tk.Scrollbar(self.text_frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.panel5 = tk.Text(self.text_frame, font=("Courier", 20), height=1, width=50,
                              yscrollcommand=self.scrollbar.set)
        self.panel5.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=self.panel5.yview)
        
        # Suggestions Area
        self.T4 = tk.Label(self.root, text="Suggestions:", fg="red", font=("Arial", 14, "bold"))
        self.T4.place(x=50, y=700)
        
        self.b1=tk.Button(self.root, font=("Courier", 12), command=self.action1)
        self.b1.place(x=180,y=700)
        self.b2 = tk.Button(self.root, font=("Courier", 12), command=self.action2)
        self.b2.place(x=300, y=700)
        self.b3 = tk.Button(self.root, font=("Courier", 12), command=self.action3)
        self.b3.place(x=420, y=700)
        self.b4 = tk.Button(self.root, font=("Courier", 12), command=self.action4)
        self.b4.place(x=540, y=700)

        # --- CONTROL PANEL (Bottom Right) ---
        # Feature Buttons
        btn_y = 650
        
        self.save_btn = tk.Button(self.root, text="💾 Save", font=("Arial", 12), command=self.save_to_file)
        self.save_btn.place(x=1100, y=btn_y)
        
        self.copy_btn = tk.Button(self.root, text="📋 Copy", font=("Arial", 12), command=self.copy_to_clipboard)
        self.copy_btn.place(x=1200, y=btn_y)
        
        self.voice_btn = tk.Button(self.root, text="🔊 Voice", font=("Arial", 12), command=self.change_voice)
        self.voice_btn.place(x=1300, y=btn_y)
        
        # Main Controls
        ctrl_y = 700
        
        self.backspace = tk.Button(self.root, text="⌫", font=("Arial", 20, "bold"), bg="#ffcccc", command=self.backspace_fun)
        self.backspace.place(x=1100, y=ctrl_y, height=45, width=90)
        
        self.clear = tk.Button(self.root, text="Clear", font=("Arial", 12), command=self.clear_fun)
        self.clear.place(x=1200, y=ctrl_y, height=45, width=90)
        
        self.speak = tk.Button(self.root, text="Speak", font=("Arial", 12, "bold"), bg="#ccffcc", command=self.speak_fun)
        self.speak.place(x=1300, y=ctrl_y, height=45, width=90)
        
        # Store layout references for dark mode
        self.layout_frames = [header_frame, settings_frame]





        self.str = " "
        self.ccc=0
        self.word = " "
        self.current_symbol = "C"
        self.photo = "Empty"


        self.word1=" "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "

        self.video_loop()

    def video_loop(self):
        try:
            ok, frame = self.vs.read()
            if not ok:
                # Handle case where frame is not read (e.g. camera disconnected)
                self.root.after(10, self.video_loop)
                return

            cv2image = cv2.flip(frame, 1)
            if cv2image.any(): # Check if image is not empty
                hands, _ = hd.findHands(cv2image, draw=False, flipType=True)
                cv2image_copy = np.array(cv2image)
                
                # Draw landmarks on the main image
                if hands: # Check if hands list is not empty
                    hand_main = hands[0] # Access the first hand's dictionary
                    lmList_main = hand_main['lmList']
                    
                    # Define connections for drawing
                    connections = [
                        (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
                        (5, 6), (6, 7), (7, 8),              # Index
                        (9, 10), (10, 11), (11, 12),         # Middle
                        (13, 14), (14, 15), (15, 16),        # Ring
                        (17, 18), (18, 19), (19, 20),        # Pinky
                        (0, 5), (5, 9), (9, 13), (13, 17), (0, 17) # Palm
                    ]
                    
                    for p1, p2 in connections:
                         pt1 = (lmList_main[p1][0], lmList_main[p1][1])
                         pt2 = (lmList_main[p2][0], lmList_main[p2][1])
                         cv2.line(cv2image, pt1, pt2, (0, 255, 0), 3)
                    
                    for lm in lmList_main:
                        cv2.circle(cv2image, (lm[0], lm[1]), 3, (0, 0, 255), -1)

                cv2image_rgb = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
                self.current_image = Image.fromarray(cv2image_rgb)
                imgtk = ImageTk.PhotoImage(image=self.current_image)
                self.panel.imgtk = imgtk
                self.panel.config(image=imgtk)

                if hands: # Check if hands list is not empty
                    hand = hands[0]
                    x, y, w, h = hand['bbox']
                    # Ensure crop coordinates are within bounds
                    y1 = max(0, y - offset)
                    y2 = min(cv2image_copy.shape[0], y + h + offset)
                    x1 = max(0, x - offset)
                    x2 = min(cv2image_copy.shape[1], x + w + offset)
                    
                    image = cv2image_copy[y1:y2, x1:x2]

                    white = cv2.imread("white.jpg")
                    if white is None:
                        # Fallback if white.jpg is missing or path issues
                        white = np.ones((400, 400, 3), np.uint8) * 255
                        
                    # img_final=img_final1=img_final2=0
                    if image.size != 0:
                        handz, _ = hd2.findHands(image, draw=False, flipType=True)
                        self.ccc += 1
                        if handz: # Check if handz list is not empty
                            hand = handz[0]
                            self.pts = hand['lmList']
                            # x1,y1,w1,h1=hand['bbox']

                            os = ((400 - w) // 2) - 15
                            os1 = ((400 - h) // 2) - 15
            self.panel.config(image=imgtk)

            if hands:
                # #print(" --------- lmlist=",hands[1])
                hand = hands[0]
                x, y, w, h = hand['bbox']
                image = cv2image_copy[y - offset:y + h + offset, x - offset:x + w + offset]

                white = cv2.imread("white.jpg")
                if white is None:
                    white = np.ones((400, 400, 3), np.uint8) * 255
                # img_final=img_final1=img_final2=0

                handz, _ = hd2.findHands(image, draw=False, flipType=True)
                print(" ", self.ccc)
                self.ccc += 1
                if handz:
                    hand = handz[0]
                    self.pts = hand['lmList']
                    # x1,y1,w1,h1=hand['bbox']

                    os = ((400 - w) // 2) - 15
                    os1 = ((400 - h) // 2) - 15
                    for t in range(0, 4, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                 (0, 255, 0), 3)
                    for t in range(5, 8, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                 (0, 255, 0), 3)
                    for t in range(9, 12, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                 (0, 255, 0), 3)
                    for t in range(13, 16, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                 (0, 255, 0), 3)
                    for t in range(17, 20, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                 (0, 255, 0), 3)
                    cv2.line(white, (self.pts[5][0] + os, self.pts[5][1] + os1), (self.pts[9][0] + os, self.pts[9][1] + os1), (0, 255, 0),
                             3)
                    cv2.line(white, (self.pts[9][0] + os, self.pts[9][1] + os1), (self.pts[13][0] + os, self.pts[13][1] + os1), (0, 255, 0),
                             3)
                    cv2.line(white, (self.pts[13][0] + os, self.pts[13][1] + os1), (self.pts[17][0] + os, self.pts[17][1] + os1),
                             (0, 255, 0), 3)
                    cv2.line(white, (self.pts[0][0] + os, self.pts[0][1] + os1), (self.pts[5][0] + os, self.pts[5][1] + os1), (0, 255, 0),
                             3)
                    cv2.line(white, (self.pts[0][0] + os, self.pts[0][1] + os1), (self.pts[17][0] + os, self.pts[17][1] + os1), (0, 255, 0),
                             3)

                    for i in range(21):
                        cv2.circle(white, (self.pts[i][0] + os, self.pts[i][1] + os1), 2, (0, 0, 255), 1)

                    res=white
                    self.predict(res)

                    self.current_image2 = Image.fromarray(res)

                    imgtk = ImageTk.PhotoImage(image=self.current_image2)

                    self.panel2.imgtk = imgtk
                    self.panel2.config(image=imgtk)

                    #self.panel4.config(text=self.word, font=("Courier", 30))

                    self.b1.config(text=self.word1, font=("Courier", 12), wraplength=100, command=self.action1)
                    self.b2.config(text=self.word2, font=("Courier", 12), wraplength=100,  command=self.action2)
                    self.b3.config(text=self.word3, font=("Courier", 12), wraplength=100,  command=self.action3)
                    self.b4.config(text=self.word4, font=("Courier", 12), wraplength=100,  command=self.action4)

                    self.b4.config(text=self.word4, font=("Courier", 12), wraplength=100,  command=self.action4)

            # self.panel5.config(text=self.str, font=("Courier", 20), wraplength=900) # Removed for Text widget compatibility
        except Exception:
            print("==", traceback.format_exc())
        finally:
            self.root.after(1, self.video_loop)

    def distance(self,x,y):
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

    def set_text(self, text):
        self.panel5.delete("1.0", tk.END)
        self.panel5.insert(tk.END, text)
        self.str = text
        self.update_statistics()

    def action1(self):
        current = self.panel5.get("1.0", "end-1c")
        idx_space = current.rfind(" ")
        idx_word = current.find(self.word, idx_space)
        current = current[:idx_word]
        current = current + self.word1.upper()
        self.set_text(current)


    def action2(self):
        current = self.panel5.get("1.0", "end-1c")
        idx_space = current.rfind(" ")
        idx_word = current.find(self.word, idx_space)
        current = current[:idx_word]
        current = current + self.word2.upper()
        self.set_text(current)


    def action3(self):
        current = self.panel5.get("1.0", "end-1c")
        idx_space = current.rfind(" ")
        idx_word = current.find(self.word, idx_space)
        current = current[:idx_word]
        current = current + self.word3.upper()
        self.set_text(current)



    def action4(self):
        current = self.panel5.get("1.0", "end-1c")
        idx_space = current.rfind(" ")
        idx_word = current.find(self.word, idx_space)
        current = current[:idx_word]
        current = current + self.word4.upper()
        self.set_text(current)


    def speak_fun(self):
        text_to_speak = self.panel5.get("1.0", "end-1c")
        self.speak_engine.say(text_to_speak)
        self.speak_engine.runAndWait()


    # === HELPER METHODS FOR TEXT WIDGET ===
    def add_char_to_text(self, char):
        self.panel5.insert(tk.END, char)
        self.panel5.see(tk.END)
        self.str = self.panel5.get("1.0", "end-1c")
        self.update_statistics()

    def backspace_fun(self):
        self.panel5.delete("end-2c", "end-1c")
        self.str = self.panel5.get("1.0", "end-1c")
        self.update_statistics()
        # Reset last typed character to allow retyping the same letter
        self.last_typed_char = ""
        self.char_cooldown = 5

    def clear_fun(self):
        self.panel5.delete("1.0", tk.END)
        self.str = " "
        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "
        
    # === NEW FEATURE FUNCTIONS ===
    
    def save_to_file(self):
        """Save sentence to a text file"""
        current_text = self.panel5.get("1.0", "end-1c")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sign_language_output_{timestamp}.txt"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(current_text)
            print(f"✅ Saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving: {e}")
    
    def copy_to_clipboard(self):
        """Copy sentence to clipboard"""
        current_text = self.panel5.get("1.0", "end-1c")
        self.root.clipboard_clear()
        self.root.clipboard_append(current_text)
        self.root.update()
        print("✅ Copied to clipboard!")
    
    def adjust_speed(self, value):
        """Adjust detection speed/sensitivity"""
        speed = int(value)
        if speed == 1:  # Fast
            self.min_stability_count = 4
            self.char_cooldown = 5
            self.stability_buffer = deque(maxlen=8)
        elif speed == 2:  # Medium (default)
            self.min_stability_count = 6
            self.char_cooldown = 8
            self.stability_buffer = deque(maxlen=10)
        else:  # Slow (speed == 3)
            self.min_stability_count = 8
            self.char_cooldown = 12
            self.stability_buffer = deque(maxlen=12)
        print(f"Speed set to: {'Fast' if speed==1 else 'Medium' if speed==2 else 'Slow'}")
    
    def insert_template(self, template_text):
        """Insert template phrase"""
        if template_text and template_text.strip():
            self.panel5.insert(tk.END, " " + template_text.upper())
            self.str = self.panel5.get("1.0", "end-1c")
            self.update_statistics()
    
    def toggle_dark_mode(self):
        """Toggle dark/light mode"""
        self.dark_mode = not self.dark_mode
        if self.dark_mode:
            # Dark mode colors
            bg_color = "#2b2b2b"
            fg_color = "#ffffff"
            self.dark_mode_btn.config(text="☀️")
        else:
            # Light mode colors  
            bg_color = "#f0f0f0"
            fg_color = "#000000"
            self.dark_mode_btn.config(text="🌙")
        
        # Update all widgets
        self.root.config(bg=bg_color)
        
        # update frames
        for frame in self.layout_frames:
            frame.config(bg=bg_color)
            
        self.panel.config(bg=bg_color)
        self.panel2.config(bg=bg_color)
        self.panel5.config(bg=bg_color, fg=fg_color)
        for widget in [self.T, self.T3, self.T4, self.stats_label, 
                       self.confidence_label, self.speed_label, self.template_label]:
            widget.config(bg=bg_color, fg=fg_color)
    
    def change_voice(self):
        """Cycle through available voices"""
        self.current_voice_index = (self.current_voice_index + 1) % len(self.available_voices)
        self.speak_engine.setProperty("voice", self.available_voices[self.current_voice_index].id)
        voice_name = self.available_voices[self.current_voice_index].name
        print(f"🔊 Voice changed to: {voice_name}")
    
    def update_statistics(self):
        """Update statistics display"""
        if self.session_start is None:
            self.session_start = time.time()
        
        self.total_chars = len(self.str.strip())
        self.total_words = len(self.str.strip().split())
        
        # Calculate WPM
        elapsed_time = time.time() - self.session_start
        if elapsed_time > 0:
            wpm = (self.total_words / elapsed_time) * 60
        else:
            wpm = 0.0
        
        # Update display
        self.stats_label.config(
            text=f"Stats: {self.total_chars} chars | {self.total_words} words | {wpm:.1f} WPM"
        )

    def predict(self, test_image):
        white=test_image
        white = white.reshape(1, 400, 400, 3)
        prob = np.array(self.model.predict(white)[0], dtype='float32')
        
        # Track confidence (max probability)
        self.last_confidence = np.max(prob) * 100
        self.confidence_label.config(
            text=f"Confidence: {self.last_confidence:.1f}%",
            fg="green" if self.last_confidence > 80 else "orange" if self.last_confidence > 60 else "red"
        )
        
        ch1 = np.argmax(prob, axis=0)
        prob[ch1] = 0
        ch2 = np.argmax(prob, axis=0)
        prob[ch2] = 0
        ch3 = np.argmax(prob, axis=0)
        prob[ch3] = 0

        pl = [ch1, ch2]

        # condition for [Aemnst]
        l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
             [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
             [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 0

        # condition for [o][s]
        l = [[2, 2], [2, 1]]
        if pl in l:
            if (self.pts[5][0] < self.pts[4][0]):
                ch1 = 0
                print("++++++++++++++++++")
                # print("00000")

        # condition for [c0][aemnst]
        l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[4][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][
                0] and self.pts[0][0] > self.pts[20][0]) and self.pts[5][0] > self.pts[4][0]:
                ch1 = 2

        # condition for [c0][aemnst]
        l = [[6, 0], [6, 6], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) < 52:
                ch1 = 2


        # condition for [gh][bdfikruvw]
        l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]

        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1] and self.pts[0][0] < self.pts[8][
                0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 3



        # con for [gh][l]
        l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 3

        # con for [gh][pqz]
        l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[2][1] + 15 < self.pts[16][1]:
                ch1 = 3

        # con for [l][x]
        l = [[6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) > 55:
                ch1 = 4

        # con for [l][d]
        l = [[1, 4], [1, 6], [1, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) > 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 4

        # con for [l][gh]
        l = [[3, 6], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[0][0]):
                ch1 = 4

        # con for [l][c0]
        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4

        # con for [l][c0]
        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4

        # con for [gh][z]
        l = [[3, 6], [3, 5], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]) and self.pts[4][1] > self.pts[10][1]:
                ch1 = 5

        # con for [gh][pq]
        l = [[3, 2], [3, 1], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][1] + 17 > self.pts[8][1] and self.pts[4][1] + 17 > self.pts[12][1] and self.pts[4][1] + 17 > self.pts[16][1] and self.pts[4][
                1] + 17 > self.pts[20][1]:
                ch1 = 5

        # con for [l][pqz]
        l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 5

        # con for [pqz][aemnst]
        l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 5

        # con for [pqz][yj]
        l = [[5, 7], [5, 2], [5, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[3][0] < self.pts[0][0]:
                ch1 = 7

        # con for [l][yj]
        l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] < self.pts[8][1]:
                ch1 = 7

        # con for [x][yj]
        l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] > self.pts[20][1]:
                ch1 = 7

        # condition for [x][aemnst]
        l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] > self.pts[16][0]:
                ch1 = 6


        # condition for [yj][x]
        print("2222  ch1=+++++++++++++++++", ch1, ",", ch2)
        l = [[7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] < self.pts[20][1] and self.pts[8][1] < self.pts[10][1]:
                ch1 = 6

        # condition for [c0][x]
        l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) > 50:
                ch1 = 6

        # con for [l][x]

        l = [[4, 6], [4, 2], [4, 1], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) < 60:
                ch1 = 6

        # con for [x][d]
        l = [[1, 4], [1, 6], [1, 0], [1, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 > 0:
                ch1 = 6

        # con for [b][pqz]
        l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0],
             [6, 3], [6, 4], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 1

        # con for [f][pqz]
        l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1],
             [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and
                    self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and
                    self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        # con for [d][pqz]
        fg = 19
        # print("_________________ch1=",ch1," ch2=",ch2)
        l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[4][1] > self.pts[14][1]):
                ch1 = 1

        l = [[4, 1], [4, 2], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) < 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 1

        l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[14][1] < self.pts[4][1]):
                ch1 = 1

        l = [[6, 6], [6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 < 0:
                ch1 = 1

        # con for [i][pqz]
        l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] > self.pts[20][1])):
                ch1 = 1

        # con for [yj][bfdi]
        l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[5][0] + 15) and (
            (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
             self.pts[18][1] > self.pts[20][1])):
                ch1 = 7

        # con for [uvr]
        l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1])) and self.pts[4][1] > self.pts[14][1]:
                ch1 = 1

        # con for [w]
        fg = 13
        l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if not (self.pts[0][0] + fg < self.pts[8][0] and self.pts[0][0] + fg < self.pts[12][0] and self.pts[0][0] + fg < self.pts[16][0] and
                    self.pts[0][0] + fg < self.pts[20][0]) and not (
                    self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][
                0]) and self.distance(self.pts[4], self.pts[11]) < 50:
                ch1 = 1

        # con for [w]

        l = [[5, 0], [5, 5], [0, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1]:
                ch1 = 1

        # -------------------------condn for 8 groups  ends

        # -------------------------condn for subgroups  starts
        #
        if ch1 == 0:
            ch1 = 'S'
            if self.pts[4][0] < self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0]:
                ch1 = 'A'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][
                0] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'T'
            if self.pts[4][1] > self.pts[8][1] and self.pts[4][1] > self.pts[12][1] and self.pts[4][1] > self.pts[16][1] and self.pts[4][1] > self.pts[20][1]:
                ch1 = 'E'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][0] > self.pts[14][0] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'M'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][1] < self.pts[18][1] and self.pts[4][1] < self.pts[14][1]:
                ch1 = 'N'

        if ch1 == 2:
            if self.distance(self.pts[12], self.pts[4]) > 42:
                ch1 = 'C'
            else:
                ch1 = 'O'

        if ch1 == 3:
            if (self.distance(self.pts[8], self.pts[12])) > 72:
                ch1 = 'G'
            else:
                ch1 = 'H'

        if ch1 == 7:
            if self.distance(self.pts[8], self.pts[4]) > 42:
                ch1 = 'Y'
            else:
                ch1 = 'J'

        if ch1 == 4:
            ch1 = 'L'

        if ch1 == 6:
            ch1 = 'X'

        if ch1 == 5:
            if self.pts[4][0] > self.pts[12][0] and self.pts[4][0] > self.pts[16][0] and self.pts[4][0] > self.pts[20][0]:
                if self.pts[8][1] < self.pts[5][1]:
                    ch1 = 'Z'
                else:
                    ch1 = 'Q'
            else:
                ch1 = 'P'

        if ch1 == 1:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'B'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 'D'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'F'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'I'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 'W'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]) and self.pts[4][1] < self.pts[9][1]:
                ch1 = 'K'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) < 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 'U'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) >= 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]) and (self.pts[4][1] > self.pts[9][1]):
                ch1 = 'V'

            if (self.pts[8][0] > self.pts[12][0]) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 'R'
            
            # --- UPDATED GESTURE LOGIC START ---
            # Custom Gesture: Thumbs Up -> Space
            # Custom Gesture: Thumbs Down/Left -> Backspace
            
            fingers_folded = True
            wrist_pt = self.pts[0]
            # Check if Index, Middle, Ring, Pinky are folded (Tip closer to wrist)
            for tip_idx in [8, 12, 16, 20]:
                if self.distance(self.pts[tip_idx], wrist_pt) > 130: # Threshold for extended finger
                    fingers_folded = False
                    break
            
            # Check thumb extension
            thumb_extended = self.distance(self.pts[4], wrist_pt) > 100
            
            if fingers_folded and thumb_extended:
                thumb_tip_y = self.pts[4][1]
                thumb_base_y = self.pts[2][1]
                thumb_tip_x = self.pts[4][0]
                thumb_base_x = self.pts[2][0]
                
                # Thumbs Up (Tip Higher than Base) -> Space
                if thumb_tip_y < thumb_base_y - 20: 
                    ch1 = ' '
                    print("👍 Thumbs Up -> Space")
                
                # Thumbs Down OR Left (Tip Lower or Left of Base) -> Backspace
                elif (thumb_tip_y > thumb_base_y + 20) or (thumb_tip_x < thumb_base_x - 20):
                    ch1 = 'Backspace'
                    print("👎/👈 Reverse Thumb -> Backspace")
             # --- UPDATED GESTURE LOGIC END ---

        if ch1 == 1 or ch1 =='E' or ch1 =='S' or ch1 =='X' or ch1 =='Y' or ch1 =='B':
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1=" "



        print(self.pts[4][0] < self.pts[5][0])
        if ch1 == 'E' or ch1=='Y' or ch1=='B':
            if (self.pts[4][0] < self.pts[5][0]) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1="next"


        if ch1 == 'Next' or 'B' or 'C' or 'H' or 'F' or 'X':
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and (self.pts[4][1] < self.pts[8][1] and self.pts[4][1] < self.pts[12][1] and self.pts[4][1] < self.pts[16][1] and self.pts[4][1] < self.pts[20][1]) and (self.pts[4][1] < self.pts[6][1] and self.pts[4][1] < self.pts[10][1] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]):
                ch1 = 'Backspace'


        # Add current detection to stability buffer
        self.stability_buffer.append(ch1)
        
        # Decrease cooldown
        if self.char_cooldown > 0:
            self.char_cooldown -= 1
        
        # Check if we have enough stable detections
        if len(self.stability_buffer) >= self.min_stability_count:
            # Count occurrences of each character in the buffer
            char_counts = Counter(self.stability_buffer)
            most_common_char, count = char_counts.most_common(1)[0]
            
            # If the character is stable enough and different from last typed
            if count >= self.min_stability_count and self.char_cooldown == 0:
                # Handle special gestures
                if most_common_char == "next":
                    if self.last_typed_char != "next":
                        # Add the previous stable character
                        prev_c = self.ten_prev_char[(self.count-2)%10]
                        if prev_c not in ["next", "Backspace", " ", ""]:
                            self.add_char_to_text(prev_c)
                            self.last_typed_char = "next"
                            self.char_cooldown = 8
                
                elif most_common_char == "Backspace":
                    if self.last_typed_char != "Backspace":
                        self.backspace_fun()
                        self.last_typed_char = "Backspace"
                        self.char_cooldown = 8
                
                elif most_common_char == " ":
                    if self.last_typed_char != " ":
                        self.add_char_to_text(" ")
                        self.last_typed_char = " "
                        self.char_cooldown = 8
                
                # Handle regular letters
                elif most_common_char and most_common_char != self.last_typed_char:
                    # Only add if it's a valid letter
                    if most_common_char in ascii_uppercase:
                        self.add_char_to_text(most_common_char)
                        self.last_typed_char = most_common_char
                        self.char_cooldown = 8  # Set cooldown to prevent rapid duplicates
        
        # Update prev_char
        self.prev_char = ch1

        self.count += 1
        self.ten_prev_char[self.count%10]=ch1
        
        # Sync self.str with widget content for suggestions
        self.str = self.panel5.get("1.0", "end-1c")


        if len(self.str.strip())!=0:
            st=self.str.rfind(" ")
            ed=len(self.str)
            word=self.str[st+1:ed]
            self.word=word
            if len(word.strip())!=0:
                ddd.check(word)
                lenn = len(ddd.suggest(word))
                if lenn >= 4:
                    self.word4 = ddd.suggest(word)[3]

                if lenn >= 3:
                    self.word3 = ddd.suggest(word)[2]

                if lenn >= 2:
                    self.word2 = ddd.suggest(word)[1]

                if lenn >= 1:
                    self.word1 = ddd.suggest(word)[0]
            else:
                self.word1 = " "
                self.word2 = " "
                self.word3 = " "
                self.word4 = " "


    def destructor(self):
        print(self.ten_prev_char)
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()


print("Starting Application...")

(Application()).root.mainloop()
