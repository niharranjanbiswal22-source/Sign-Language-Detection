import cv2
import numpy as np
import math
import os
import sqlite3
import enchant
import random
import string
import io
from collections import deque, Counter
from datetime import datetime
from functools import wraps
from flask import Flask, render_template_string, Response, jsonify, request, redirect, url_for, session, flash, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from PIL import Image, ImageDraw

app = Flask(__name__)
app.secret_key = "sign_ai_pro_ultra_secret_key_2026_#$%"
DB_NAME = "app_data.db"

# Initialize SQLite Database
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            sentence TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Login Required Decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Initialize AI & CV Components
ddd = enchant.Dict("en-US")
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)
model = load_model('cnn8grps_rad1_model.h5')
offset = 29

class SignPredictor:
    def __init__(self):
        self.current_symbol = "-"
        self.sentence = ""
        self.word = ""
        self.suggestions = []
        self.prev_char = ""
        self.count = 0
        self.ten_prev_char = [" "] * 10
        self.pts = []
        self.pred_buffer = deque(maxlen=5)

    def distance(self, x, y):
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

    def predict_raw(self, test_image):
        white = test_image.reshape(1, 400, 400, 3)
        prob = np.array(model.predict(white, verbose=0)[0], dtype='float32')
        ch1 = np.argmax(prob, axis=0)
        prob[ch1] = 0
        ch2 = np.argmax(prob, axis=0)
        prob[ch2] = 0
        ch3 = np.argmax(prob, axis=0)
        prob[ch3] = 0

        pl = [ch1, ch2]

        l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
             [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
             [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 0

        l = [[2, 2], [2, 1]]
        if pl in l:
            if (self.pts[5][0] < self.pts[4][0]):
                ch1 = 0

        l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[4][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and self.pts[5][0] > self.pts[4][0]:
                ch1 = 2

        l = [[6, 0], [6, 6], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) < 52:
                ch1 = 2

        l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1] and self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 3

        l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 3

        l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[2][1] + 15 < self.pts[16][1]:
                ch1 = 3

        l = [[6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) > 55:
                ch1 = 4

        l = [[1, 4], [1, 6], [1, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) > 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 4

        l = [[3, 6], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[0][0]):
                ch1 = 4

        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4

        l = [[3, 6], [3, 5], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and self.pts[4][1] > self.pts[10][1]:
                ch1 = 5

        l = [[3, 2], [3, 1], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][1] + 17 > self.pts[8][1] and self.pts[4][1] + 17 > self.pts[12][1] and self.pts[4][1] + 17 > self.pts[16][1] and self.pts[4][1] + 17 > self.pts[20][1]:
                ch1 = 5

        l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 5

        l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 5

        l = [[5, 7], [5, 2], [5, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[3][0] < self.pts[0][0]:
                ch1 = 7

        l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] < self.pts[8][1]:
                ch1 = 7

        l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] > self.pts[20][1]:
                ch1 = 7

        l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] > self.pts[16][0]:
                ch1 = 6

        l = [[7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] < self.pts[20][1] and self.pts[8][1] < self.pts[10][1]:
                ch1 = 6

        l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) > 50:
                ch1 = 6

        l = [[4, 6], [4, 2], [4, 1], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) < 60:
                ch1 = 6

        l = [[1, 4], [1, 6], [1, 0], [1, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 > 0:
                ch1 = 6

        l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0], [6, 3], [6, 4], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        fg = 19
        l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[4][1] > self.pts[14][1]):
                ch1 = 1

        l = [[4, 1], [4, 2], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) < 50) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 1

        l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[14][1] < self.pts[4][1]):
                ch1 = 1

        l = [[6, 6], [6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 < 0:
                ch1 = 1

        l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1])):
                ch1 = 1

        l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[5][0] + 15) and ((self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1])):
                ch1 = 7

        l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1])) and self.pts[4][1] > self.pts[14][1]:
                ch1 = 1

        fg = 13
        l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if not (self.pts[0][0] + fg < self.pts[8][0] and self.pts[0][0] + fg < self.pts[12][0] and self.pts[0][0] + fg < self.pts[16][0] and self.pts[0][0] + fg < self.pts[20][0]) and not (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and self.distance(self.pts[4], self.pts[11]) < 50:
                ch1 = 1

        l = [[5, 0], [5, 5], [0, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1]:
                ch1 = 1

        # Subgroups mapping
        if ch1 == 0:
            ch1 = 'S'
            if self.pts[4][0] < self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0]:
                ch1 = 'A'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]:
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
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 'B'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'D'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 'F'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 'I'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'W'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and self.pts[4][1] < self.pts[9][1]:
                ch1 = 'K'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) < 8) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'U'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) >= 8) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[4][1] > self.pts[9][1]):
                ch1 = 'V'
            if (self.pts[8][0] > self.pts[12][0]) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'R'

        if ch1 in [1, 'E', 'S', 'X', 'Y', 'B']:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = " "

        if ch1 in ['E', 'Y', 'B']:
            if (self.pts[4][0] < self.pts[5][0]) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = "next"

        if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and (self.pts[4][1] < self.pts[8][1] and self.pts[4][1] < self.pts[12][1] and self.pts[4][1] < self.pts[16][1] and self.pts[4][1] < self.pts[20][1]) and (self.pts[4][1] < self.pts[6][1] and self.pts[4][1] < self.pts[10][1] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]):
            ch1 = 'Backspace'

        return str(ch1)

    def process_frame_prediction(self, test_image):
        raw_pred = self.predict_raw(test_image)
        self.pred_buffer.append(raw_pred)

        counts = Counter(self.pred_buffer)
        most_common_char, freq = counts.most_common(1)[0]

        if freq >= 3:
            self.current_symbol = most_common_char
            
            if most_common_char == "next" and self.prev_char != "next":
                target = self.ten_prev_char[(self.count - 2) % 10]
                if target == "Backspace":
                    self.sentence = self.sentence[0:-1]
                elif target != "next":
                    self.sentence += target
            elif most_common_char == "Backspace" and self.prev_char != "Backspace":
                self.sentence = self.sentence[0:-1]

            self.prev_char = most_common_char
            self.count += 1
            self.ten_prev_char[self.count % 10] = most_common_char

            if len(self.sentence.strip()) > 0:
                st = self.sentence.rfind(" ")
                word = self.sentence[st + 1:]
                self.word = word
                if len(word.strip()) > 0:
                    sug = ddd.suggest(word)
                    self.suggestions = [s.upper() for s in sug[:4]]
                else:
                    self.suggestions = []
            else:
                self.suggestions = []

predictor = SignPredictor()

# --- CAPTCHA GENERATOR ROUTE ---
@app.route('/captcha_image')
def captcha_image():
    chars = ''.join(random.choices(string.ascii_uppercase + '23456789', k=5))
    session['captcha_text'] = chars

    img = Image.new('RGB', (160, 50), color=(15, 23, 42))
    draw = ImageDraw.Draw(img)

    for _ in range(5):
        x1 = random.randint(0, 160)
        y1 = random.randint(0, 50)
        x2 = random.randint(0, 160)
        y2 = random.randint(0, 50)
        draw.line([(x1, y1), (x2, y2)], fill=(0, 242, 254), width=1)

    for _ in range(40):
        x = random.randint(0, 160)
        y = random.randint(0, 50)
        draw.point((x, y), fill=(79, 172, 254))

    for i, char in enumerate(chars):
        x = 18 + i * 26
        y = random.randint(10, 16)
        draw.text((x, y), char, fill=(255, 255, 255))

    buffer = io.BytesIO()
    img.save(buffer, 'PNG')
    buffer.seek(0)
    return Response(buffer.getvalue(), mimetype='image/png')

# --- HTML TEMPLATES ---

LOGIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sign AI Pro - Secure Login</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@600&display=swap" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css">
    <style>
        body {
            background: linear-gradient(135deg, #070a12 0%, #0d1527 50%, #050810 100%);
            color: #f1f5f9;
            font-family: 'Inter', sans-serif;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .auth-card {
            background: rgba(19, 28, 49, 0.85);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 40px;
            width: 100%;
            max-width: 440px;
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5), 0 0 40px rgba(0, 242, 254, 0.15);
        }
        .brand-title {
            background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }
        .form-control-custom {
            background: rgba(9, 13, 22, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.12);
            color: #f8fafc;
            border-radius: 12px;
            padding: 12px 16px;
            transition: all 0.25s;
        }
        .form-control-custom:focus {
            background: rgba(9, 13, 22, 0.9);
            border-color: #00f2fe;
            box-shadow: 0 0 15px rgba(0, 242, 254, 0.3);
            color: white;
        }
        .btn-glow {
            background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
            color: #070a12;
            font-weight: 700;
            border: none;
            border-radius: 12px;
            padding: 14px;
            transition: all 0.25s;
        }
        .btn-glow:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(0, 242, 254, 0.4);
            color: #070a12;
        }
        .captcha-box {
            background: #0f172a;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(0, 242, 254, 0.3);
        }
        .dev-badge {
            color: #38bdf8;
            font-weight: 600;
            letter-spacing: 0.5px;
        }
    </style>
</head>
<body>
    <div class="auth-card text-center">
        <div class="fs-1 mb-1">🤟</div>
        <h2 class="brand-title mb-1">Sign AI Pro</h2>
        <p class="text-secondary small mb-3">Secure Login with Verification</p>

        {% with messages = get_flashed_messages(with_categories=true) %}
          {% if messages %}
            {% for category, message in messages %}
              <div class="alert alert-{{ 'danger' if category == 'error' else 'success' }} py-2 small mb-3 border-0 rounded-3">
                {{ message }}
              </div>
            {% endfor %}
          {% endif %}
        {% endwith %}

        <form action="/login" method="POST">
            <div class="mb-3 text-start">
                <label class="form-label text-secondary small fw-semibold">Username</label>
                <input type="text" name="username" class="form-control form-control-custom" placeholder="Username (e.g. admin)" required>
            </div>

            <div class="mb-3 text-start">
                <label class="form-label text-secondary small fw-semibold">Password</label>
                <input type="password" name="password" class="form-control form-control-custom" placeholder="••••••••" required>
            </div>

            <!-- CAPTCHA Box -->
            <div class="mb-4 text-start">
                <label class="form-label text-secondary small fw-semibold d-block">Security CAPTCHA</label>
                <div class="d-flex align-items-center gap-2 mb-2">
                    <div class="captcha-box">
                        <img id="captcha-img" src="/captcha_image" alt="CAPTCHA" height="42">
                    </div>
                    <button type="button" class="btn btn-outline-secondary btn-sm rounded-3 py-2 px-3" onclick="refreshCaptcha()">
                        <i class="bi bi-arrow-clockwise"></i>
                    </button>
                </div>
                <input type="text" name="captcha" class="form-control form-control-custom font-monospace text-uppercase" placeholder="Enter CAPTCHA Code" required autocomplete="off">
            </div>

            <button type="submit" class="btn btn-glow w-100 mb-3">Verify & Sign In</button>
        </form>

        <p class="text-secondary small mb-3">Don't have an account? <a href="/register" class="text-info text-decoration-none fw-semibold">Create Account</a></p>
        
        <div class="pt-3 border-top border-secondary border-opacity-25 text-center">
            <span class="dev-badge small"><i class="bi bi-code-slash me-1"></i>Developed by DEEPAK SETHI</span>
        </div>
    </div>

    <script>
        function refreshCaptcha() {
            document.getElementById('captcha-img').src = '/captcha_image?' + new Date().getTime();
        }
    </script>
</body>
</html>
"""

REGISTER_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sign AI Pro - Register</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css">
    <style>
        body {
            background: linear-gradient(135deg, #070a12 0%, #0d1527 50%, #050810 100%);
            color: #f1f5f9;
            font-family: 'Inter', sans-serif;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .auth-card {
            background: rgba(19, 28, 49, 0.85);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 40px;
            width: 100%;
            max-width: 460px;
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5), 0 0 40px rgba(0, 242, 254, 0.15);
        }
        .brand-title {
            background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }
        .form-control-custom {
            background: rgba(9, 13, 22, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.12);
            color: #f8fafc;
            border-radius: 12px;
            padding: 12px 16px;
        }
        .form-control-custom:focus {
            background: rgba(9, 13, 22, 0.9);
            border-color: #00f2fe;
            box-shadow: 0 0 15px rgba(0, 242, 254, 0.3);
            color: white;
        }
        .btn-glow {
            background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
            color: #070a12;
            font-weight: 700;
            border: none;
            border-radius: 12px;
            padding: 14px;
            transition: all 0.25s;
        }
        .btn-glow:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(0, 242, 254, 0.4);
            color: #070a12;
        }
        .dev-badge { color: #38bdf8; font-weight: 600; }
    </style>
</head>
<body>
    <div class="auth-card text-center">
        <div class="fs-1 mb-2">🤟</div>
        <h2 class="brand-title mb-1">Create Account</h2>
        <p class="text-secondary small mb-4">Join Sign AI Pro Translation Studio</p>

        {% with messages = get_flashed_messages(with_categories=true) %}
          {% if messages %}
            {% for category, message in messages %}
              <div class="alert alert-{{ 'danger' if category == 'error' else 'success' }} py-2 small mb-3 border-0 rounded-3">
                {{ message }}
              </div>
            {% endfor %}
          {% endif %}
        {% endwith %}

        <form action="/register" method="POST">
            <div class="mb-3 text-start">
                <label class="form-label text-secondary small fw-semibold">Username</label>
                <input type="text" name="username" class="form-control form-control-custom" placeholder="Choose username" required>
            </div>
            <div class="mb-3 text-start">
                <label class="form-label text-secondary small fw-semibold">Email Address</label>
                <input type="email" name="email" class="form-control form-control-custom" placeholder="name@example.com" required>
            </div>
            <div class="mb-4 text-start">
                <label class="form-label text-secondary small fw-semibold">Password</label>
                <input type="password" name="password" class="form-control form-control-custom" placeholder="••••••••" required>
            </div>
            <button type="submit" class="btn btn-glow w-100 mb-3">Create Free Account</button>
        </form>

        <p class="text-secondary small mb-3">Already registered? <a href="/login" class="text-info text-decoration-none fw-semibold">Sign In</a></p>

        <div class="pt-3 border-top border-secondary border-opacity-25 text-center">
            <span class="dev-badge small"><i class="bi bi-code-slash me-1"></i>Developed by DEEPAK SETHI</span>
        </div>
    </div>
</body>
</html>
"""

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sign AI Pro - Studio Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css">
    
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #070a12 0%, #0d1527 50%, #050810 100%);
            --card-bg: rgba(19, 28, 49, 0.75);
            --card-border: rgba(255, 255, 255, 0.08);
            --neon-blue: #00f2fe;
            --neon-purple: #4facfe;
            --neon-green: #10b981;
        }

        body {
            background: var(--bg-gradient);
            color: #f1f5f9;
            font-family: 'Inter', sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .glass-card {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }

        .header-title {
            background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }

        .status-dot {
            width: 10px;
            height: 10px;
            background-color: var(--neon-green);
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 10px var(--neon-green);
            animation: pulse-green 2s infinite;
        }

        @keyframes pulse-green {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }

        .video-container {
            position: relative;
            border-radius: 16px;
            overflow: hidden;
            border: 2px solid rgba(0, 242, 254, 0.3);
            box-shadow: 0 0 30px rgba(0, 242, 254, 0.15);
            background: #000;
        }

        .video-feed {
            width: 100%;
            height: auto;
            max-height: 440px;
            object-fit: cover;
            display: block;
        }

        .symbol-hero {
            font-family: 'JetBrains Mono', monospace;
            font-size: 4.8rem;
            font-weight: 800;
            background: linear-gradient(180deg, #ffffff 0%, #00f2fe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 30px rgba(0, 242, 254, 0.5);
            line-height: 1;
        }

        .sentence-display {
            background: rgba(9, 13, 22, 0.85);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 16px 20px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.5rem;
            color: #38bdf8;
            min-height: 80px;
            word-break: break-all;
            display: flex;
            align-items: center;
        }

        .btn-action {
            border: none;
            border-radius: 12px;
            padding: 12px 18px;
            font-weight: 600;
            font-size: 0.92rem;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }

        .btn-speak {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
        }

        .btn-speak:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(16, 185, 129, 0.5);
            color: white;
        }

        .btn-save {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        }

        .btn-save:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(59, 130, 246, 0.5);
            color: white;
        }

        .btn-control {
            background: rgba(255, 255, 255, 0.05);
            color: #cbd5e1;
            border: 1px solid var(--card-border);
        }

        .btn-control:hover {
            background: rgba(255, 255, 255, 0.12);
            color: white;
            transform: translateY(-2px);
        }

        .sugg-chip {
            background: rgba(0, 242, 254, 0.08);
            border: 1px solid rgba(0, 242, 254, 0.3);
            color: #00f2fe;
            padding: 6px 14px;
            border-radius: 30px;
            font-weight: 600;
            font-size: 0.88rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .sugg-chip:hover {
            background: #00f2fe;
            color: #090d16;
            transform: translateY(-2px);
        }

        .user-pill {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid var(--card-border);
            border-radius: 30px;
            padding: 6px 16px;
        }

        .history-item {
            background: rgba(9, 13, 22, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            padding: 12px 16px;
            margin-bottom: 10px;
            transition: all 0.2s;
        }
        .history-item:hover {
            border-color: rgba(0, 242, 254, 0.3);
            background: rgba(9, 13, 22, 0.9);
        }

        .toast-notify {
            position: fixed;
            bottom: 24px;
            right: 24px;
            background: #10b981;
            color: white;
            padding: 12px 24px;
            border-radius: 12px;
            font-weight: 600;
            display: none;
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
            z-index: 9999;
        }

        .dev-footer {
            color: #38bdf8;
            font-weight: 600;
            letter-spacing: 0.5px;
        }
    </style>
</head>
<body class="p-3 p-md-4">
    <div class="container-fluid max-width-1400">
        <!-- Top Studio Navbar -->
        <header class="d-flex justify-content-between align-items-center mb-4 pb-3 border-bottom border-secondary border-opacity-25">
            <div class="d-flex align-items-center gap-3">
                <div class="fs-2">🤟</div>
                <div>
                    <h2 class="header-title mb-0">Sign AI Pro Studio</h2>
                    <p class="text-secondary small mb-0">Realtime Deep Learning ASL Translation Platform</p>
                </div>
            </div>
            
            <div class="d-flex align-items-center gap-3">
                <div class="user-pill d-flex align-items-center gap-2">
                    <i class="bi bi-person-circle text-info fs-5"></i>
                    <span class="fw-semibold text-light small">{{ username }}</span>
                </div>
                <a href="/logout" class="btn btn-outline-danger btn-sm rounded-pill px-3 fw-semibold">
                    <i class="bi bi-box-arrow-right me-1"></i>Logout
                </a>
            </div>
        </header>

        <!-- Main Workspace Row -->
        <div class="row g-4 mb-4">
            <!-- Left Column: Video Feed -->
            <div class="col-lg-7">
                <div class="glass-card p-3 h-100 d-flex flex-column justify-content-between">
                    <div>
                        <div class="d-flex justify-content-between align-items-center mb-3 px-2">
                            <span class="small fw-semibold text-secondary text-uppercase tracking-wider">
                                <i class="bi bi-camera-video me-2"></i>Live Vision Stream
                            </span>
                            <div class="d-flex align-items-center gap-2">
                                <span class="status-dot"></span>
                                <span class="small text-light">Webcam Active</span>
                            </div>
                        </div>

                        <div class="video-container">
                            <img src="/video_feed" class="video-feed" alt="Realtime ASL Stream">
                        </div>
                    </div>

                    <div class="mt-3 pt-3 border-top border-secondary border-opacity-25 text-center">
                        <span class="dev-footer small"><i class="bi bi-code-slash me-1"></i>Developed by DEEPAK SETHI</span>
                    </div>
                </div>
            </div>

            <!-- Right Column: Recognition, Controls & History -->
            <div class="col-lg-5">
                <div class="glass-card p-4">
                    <!-- Detected Character Box -->
                    <div class="text-center p-3 mb-3 rounded-4" style="background: rgba(0, 242, 254, 0.03); border: 1px solid rgba(0, 242, 254, 0.15);">
                        <span class="small text-uppercase tracking-wider fw-semibold text-secondary">Active Sign Symbol</span>
                        <div id="character" class="symbol-hero mt-2 mb-1">-</div>
                    </div>

                    <!-- Sentence Builder -->
                    <div class="mb-3">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <label class="small text-uppercase tracking-wider fw-semibold text-secondary">Constructed Sentence</label>
                            <button class="btn btn-sm btn-control py-1 px-3" onclick="copySentence()">
                                <i class="bi bi-clipboard me-1"></i>Copy
                            </button>
                        </div>
                        <div id="sentence" class="sentence-display"></div>
                    </div>

                    <!-- Quick Editing Actions -->
                    <div class="d-flex gap-2 mb-3">
                        <button class="btn btn-control flex-fill btn-action" onclick="sendControl('space')">
                            <i class="bi bi-space me-1"></i>Space
                        </button>
                        <button class="btn btn-control flex-fill btn-action" onclick="sendControl('backspace')">
                            <i class="bi bi-backspace me-1"></i>Backspace
                        </button>
                        <button class="btn btn-outline-danger flex-fill btn-action" onclick="clearSentence()">
                            <i class="bi bi-trash3 me-1"></i>Clear
                        </button>
                    </div>

                    <!-- Auto Suggestions -->
                    <div class="mb-3">
                        <label class="small text-uppercase tracking-wider fw-semibold text-secondary d-block mb-2">
                            <i class="bi bi-lightbulb me-1"></i>Smart Auto-Suggestions
                        </label>
                        <div id="suggestions" class="d-flex flex-wrap gap-2">
                            <span class="text-secondary small fst-italic">Type gestures to see suggestions...</span>
                        </div>
                    </div>

                    <!-- Primary Actions: Speak & Save -->
                    <div class="d-flex gap-2 mb-4 pt-2">
                        <button class="btn btn-speak flex-fill btn-action py-3" onclick="speakSentence()">
                            <i class="bi bi-volume-up-fill fs-5"></i>Speak
                        </button>
                        <button class="btn btn-save flex-fill btn-action py-3" onclick="saveToHistory()">
                            <i class="bi bi-bookmark-check-fill fs-5"></i>Save Sentence
                        </button>
                    </div>

                    <!-- Saved History Section -->
                    <div class="pt-3 border-top border-secondary border-opacity-25">
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <span class="small fw-semibold text-uppercase tracking-wider text-secondary">
                                <i class="bi bi-clock-history me-1"></i>Saved History
                            </span>
                            <button class="btn btn-sm btn-control py-1 px-3" onclick="exportHistory()">
                                <i class="bi bi-download me-1"></i>Export TXT
                            </button>
                        </div>
                        
                        <div id="history-list" style="max-height: 160px; overflow-y: auto;">
                            <!-- History items loaded via JS -->
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Notification Toast -->
    <div id="toast" class="toast-notify">Action completed!</div>

    <script>
        function updateData() {
            fetch('/data')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('character').innerText = data.character || '-';
                    document.getElementById('sentence').innerText = data.sentence || '';
                    
                    const suggDiv = document.getElementById('suggestions');
                    if (data.suggestions && data.suggestions.length > 0) {
                        suggDiv.innerHTML = '';
                        data.suggestions.forEach(s => {
                            const chip = document.createElement('div');
                            chip.className = 'sugg-chip';
                            chip.innerText = s;
                            chip.onclick = () => applySuggestion(s);
                            suggDiv.appendChild(chip);
                        });
                    } else {
                        suggDiv.innerHTML = '<span class="text-secondary small fst-italic">Type gestures to see suggestions...</span>';
                    }
                });
        }

        function loadHistory() {
            fetch('/history/get')
                .then(r => r.json())
                .then(data => {
                    const listDiv = document.getElementById('history-list');
                    if (data.history && data.history.length > 0) {
                        listDiv.innerHTML = '';
                        data.history.forEach(item => {
                            const div = document.createElement('div');
                            div.className = 'history-item d-flex justify-content-between align-items-center';
                            div.innerHTML = `
                                <div>
                                    <div class="fw-semibold text-info mb-1">${item.sentence}</div>
                                    <div class="text-secondary" style="font-size: 0.75rem;">${item.timestamp}</div>
                                </div>
                                <button class="btn btn-sm text-danger border-0 p-1" onclick="deleteHistory(${item.id})">
                                    <i class="bi bi-x-lg"></i>
                                </button>
                            `;
                            listDiv.appendChild(div);
                        });
                    } else {
                        listDiv.innerHTML = '<div class="text-secondary small fst-italic text-center py-2">No saved translations yet.</div>';
                    }
                });
        }

        function saveToHistory() {
            const text = document.getElementById('sentence').innerText;
            if (text.trim() !== '') {
                fetch('/history/save', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({sentence: text})
                }).then(() => {
                    showToast('Saved to history!');
                    loadHistory();
                });
            }
        }

        function deleteHistory(id) {
            fetch('/history/delete', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({id: id})
            }).then(() => loadHistory());
        }

        function exportHistory() {
            window.location.href = '/history/export';
        }

        function speakSentence() {
            const text = document.getElementById('sentence').innerText;
            if (text.trim() !== '' && 'speechSynthesis' in window) {
                const utterance = new SpeechSynthesisUtterance(text);
                window.speechSynthesis.speak(utterance);
            }
        }

        function clearSentence() {
            fetch('/clear', {method: 'POST'}).then(() => updateData());
        }

        function sendControl(action) {
            fetch('/control', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action: action})
            }).then(() => updateData());
        }

        function applySuggestion(word) {
            fetch('/suggestion', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({word: word})
            }).then(() => updateData());
        }

        function copySentence() {
            const text = document.getElementById('sentence').innerText;
            if (text) {
                navigator.clipboard.writeText(text).then(() => showToast('Copied to clipboard!'));
            }
        }

        function showToast(msg) {
            const toast = document.getElementById('toast');
            toast.innerText = msg;
            toast.style.display = 'block';
            setTimeout(() => toast.style.display = 'none', 2000);
        }

        setInterval(updateData, 350);
        loadHistory();
    </script>
</body>
</html>
"""

# --- ROUTES & AUTHENTICATION ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        user_captcha = request.form.get('captcha', '').strip().upper()
        expected_captcha = session.get('captcha_text', '')

        # Verify CAPTCHA first
        if not user_captcha or user_captcha != expected_captcha:
            flash("Invalid Security CAPTCHA code. Please try again.", "error")
            return render_template_string(LOGIN_HTML)

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT id, password FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid username or password", "error")
    
    return render_template_string(LOGIN_HTML)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        hashed_pw = generate_password_hash(password)

        try:
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (username, email, hashed_pw))
            conn.commit()
            conn.close()
            flash("Account created successfully! Please sign in.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username or email already exists.", "error")
            
    return render_template_string(REGISTER_HTML)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def dashboard():
    return render_template_string(DASHBOARD_HTML, username=session.get('username'))

@app.route('/video_feed')
@login_required
def video_feed():
    def generate_frames():
        camera = cv2.VideoCapture(0)
        white_template = cv2.imread("white.jpg")
        if white_template is None:
            white_template = np.ones((400, 400, 3), dtype=np.uint8) * 255

        while True:
            success, frame = camera.read()
            if not success:
                break
            
            frame = cv2.flip(frame, 1)
            hands = hd.findHands(frame, draw=False, flipType=True)
            
            if hands and hands[0]:
                try:
                    hand = hands[0]
                    map_info = hand[0] if isinstance(hand, list) else hand
                    x, y, w, h = map_info['bbox']
                    
                    h_img, w_img, _ = frame.shape
                    y1, y2 = max(0, y - offset), min(h_img, y + h + offset)
                    x1, x2 = max(0, x - offset), min(w_img, x + w + offset)
                    
                    cropped = frame[y1:y2, x1:x2]
                    
                    if cropped.size > 0:
                        handz = hd2.findHands(cropped, draw=False, flipType=True)
                        if handz and handz[0]:
                            hand_inner = handz[0]
                            map_inner = hand_inner[0] if isinstance(hand_inner, list) else hand_inner
                            pts = map_inner['lmList']
                            predictor.pts = pts

                            white = white_template.copy()
                            os_x = ((400 - w) // 2) - 15
                            os_y = ((400 - h) // 2) - 15

                            lines_idx = [
                                (0, 1), (1, 2), (2, 3), (3, 4),
                                (5, 6), (6, 7), (7, 8),
                                (9, 10), (10, 11), (11, 12),
                                (13, 14), (14, 15), (15, 16),
                                (17, 18), (18, 19), (19, 20),
                                (5, 9), (9, 13), (13, 17), (0, 5), (0, 17)
                            ]
                            for p1, p2 in lines_idx:
                                cv2.line(white, (pts[p1][0] + os_x, pts[p1][1] + os_y), (pts[p2][0] + os_x, pts[p2][1] + os_y), (0, 255, 0), 3)

                            for i in range(21):
                                cv2.circle(white, (pts[i][0] + os_x, pts[i][1] + os_y), 2, (0, 0, 255), 1)

                            predictor.process_frame_prediction(white)
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (254, 242, 0), 2)
                            cv2.putText(frame, f"Sign: {predictor.current_symbol}", (x1, max(30, y1 - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 242, 254), 2)
                except Exception as e:
                    pass

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
@login_required
def get_data():
    return jsonify({
        'character': predictor.current_symbol,
        'sentence': predictor.sentence,
        'suggestions': predictor.suggestions
    })

@app.route('/clear', methods=['POST'])
@login_required
def clear_data():
    predictor.sentence = ""
    predictor.current_symbol = "-"
    predictor.suggestions = []
    return jsonify({'status': 'ok'})

@app.route('/control', methods=['POST'])
@login_required
def handle_control():
    data = request.json
    action = data.get('action', '')
    if action == 'space':
        predictor.sentence += " "
    elif action == 'backspace':
        predictor.sentence = predictor.sentence[:-1]
    return jsonify({'status': 'ok'})

@app.route('/suggestion', methods=['POST'])
@login_required
def apply_suggestion():
    data = request.json
    word = data.get('word', '')
    if word and predictor.sentence:
        idx_space = predictor.sentence.rfind(" ")
        if idx_space != -1:
            predictor.sentence = predictor.sentence[:idx_space + 1] + word.upper() + " "
        else:
            predictor.sentence = word.upper() + " "
    elif word:
        predictor.sentence = word.upper() + " "
    return jsonify({'status': 'ok'})

# History API Endpoints
@app.route('/history/save', methods=['POST'])
@login_required
def save_history():
    data = request.json
    sentence = data.get('sentence', '').strip()
    if sentence:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO history (user_id, sentence) VALUES (?, ?)", (session['user_id'], sentence))
        conn.commit()
        conn.close()
    return jsonify({'status': 'ok'})

@app.route('/history/get')
@login_required
def get_history():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, sentence, timestamp FROM history WHERE user_id = ? ORDER BY timestamp DESC LIMIT 10", (session['user_id'],))
    rows = cursor.fetchall()
    conn.close()

    history_data = [{'id': row[0], 'sentence': row[1], 'timestamp': row[2]} for row in rows]
    return jsonify({'history': history_data})

@app.route('/history/delete', methods=['POST'])
@login_required
def delete_history():
    data = request.json
    hist_id = data.get('id')
    if hist_id:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM history WHERE id = ? AND user_id = ?", (hist_id, session['user_id']))
        conn.commit()
        conn.close()
    return jsonify({'status': 'ok'})

@app.route('/history/export')
@login_required
def export_history():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT sentence, timestamp FROM history WHERE user_id = ? ORDER BY timestamp DESC", (session['user_id'],))
    rows = cursor.fetchall()
    conn.close()

    content = f"--- Sign AI Pro Translation History for {session.get('username')} ---\nDeveloped by DEEPAK SETHI\n\n"
    for r in rows:
        content += f"[{r[1]}] {r[0]}\n"

    buffer = io.BytesIO()
    buffer.write(content.encode('utf-8'))
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name=f"asl_history_{session.get('username')}.txt", mimetype='text/plain')

if __name__ == '__main__':
    print("Starting Sign AI Pro (Developed by DEEPAK SETHI) on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
