# --- Import Library ---
import cv2
import numpy as np
from numpy import expand_dims
import pickle
from keras_facenet import FaceNet
import datetime
import mysql.connector
import telebot
import io
import time
# DIUBAH: Menggunakan library untuk Orange Pi
import OPi.GPIO as GPIO 

# ... (sisa kode lainnya sama persis seperti sebelumnya) ...

# =====================================================================
# --- KONFIGURASI TAMBAHAN (WAJIB DIISI) ---
# =====================================================================

# --- Konfigurasi MySQL ---
DB_HOST = "localhost"
DB_USER = "root"
DB_PASS = ""
DB_NAME = "face_recognition_db"

# --- Konfigurasi Telegram ---
TELEGRAM_TOKEN = "8178565679:AAH7wcfG20hyA1LSR4-yKquCc305nCqHuBc"
TELEGRAM_CHAT_ID = "1370373890"

# --- Konfigurasi GPIO untuk Buzzer ---
BUZZER_PIN = 17 # Pastikan pin ini sesuai dengan koneksi di Orange Pi Anda

# =====================================================================
# --- KONFIGURASI SCRIPT (SESUAI PERMINTAAN) ---
# =====================================================================
FRAME_SKIP = 15
PROC_WIDTH = 320
PROC_HEIGHT = 240
RECOGNITION_THRESHOLD = 0.75

# ... (dan seterusnya, seluruh sisa kode tidak perlu diubah) ...
# Konfigurasi Zona Trigger disesuaikan dengan resolusi baru
TRIGGER_ZONE_Y_START = int(PROC_HEIGHT * 0.70) # Mulai dari 70% bagian bawah layar
TRIGGER_ZONE = (0, TRIGGER_ZONE_Y_START, PROC_WIDTH, PROC_HEIGHT - TRIGGER_ZONE_Y_START)


# =====================================================================
# --- Inisialisasi Tambahan ---
# =====================================================================

# --- Inisialisasi GPIO ---
try:
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUZZER_PIN, GPIO.OUT)
    GPIO.output(BUZZER_PIN, GPIO.LOW) # Pastikan buzzer mati saat mulai
    print(f"GPIO pin {BUZZER_PIN} untuk buzzer berhasil diinisialisasi.")
except Exception as e:
    print(f"Gagal menginisialisasi GPIO. Fitur buzzer tidak akan aktif. Error: {e}")

# --- Inisialisasi Bot Telegram ---
try:
    bot = telebot.TeleBot(TELEGRAM_TOKEN)
    print("Bot Telegram berhasil diinisialisasi.")
except Exception as e:
    bot = None
    print(f"Gagal menginisialisasi Bot Telegram: {e}")

# --- Koneksi ke Database MySQL ---
try:
    db_connection = mysql.connector.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME)
    db_cursor = db_connection.cursor()
    print("Koneksi ke database MySQL berhasil.")
except mysql.connector.Error as err:
    db_connection = None
    print(f"Error koneksi database: {err}. Logging ke database dinonaktifkan.")

# =====================================================================
# --- FUNGSI HELPER ---
# =====================================================================
def activate_buzzer(duration=1):
    """Membunyikan buzzer selama durasi tertentu (dalam detik)."""
    try:
        print("ALERT: Wajah tidak dikenal! Membunyikan buzzer...")
        GPIO.output(BUZZER_PIN, GPIO.HIGH)
        time.sleep(duration)
        GPIO.output(BUZZER_PIN, GPIO.LOW)
    except Exception as e:
        print(f"Gagal membunyikan buzzer: {e}")

def log_to_database(frame, timestamp, status, name):
    """Menyimpan data deteksi ke database MySQL."""
    if db_connection is None: return
    try:
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()
        sql = "INSERT INTO deteksi_wajah (timestamp, status, nama, gambar) VALUES (%s, %s, %s, %s)"
        val = (timestamp, status, name, img_bytes)
        db_cursor.execute(sql, val)
        db_connection.commit()
        print(f"Data berhasil disimpan ke DB: {name} - {status}")
    except mysql.connector.Error as err:
        print(f"Gagal menyimpan ke DB: {err}")

def send_telegram_notification(frame_with_box, caption):
    """Mengirim notifikasi foto ke Telegram."""
    if bot is None: return
    try:
        _, img_encoded = cv2.imencode('.jpg', frame_with_box)
        image_stream = io.BytesIO(img_encoded.tobytes())
        image_stream.name = 'detection.jpg'; image_stream.seek(0)
        bot.send_photo(TELEGRAM_CHAT_ID, image_stream, caption=caption, timeout=60)
        print("Notifikasi Telegram terkirim.")
    except Exception as e:
        print(f"Gagal mengirim notifikasi Telegram: {e}")

# =====================================================================
# --- Inisialisasi Model ---
# =====================================================================
print("Memuat model dan data...")
FaceCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
BodyCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_fullbody.xml'))
MyFaceNet = FaceNet()
with open("data.pkl", "rb") as myfile:
    database = pickle.load(myfile)
print("Model dan data berhasil dimuat.")

# =====================================================================
# --- Mulai Video Capture ---
# =====================================================================
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, PROC_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PROC_HEIGHT)
frame_counter = 0; last_identity = ' '; last_box = (0, 0, 0, 0); is_triggered = False

print("Memulai kamera...")
try:
    while(True):
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera. Keluar..."); break

        gbr_small = cv2.resize(frame, (PROC_WIDTH, PROC_HEIGHT))
        (zx, zy, zw, zh) = TRIGGER_ZONE
        cv2.rectangle(gbr_small, (zx, zy), (zx + zw, zy + zh), (255, 255, 0), 1)
        cv2.putText(gbr_small, "Zona Trigger", (zx + 5, zy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        if frame_counter % FRAME_SKIP == 0:
            bodies = BodyCascade.detectMultiScale(gbr_small, 1.2, 3)
            is_triggered = any(by + bh > TRIGGER_ZONE_Y_START for (_, by, _, bh) in bodies)
            
            if is_triggered:
                wajah = FaceCascade.detectMultiScale(gbr_small, 1.2, 5)
                if len(wajah) > 0:
                    x1, y1, width, height = wajah[0]; x2, y2 = x1 + width, y1 + height
                    gbr_rgb = cv2.cvtColor(gbr_small, cv2.COLOR_BGR2RGB)
                    face = gbr_rgb[y1:y2, x1:x2]
                    
                    if face.size > 0:
                        face_resized = cv2.resize(face, (160, 160))
                        signature = MyFaceNet.embeddings(expand_dims(face_resized, axis=0))
                        min_dist = 100; identity = ' '
                        for key, value in database.items():
                            dist = np.linalg.norm(signature - value)
                            if dist < min_dist:
                                min_dist, identity = dist, key
                        
                        timestamp = datetime.datetime.now()
                        frame_to_notify = gbr_small.copy()
                        
                        if min_dist > RECOGNITION_THRESHOLD:
                            status, identity_final = "Tidak Dikenali", "Tidak Dikenali"
                            caption_text = f"Wajah tidak dikenali terdeteksi!\nTimestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                            cv2.rectangle(frame_to_notify, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            activate_buzzer(duration=1) # Bunyikan buzzer
                        else:
                            status, identity_final = "Dikenali", identity
                            caption_text = f"Wajah dikenali: {identity_final}\nTimestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                            cv2.rectangle(frame_to_notify, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        send_telegram_notification(frame_to_notify, caption_text)
                        log_to_database(frame_to_notify, timestamp, status, identity_final)
                        last_identity, last_box = f'{identity_final} ({min_dist:.2f})', (x1, y1, x2, y2)
                else:
                    last_identity, last_box = ' ', (0,0,0,0) # Wajah tidak ada, reset
            else:
                last_identity, last_box = ' ', (0,0,0,0) # Trigger tidak aktif, reset
        
        if last_identity != ' ':
            (x1, y1, x2, y2) = last_box
            cv2.putText(gbr_small, last_identity, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(gbr_small, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('Face Recognition on Pi', gbr_small)
        k = cv2.waitKey(5) & 0xFF
        if k == 27: break # Tekan ESC untuk keluar
        frame_counter += 1

finally:
    # --- Lepaskan semua resource ---
    print("\nMenutup aplikasi.")
    cv2.destroyAllWindows()
    cap.release()
    if db_connection and db_connection.is_connected():
        db_cursor.close(); db_connection.close()
        print("Koneksi database ditutup.")
    GPIO.cleanup()
    print("GPIO dibersihkan.")