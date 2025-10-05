# =====================================================================
# --- Import Library ---
# =====================================================================
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
import platform # <-- Tambahkan ini untuk cek OS

# --- MODIFIKASI: Import GPIO secara kondisional ---
# Cek apakah sistem operasinya Linux (seperti di Orange Pi)
IS_ORANGE_PI = platform.system() == "Linux"
if IS_ORANGE_PI:
    try:
        import OPi.GPIO as GPIO
        PI_TYPE = "Orange Pi"
    except (ImportError, RuntimeError):
        print("PERINGATAN: Gagal import OPi.GPIO. Fitur buzzer tidak akan aktif.")
        IS_ORANGE_PI = False
else:
    print("INFO: Sistem non-Linux terdeteksi (Windows/Mac). Fitur GPIO (buzzer) akan dinonaktifkan.")

# =====================================================================
# --- KONFIGURASI (WAJIB DIISI SESUAI DATA ANDA) ---
# =====================================================================

# --- Konfigurasi MySQL ---
DB_HOST = "10.122.15.45"
DB_USER = "opiuser"
DB_PASS = "passwordku"  # Isi password database Anda
DB_NAME = "face_recognition_db"

# --- Konfigurasi Telegram ---
TELEGRAM_TOKEN = "8178565679:AAH7wcfG20hyA1LSR4-yKquCc305nCqHuBc"  # Ganti dengan Token Bot Anda
TELEGRAM_CHAT_ID = "1370373890"  # Ganti dengan Chat ID Anda

# --- Konfigurasi GPIO untuk Buzzer ---
BUZZER_PIN = "PC7"  # Sesuaikan pin GPIO yang Anda gunakan

# =====================================================================
# --- PENGATURAN SCRIPT ---
# =====================================================================
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
RECOGNITION_THRESHOLD = 0.7
NOTIFICATION_COOLDOWN = 10
last_action_times = {}

# =====================================================================
# --- FUNGSI-FUNGSI BANTUAN ---
# =====================================================================
def activate_buzzer(duration=1.5):
    """Membunyikan buzzer hanya jika di Orange Pi."""
    if not IS_ORANGE_PI: return # <-- Jangan lakukan apa-apa jika bukan di Orange Pi
    try:
        print("ALERT: Wajah tidak dikenal! Membunyikan buzzer...")
        GPIO.output(BUZZER_PIN, GPIO.HIGH)
        time.sleep(duration)
        GPIO.output(BUZZER_PIN, GPIO.LOW)
    except Exception as e:
        print(f"Gagal membunyikan buzzer: {e}")

def log_to_database(timestamp, status, name, frame):
    """Menyimpan data deteksi ke database MySQL."""
    if db_connection is None: return
    try:
        _, img_encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        img_bytes = img_encoded.tobytes()
        sql = "INSERT INTO deteksi_wajah (timestamp, status, nama, gambar) VALUES (%s, %s, %s, %s)"
        val = (timestamp, status, name, img_bytes)
        db_cursor.execute(sql, val)
        db_connection.commit()
        print(f"Data berhasil disimpan ke DB: Status='{status}', Nama='{name}'")
    except mysql.connector.Error as err:
        print(f"Gagal menyimpan ke DB: {err}")
        db_connection.rollback()

def send_telegram_notification(caption, frame_with_box):
    """Mengirim notifikasi foto ke Telegram."""
    if bot is None: return
    try:
        _, img_encoded = cv2.imencode('.jpg', frame_with_box, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        image_stream = io.BytesIO(img_encoded.tobytes())
        image_stream.name = 'detection.jpg'
        image_stream.seek(0)
        bot.send_photo(TELEGRAM_CHAT_ID, image_stream, caption=caption, timeout=30)
        print("Notifikasi Telegram terkirim.")
    except Exception as e:
        print(f"Gagal mengirim notifikasi Telegram: {e}")

# =====================================================================
# --- INISIALISASI ---
# =====================================================================
print("Mempersiapkan sistem...")

# --- MODIFIKASI: Inisialisasi GPIO hanya jika di Orange Pi ---
if IS_ORANGE_PI:
    try:
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.SUNXI)
        GPIO.setup(BUZZER_PIN, GPIO.OUT)
        GPIO.output(BUZZER_PIN, GPIO.LOW)
        print(f"GPIO pin {BUZZER_PIN} untuk buzzer berhasil diinisialisasi pada {PI_TYPE}.")
    except Exception as e:
        print(f"Gagal menginisialisasi GPIO. Fitur buzzer tidak akan aktif. Error: {e}")
        IS_ORANGE_PI = False # <-- Matikan flag jika setup gagal

# Sisa inisialisasi lainnya tetap sama
# ... (inisialisasi Telegram, Database, Model, Kamera, dll.) ...
# (Kode di bawah ini sengaja diringkas karena tidak ada perubahan)

# --- Inisialisasi Bot Telegram ---
bot = None
try:
    bot = telebot.TeleBot(TELEGRAM_TOKEN); bot.get_me(); print("Bot Telegram berhasil diinisialisasi.")
except Exception as e:
    print(f"Gagal menginisialisasi Bot Telegram: {e}. Notifikasi Telegram dinonaktifkan.")

# --- Koneksi ke Database MySQL ---
db_connection = None
try:
    db_connection = mysql.connector.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME, ssl_disabled=True)
    db_cursor = db_connection.cursor(buffered=True); print("Koneksi ke database MySQL berhasil.")
except mysql.connector.Error as err:
    print(f"Error koneksi database: {err}. Logging ke database dinonaktifkan.")

# --- Inisialisasi Model ---
print("Memuat model pengenalan wajah..."); MyFaceNet = FaceNet(); HaarCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Muat Dataset Wajah ---
try:
    with open("data.pkl", "rb") as f: database = pickle.load(f)
    all_embeddings = []; all_names = []
    for name, embeddings_list in database.items():
        for embedding in embeddings_list: all_embeddings.append(embedding); all_names.append(name)
    all_embeddings = np.array(all_embeddings); print("Database wajah berhasil dimuat.")
except FileNotFoundError: print("Error: File 'data.pkl' tidak ditemukan."); exit()

# --- Inisialisasi Kamera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened(): print("Error: Tidak bisa membuka kamera."); exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

print("\nSistem Siap. Memulai deteksi wajah...")
# =====================================================================
# --- LOOP UTAMA (Tidak ada perubahan di sini) ---
# =====================================================================
try:
    while True:
        ret, frame = cap.read()
        if not ret: time.sleep(1); continue
        
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = HaarCascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size == 0: continue
            
            embedding = MyFaceNet.embeddings(expand_dims(cv2.resize(face_roi, (160, 160)).astype('float32'), axis=0))[0]
            distances = np.linalg.norm(all_embeddings - embedding, axis=1)
            min_dist = np.min(distances)
            
            if min_dist < RECOGNITION_THRESHOLD:
                name = all_names[np.argmin(distances)]; status = "Dikenali"; color = (0, 255, 0)
            else:
                name = "Tidak Dikenal"; status = "Tidak Dikenali"; color = (0, 0, 255)

            current_time = time.time()
            if (current_time - last_action_times.get(name, 0)) > NOTIFICATION_COOLDOWN:
                timestamp = datetime.datetime.now()
                if status == "Tidak Dikenali":
                    caption_text = f"⚠️ PERINGATAN: Wajah Asing Terdeteksi!\n🕒 Waktu: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                    activate_buzzer()
                else:
                    caption_text = f"👤 Wajah Dikenali: {name}\n🕒 Waktu: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                
                frame_with_box = frame.copy()
                cv2.rectangle(frame_with_box, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame_with_box, f"{name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                
                send_telegram_notification(caption_text, frame_with_box)
                log_to_database(timestamp, status, name, frame_with_box)
                last_action_times[name] = current_time

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{name} ({min_dist:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        cv2.imshow('Sistem Pengenalan Wajah - Tekan Q untuk Keluar', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    print("\nMenutup program...")
    cap.release(); cv2.destroyAllWindows()
    if db_connection and db_connection.is_connected(): db_connection.close(); print("Koneksi database ditutup.")
    if IS_ORANGE_PI: GPIO.cleanup(); print("GPIO dibersihkan.")