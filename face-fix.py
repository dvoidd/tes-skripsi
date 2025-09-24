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
import OPi.GPIO as GPIO
PI_TYPE = "Orange Pi"

# =====================================================================
# --- KONFIGURASI (WAJIB DIISI SESUAI DATA ANDA) ---
# =====================================================================

# --- Konfigurasi MySQL ---
DB_HOST = "localhost"
DB_USER = "root"
DB_PASS = ""  # Isi password database Anda, kosongkan jika tidak ada
DB_NAME = "face_recognition_db"

# --- Konfigurasi Telegram ---
TELEGRAM_TOKEN = "8178565679:AAH7wcfG20hyA1LSR4-yKquCc305nCqHuBc"      # Ganti dengan Token Bot Anda
TELEGRAM_CHAT_ID = "1370373890"    # Ganti dengan Chat ID Anda

# --- Konfigurasi GPIO untuk Buzzer ---
BUZZER_PIN = 17 # Sesuaikan pin GPIO yang Anda gunakan

# =====================================================================
# --- PENGATURAN SCRIPT ---
# =====================================================================
# Resolusi kamera diatur ke 320x240 dan FPS ke 15
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
FRAME_RATE = 15

# Ambang batas, makin kecil makin ketat pengenalannya
RECOGNITION_THRESHOLD = 0.75

# Jeda waktu (dalam detik) untuk tidak mengirim notif ke orang yang sama
NOTIFICATION_COOLDOWN = 60
last_notification_times = {}

# =====================================================================
# --- INISIALISASI ---
# =====================================================================

# --- Inisialisasi GPIO ---
def setup_gpio():
    if GPIO:
        try:
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(BUZZER_PIN, GPIO.OUT)
            GPIO.output(BUZZER_PIN, GPIO.LOW) # Pastikan buzzer mati saat mulai
            print(f"GPIO pin {BUZZER_PIN} untuk buzzer berhasil diinisialisasi pada {PI_TYPE}.")
            return True
        except Exception as e:
            print(f"Gagal menginisialisasi GPIO. Fitur buzzer tidak akan aktif. Error: {e}")
            return False
    return False

is_gpio_ready = setup_gpio()

# --- Inisialisasi Bot Telegram ---
try:
    bot = telebot.TeleBot(TELEGRAM_TOKEN)
    bot.get_me() # Cek koneksi ke API Telegram
    print("Bot Telegram berhasil diinisialisasi.")
except Exception as e:
    bot = None
    print(f"Gagal menginisialisasi Bot Telegram: {e}. Notifikasi Telegram dinonaktifkan.")

# --- Koneksi ke Database MySQL ---
try:
    db_connection = mysql.connector.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME)
    db_cursor = db_connection.cursor(buffered=True)
    print("Koneksi ke database MySQL berhasil.")
except mysql.connector.Error as err:
    db_connection = None
    print(f"Error koneksi database: {err}. Logging ke database dinonaktifkan.")

# =====================================================================
# --- FUNGSI-FUNGSI BANTUAN ---
# =====================================================================
def activate_buzzer(duration=0.5):
    """Membunyikan buzzer selama durasi tertentu (dalam detik)."""
    if not is_gpio_ready: return
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
        # Konversi gambar ke format biner untuk disimpan di BLOB
        _, img_encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        img_bytes = img_encoded.tobytes()
        
        sql = "INSERT INTO deteksi_wajah (timestamp, status, nama, gambar) VALUES (%s, %s, %s, %s)"
        val = (timestamp, status, name, img_bytes)
        
        db_cursor.execute(sql, val)
        db_connection.commit()
        print(f"Data berhasil disimpan ke DB: Status='{status}', Nama='{name}'")
    except mysql.connector.Error as err:
        print(f"Gagal menyimpan ke DB: {err}")
        db_connection.rollback() # Batalkan transaksi jika gagal

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
# --- Inisialisasi Model ---
# =====================================================================
print("Memuat model dan database wajah...")
try:
    FaceCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
    MyFaceNet = FaceNet()
    with open("data.pkl", "rb") as myfile:
        database = pickle.load(myfile)
    print("Model dan data berhasil dimuat.")
except FileNotFoundError:
    print("ERROR: File 'data.pkl' tidak ditemukan! Pastikan Anda sudah menjalankan script training.")
    exit()
except Exception as e:
    print(f"ERROR saat memuat model: {e}")
    exit()

# =====================================================================
# --- Mulai Video Capture ---
# =====================================================================
cap = cv2.VideoCapture(0)
# Terapkan konfigurasi resolusi dan FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)

print("Memulai kamera...")
try:
    while(True):
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera. Mencoba lagi...")
            time.sleep(1)
            continue
        
        # Deteksi wajah
        faces = FaceCascade.detectMultiScale(frame, 1.2, 5)
        
        # Proses hanya jika ada wajah terdeteksi
        if len(faces) > 0:
            # Ambil wajah terbesar (biasanya yang paling dekat)
            x1, y1, width, height = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            x2, y2 = x1 + width, y1 + height
            
            # Ekstrak ROI (Region of Interest) dari wajah
            face_roi_rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            
            # Pastikan ROI tidak kosong sebelum di-resize
            if face_roi_rgb.size == 0:
                continue

            face_resized = cv2.resize(face_roi_rgb, (160, 160))
            signature = MyFaceNet.embeddings(expand_dims(face_resized, axis=0))
            
            # Cari kemiripan di database
            min_dist = float('inf')
            identity = "Tidak Dikenali"
            
            for key, value in database.items():
                dist = np.linalg.norm(signature - value)
                if dist < min_dist:
                    min_dist, identity_candidate = dist, key
            
            if min_dist <= RECOGNITION_THRESHOLD:
                identity = identity_candidate
                status = "Dikenali"
                box_color = (0, 255, 0) # Hijau
            else:
                status = "Tidak Dikenali"
                box_color = (0, 0, 255) # Merah

            # Cek cooldown sebelum mengirim notifikasi dan logging
            current_time = time.time()
            last_notified = last_notification_times.get(identity, 0)

            if (current_time - last_notified) > NOTIFICATION_COOLDOWN:
                timestamp = datetime.datetime.now()
                caption_text = (f"👤 Wajah Dikenali: {identity}\n"
                                f"🕒 Waktu: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                
                if status == "Tidak Dikenali":
                    caption_text = (f"⚠️ Wajah Tidak Dikenali Terdeteksi!\n"
                                    f"🕒 Waktu: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                    activate_buzzer()

                # Buat frame dengan kotak untuk notifikasi
                frame_for_notif = frame.copy()
                cv2.rectangle(frame_for_notif, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame_for_notif, f"{identity} [{status}]", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                
                # Kirim notifikasi dan simpan ke DB
                send_telegram_notification(caption_text, frame_for_notif)
                log_to_database(timestamp, status, identity, frame)
                
                # Perbarui waktu notifikasi terakhir
                last_notification_times[identity] = current_time

            # Tampilkan di layar (opsional, bisa dinonaktifkan jika headless)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, f"{identity}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # Tampilkan frame hasil (nonaktifkan jika tidak perlu GUI)
        cv2.imshow('Sistem Pengenalan Wajah', frame)

        # Tombol 'ESC' untuk keluar
        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    print("\nProgram dihentikan oleh pengguna.")
finally:
    # --- Lepaskan semua resource ---
    print("Menutup aplikasi dan membersihkan resource...")
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    if db_connection and db_connection.is_connected():
        db_cursor.close()
        db_connection.close()
        print("Koneksi database ditutup.")
    if is_gpio_ready:
        GPIO.cleanup()
        print("GPIO dibersihkan.")
coba hapus kode gpio nya, tidak usah menggunakan buzzer