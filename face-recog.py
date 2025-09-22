from PIL import Image
# from keras.models import load_model # Not used in the original code
import numpy as np
from numpy import asarray, expand_dims

import pickle
import cv2
from keras_facenet import FaceNet

# --- KONFIGURASI ---
# Lewati beberapa frame untuk mengurangi beban CPU. 
# Nilai 5 berarti pengenalan wajah hanya dilakukan setiap 5 frame.
FRAME_SKIP = 7 
# Atur resolusi yang lebih rendah untuk pemrosesan
PROC_WIDTH = 640
PROC_HEIGHT = 480
# Atur threshold untuk mengenali wajah
RECOGNITION_THRESHOLD = 0.7

# --- Inisialisasi Model ---
print("Memuat model dan data...")
# Initialize HaarCascade for face detection
HaarCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))

# Initialize FaceNet model
MyFaceNet = FaceNet()

# Load the known faces database
with open("data.pkl", "rb") as myfile:
    database = pickle.load(myfile)
print("Model dan data berhasil dimuat.")

# --- Mulai Video Capture ---
# Coba ganti '3' dengan 0 atau 1 jika kamera tidak ditemukan
cap = cv2.VideoCapture(3) 
# Set resolusi kamera jika memungkinkan (opsional, karena kita akan resize manual)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, PROC_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PROC_HEIGHT)

# Variabel untuk menyimpan hasil terakhir & frame counter
frame_counter = 0
last_identity = ' '
last_box = (0, 0, 0, 0)

print("Memulai kamera...")
while(True):
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera. Keluar...")
        break

    # 1. MENGURANGI RESOLUSI
    # Resize frame untuk mempercepat deteksi. Ini adalah optimisasi terpenting.
    gbr_small = cv2.resize(frame, (PROC_WIDTH, PROC_HEIGHT))

    # 2. MENGURANGI RENDERING/PENGENALAN
    # Hanya jalankan deteksi & pengenalan setiap FRAME_SKIP frame
    if frame_counter % FRAME_SKIP == 0:
        # Deteksi wajah pada frame yang sudah dikecilkan
        wajah = HaarCascade.detectMultiScale(gbr_small, 1.1, 4)

        if len(wajah) > 0:
            x1, y1, width, height = wajah[0]
            x2, y2 = x1 + width, y1 + height
            
            # Konversi ke RGB untuk FaceNet
            gbr_rgb = cv2.cvtColor(gbr_small, cv2.COLOR_BGR2RGB)
            
            # Ekstrak wajah dan resize (lebih efisien tanpa PIL)
            face = gbr_rgb[y1:y2, x1:x2]
            face_resized = cv2.resize(face, (160, 160))

            # Dapatkan face embedding
            face_array = expand_dims(face_resized, axis=0)
            signature = MyFaceNet.embeddings(face_array)

            # Logika Pengenalan
            min_dist = 100
            identity = ' '
            for key, value in database.items():
                dist = np.linalg.norm(signature - value)
                if dist < min_dist:
                    min_dist = dist
                    identity = key
            
            if min_dist > RECOGNITION_THRESHOLD:
                identity = 'Tidak Dikenal'
            
            # Simpan hasil terakhir
            last_identity = f'{identity} ({min_dist:.2f})'
            last_box = (x1, y1, x2, y2)
        else:
            # Jika tidak ada wajah terdeteksi, reset hasil terakhir
            last_identity = ' '
            last_box = (0, 0, 0, 0)

    # Gambar kotak dan nama di setiap frame menggunakan hasil terakhir
    # Ini membuat tampilan video tetap halus dan tidak berkedip
    if last_identity != ' ':
        (x1, y1, x2, y2) = last_box
        cv2.putText(gbr_small, last_identity, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(gbr_small, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Tampilkan frame hasil (yang sudah dikecilkan)
    cv2.imshow('Face Recognition', gbr_small)
    
    frame_counter += 1
    k = cv2.waitKey(5) & 0xFF
    if k == 27: # Tekan ESC untuk keluar
        break

# Lepaskan semua resource
print("Menutup aplikasi.")
cv2.destroyAllWindows()
cap.release()