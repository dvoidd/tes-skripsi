# --- Import Library ---
import cv2
import numpy as np
from numpy import expand_dims
import pickle
from keras_facenet import FaceNet
import time

print("Mempersiapkan model dan data...")

# --- Inisialisasi Model ---
# 1. Inisialisasi FaceNet untuk membuat embedding wajah
MyFaceNet = FaceNet()

# 2. Muat Haar Cascade classifier untuk mendeteksi lokasi wajah
# Pastikan file haarcascade berada di path yang benar
try:
    HaarCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception as e:
    print(f"Error memuat file Haarcascade: {e}")
    print("Pastikan OpenCV terinstall dengan benar.")
    exit()

# 3. Muat dataset wajah yang sudah disimpan (file .pkl)
try:
    with open("data.pkl", "rb") as f:
        database = pickle.load(f)
except FileNotFoundError:
    print("Error: File 'data.pkl' tidak ditemukan.")
    print("Pastikan Anda sudah membuat dan menyimpan file dataset embedding wajah.")
    exit()
except Exception as e:
    print(f"Error saat memuat file pickle: {e}")
    exit()

# --- Memproses Dataset ---
# Pisahkan embedding dan nama dari database untuk perbandingan
all_embeddings = []
all_names = []

for name, embeddings_list in database.items():
    for embedding in embeddings_list:
        all_embeddings.append(embedding)
        all_names.append(name)

# Konversi ke numpy array agar proses kalkulasi lebih cepat
all_embeddings = np.array(all_embeddings)

print("Model dan data berhasil dimuat. Memulai kamera...")

# --- Inisialisasi Kamera ---
# Ganti angka '0' jika Anda menggunakan kamera USB eksternal (misal: 1, 2, dst)
cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("Error: Tidak bisa membuka kamera.")
    exit()

# --- Loop Utama untuk Deteksi Real-time ---
while True:
    # Baca frame dari kamera
    ret, frame = cap.read()
    if not ret:
        print("Gagal mengambil frame. Keluar...")
        break

    # Konversi frame ke grayscale karena Haarcascade bekerja lebih baik pada citra grayscale
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah dalam frame
    # parameter ini bisa di-tuning untuk performa di Orange Pi
    faces = HaarCascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    # Loop untuk setiap wajah yang terdeteksi
    for (x, y, w, h) in faces:
        # Ekstrak area wajah (Region of Interest - ROI) dari frame asli (berwarna)
        face_roi = frame[y:y+h, x:x+w]

        # Periksa jika ROI valid
        if face_roi.size == 0:
            continue

        # Resize wajah ke ukuran yang dibutuhkan oleh FaceNet (160x160)
        face_roi_resized = cv2.resize(face_roi, (160, 160))
        
        # Konversi tipe data dan tambahkan dimensi batch
        face_pixels = face_roi_resized.astype('float32')
        samples = expand_dims(face_pixels, axis=0)

        # Dapatkan embedding dari wajah yang terdeteksi
        embedding = MyFaceNet.embeddings(samples)[0]

        # --- Proses Pengenalan Wajah ---
        # Hitung jarak Euclidean antara embedding wajah terdeteksi dengan semua embedding di database
        distances = np.linalg.norm(all_embeddings - embedding, axis=1)
        
        # Dapatkan jarak terkecil dan indexnya
        min_dist = np.min(distances)
        min_dist_idx = np.argmin(distances)
        
        # Tentukan threshold untuk pengenalan. Jika jarak lebih kecil dari threshold, wajah dikenali.
        # Nilai threshold ini mungkin perlu disesuaikan (umumnya antara 0.9 - 1.2)
        recognition_threshold = 0.7

        if min_dist < recognition_threshold:
            # Jika dikenali, ambil nama yang sesuai
            name = all_names[min_dist_idx]
            color = (0, 255, 0) # Hijau untuk dikenali
        else:
            # Jika tidak, beri label "Tidak Dikenal"
            name = "Tidak Dikenal"
            color = (0, 0, 255) # Merah untuk tidak dikenali

        # --- Tampilkan Hasil ---
        # Gambar kotak di sekitar wajah
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Tulis nama di atas kotak
        cv2.putText(frame, f"{name} ({min_dist:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    # Tampilkan frame hasil
    cv2.imshow('Pendeteksi Wajah - Tekan Q untuk Keluar', frame)

    # Tombol untuk keluar dari loop (tekan 'q')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
print("Menutup program...")
cap.release()
cv2.destroyAllWindows()