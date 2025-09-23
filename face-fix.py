from PIL import Image
from keras.models import load_model
import numpy as np
from numpy import asarray
from numpy import expand_dims

import pickle
import cv2
from keras_facenet import FaceNet

# Initialize HaarCascade for face detection
HaarCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))

# Initialize FaceNet model
MyFaceNet = FaceNet()

# Load the known faces database
myfile = open("data.pkl", "rb")
database = pickle.load(myfile)
myfile.close()

# Start video capture
# MODIFIKASI 1: GANTI INDEKS KAMERA SESUAI KEBUTUHAN (misal: 1)
cap = cv2.VideoCapture(1) 

# MODIFIKASI 2: TURUNKAN RESOLUSI KAMERA
# Resolusi umum: 640x480 (480p) atau 320x240.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# MODIFIKASI 3: PENGATURAN UNTUK MENGURANGI RENDERING
frame_count = 0
RECOGNITION_INTERVAL = 5 # Lakukan pengenalan setiap 5 frame
identity = 'Menginisialisasi...' # Teks awal

while(True):
    ret, gbr1 = cap.read()
    if not ret:
        print("Gagal mengambil frame, keluar...")
        break
    
    # Hanya proses frame sesuai interval untuk mengurangi beban kerja
    if frame_count % RECOGNITION_INTERVAL == 0:
        # Ubah frame ke grayscale untuk deteksi wajah (lebih cepat)
        gray_frame = cv2.cvtColor(gbr1, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame
        wajah = HaarCascade.detectMultiScale(gray_frame, 1.1, 4)

        if len(wajah) > 0:
            x1, y1, width, height = wajah[0]
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height

            # Convert the frame to RGB for processing
            gbr = cv2.cvtColor(gbr1, cv2.COLOR_BGR2RGB)
            gbr = Image.fromarray(gbr)
            gbr_array = asarray(gbr)
            
            # Extract the face from the frame
            face = gbr_array[y1:y2, x1:x2]
            face = Image.fromarray(face)
            face = face.resize((160, 160))
            face = asarray(face)

            # Get the face embedding
            face = expand_dims(face, axis=0)
            signature = MyFaceNet.embeddings(face)

            # --- RECOGNITION LOGIC ---
            min_dist = 100
            
            # Compare the detected face with faces in the database
            for key, value in database.items():
                dist = np.linalg.norm(signature - value)
                if dist < min_dist:
                    min_dist = dist
                    identity = key # Simpan identitas yang ditemukan

            # --- ALGORITHM FOR UNRECOGNIZED FACES ---
            threshold = 1.0 
            if min_dist > threshold:
                identity = 'Wajah Tidak Dikenal'
        else:
            identity = ' ' # Tidak ada wajah terdeteksi

    # Naikkan penghitung frame
    frame_count += 1

    # Gambar kotak dan teks di SETIAP frame menggunakan hasil terakhir
    # Ini membuat tampilan tetap responsif meskipun pemrosesan berat dilewati
    if identity != ' ':
        # Cari wajah lagi dengan cepat untuk menggambar kotak (opsional, bisa diskip)
        # Atau gunakan koordinat terakhir. Untuk simpelnya, kita gambar saja teksnya.
        cv2.putText(gbr1, identity, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

    # Tampilkan gambar hasil akhir
    cv2.imshow('res', gbr1)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27: # Tekan ESC untuk keluar
        break

# Release resources
cv2.destroyAllWindows()
cap.release()