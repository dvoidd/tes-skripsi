from PIL import Image
import numpy as np
from numpy import asarray, expand_dims

import pickle
import cv2
# Import TFLite Interpreter, bukan keras_facenet
import tflite_runtime.interpreter as tflite

# --- KONFIGURASI ---
FRAME_SKIP = 7
PROC_WIDTH = 640
PROC_HEIGHT = 480
RECOGNITION_THRESHOLD = 0.7

# --- Inisialisasi Model ---
print("Memuat model dan data...")
HaarCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))

# Muat model TFLite
interpreter = tflite.Interpreter(model_path='facenet_model.tflite')
interpreter.allocate_tensors()

# Dapatkan detail input dan output dari model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Muat database wajah
with open("data.pkl", "rb") as myfile:
    database = pickle.load(myfile)
print("Model dan data berhasil dimuat.")

# --- Mulai Video Capture ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, PROC_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PROC_HEIGHT)

frame_counter = 0
last_identity = ' '
last_box = (0, 0, 0, 0)

print("Memulai kamera...")
while(True):
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera. Keluar...")
        break

    gbr_small = cv2.resize(frame, (PROC_WIDTH, PROC_HEIGHT))

    if frame_counter % FRAME_SKIP == 0:
        wajah = HaarCascade.detectMultiScale(gbr_small, 1.1, 4)

        if len(wajah) > 0:
            x1, y1, width, height = wajah[0]
            x2, y2 = x1 + width, y1 + height

            # Ekstrak wajah dan resize
            face = gbr_small[y1:y2, x1:x2]
            face_resized = cv2.resize(face, (160, 160))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

            # Pre-proses untuk model TFLite
            face_array = expand_dims(face_rgb, axis=0)
            face_array = face_array.astype('float32')
            
            # Dapatkan face embedding dari model TFLite
            interpreter.set_tensor(input_details[0]['index'], face_array)
            interpreter.invoke()
            signature = interpreter.get_tensor(output_details[0]['index'])

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

            last_identity = f'{identity} ({min_dist:.2f})'
            last_box = (x1, y1, x2, y2)
        else:
            last_identity = ' '
            last_box = (0, 0, 0, 0)

    if last_identity != ' ':
        (x1, y1, x2, y2) = last_box
        cv2.putText(gbr_small, last_identity, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(gbr_small, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Face Recognition', gbr_small)
    
    frame_counter += 1
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

print("Menutup aplikasi.")
cv2.destroyAllWindows()
cap.release()