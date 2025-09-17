import tensorflow as tf
from keras_facenet import FaceNet

# Inisialisasi model FaceNet
MyFaceNet = FaceNet()

# Simpan model sebagai file .h5
MyFaceNet.model.save('facenet_model.h5')

# Muat model .h5 dan konversi ke TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(MyFaceNet.model)
tflite_model = converter.convert()

# Simpan model TFLite
with open('facenet_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model berhasil dikonversi dan disimpan sebagai facenet_model.tflite")