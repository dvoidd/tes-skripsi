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
cap = cv2.VideoCapture(0)

while(True):
    _, gbr1 = cap.read()

    # Detect faces in the frame
    wajah = HaarCascade.detectMultiScale(gbr1, 1.1, 4)

    if len(wajah) > 0:
        x1, y1, width, height = wajah[0]
    else:
        # If no face is detected, continue to the next frame
        cv2.imshow('res', gbr1)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
        continue

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
    identity = ' '
    
    # Compare the detected face with faces in the database
    for key, value in database.items():
        dist = np.linalg.norm(signature - value)
        if dist < min_dist:
            min_dist = dist
            identity = key

    # --- ALGORITHM FOR UNRECOGNIZED FACES ---
    # Set a threshold for recognition. You may need to adjust this value.
    threshold = 1.0 
    
    if min_dist > threshold:
        identity = 'Wajah Tidak Dikenal' # Unrecognized Face

    # Display the result on the screen
    cv2.putText(gbr1, identity, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(gbr1, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show the final image
    cv2.imshow('res', gbr1)
    k = cv2.waitKey(5) & 0xFF
    if k == 27: # Press ESC to exit
        break

# Release resources
cv2.destroyAllWindows()
cap.release()