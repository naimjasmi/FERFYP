from tensorflow.keras.models import load_model
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

# Load the model
model = load_model('C:/Users/Asus/emotion_model')

# Define emotions
emotions = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad']

def detect_emotion(emotion_source, stop_flag):
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Set the width
    cap.set(4, 480)  # Set the height

    while not stop_flag.is_set():
        ret, frame = cap.read()
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # Consider only the first detected face
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = model.predict(roi)[0]
            emotion_label = emotions[preds.argmax()]

            if emotion_source:
                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                emotion_source.put((frame_bytes, emotion_label))
        else:
            # No faces detected, yield the frame without emotion label
            if emotion_source:
                emotion_source.put((frame_bytes, "NoFaceDetected"))

    cap.release()

