import cv2
import numpy as np
import os
from sklearn.svm import SVC
import pickle

def extract_features(frame, face_rect):
    """Extracts features from the frame within the given face rectangle."""
    x, y, w, h = face_rect
    mouth_roi = frame[y + h // 2:y + h, x:x + w]
    mouth_gray = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
    mouth_thresh = cv2.threshold(mouth_gray, 50, 255, cv2.THRESH_BINARY)[1]

    # Calculate features: white pixel count and mouth shape
    white_pixels = np.sum(mouth_thresh == 255)
    mouth_shape = np.mean(mouth_thresh)

    return white_pixels, mouth_shape

def train_svm_model(data, labels):
    """Trains an SVM model using the provided data and labels."""
    svm = SVC()
    svm.fit(data, labels)
    return svm

def detect_coffee_drinking(frame, face_rect, svm_model):
    """Detects coffee drinking using the SVM model and extracted features."""
    features = extract_features(frame, face_rect)
    prediction = svm_model.predict([features])
    return prediction[0] == 1  # Assuming 1 indicates drinking

# Load pre-trained SVM model (if available)
try:
    svm_model = pickle.load(open('svm_model.pkl', 'rb'))
except FileNotFoundError:
    print("SVM model not found. Training a new model...")
    # Collect training data (replace with your own data)
    training_data = []
    training_labels = []
    # ...
    svm_model = train_svm_model(training_data, training_labels)
    pickle.dump(svm_model, open('svm_model.pkl', 'wb'))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray,1.3, 5)

    for (x, y, w, h) in faces: 
        prediction = detect_coffee_drinking(frame, (x, y, w, h), svm_model)
        if prediction:
            print("Drinking coffee detected")
            # Play sound or perform other actions

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 