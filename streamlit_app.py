import streamlit as st

import cv2
import pickle
import numpy as np
import os

# Initialize video capture and face detector
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces_data = []
i = 0
name = input("ENTER YOUR NAME: ")

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to access the camera.")
        break

    # Convert to grayscale for detection
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(grey, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (50, 50))

        # Save frames periodically
        if len(faces_data) <= 100 and i % 10 == 0:
            faces_data.append(resized_img)

        i += 1

        # Display data on screen
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) == 100:
        break

video.release()
cv2.destroyAllWindows()

# Reshape and save data
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(100, -1)

data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

names_file = os.path.join(data_dir, 'names.pkl')
faces_file = os.path.join(data_dir, 'faces_data.pkl')

if not os.path.exists(names_file):
    names = [name] * 100
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)
else:
    with open(names_file, 'rb') as f:
        names = pickle.load(f)
    names += [name] * 100
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)

if not os.path.exists(faces_file):
    with open(faces_file, 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open(faces_file, 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)
    with open(faces_file, 'wb') as f:
        pickle.dump(faces, f)
