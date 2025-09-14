import streamlit as st
import cv2
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime
import os
import time

# --- File and Directory Setup ---
# Create the 'faces' directory if it doesn't exist
if not os.path.exists('faces'):
    os.makedirs('faces')

# Path to the CSV file for attendance logging
ATTENDANCE_FILE = 'attendance.csv'

# --- Utility Functions ---

# Function to load known faces and their encodings
def load_known_faces():
    known_face_encodings = []
    known_face_names = []

    for name in os.listdir('faces'):
        # Check if it's a directory
        if os.path.isdir(os.path.join('faces', name)):
            for filename in os.listdir(os.path.join('faces', name)):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join('faces', name, filename)
                    try:
                        image = face_recognition.load_image_file(image_path)
                        encoding = face_recognition.face_encodings(image)[0]
                        known_face_encodings.append(encoding)
                        known_face_names.append(name)
                    except IndexError:
                        st.warning(f"Could not find a face in the image: {image_path}")
    return known_face_encodings, known_face_names

# Function to mark attendance in the CSV file
def mark_attendance(name):
    with open(ATTENDANCE_FILE, 'a+') as f:
        f.seek(0)
        df = pd.read_csv(f) if os.path.getsize(ATTENDANCE_FILE) > 0 else pd.DataFrame(columns=['Name', 'Date', 'Time'])
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Check if attendance is already marked for today
        if not ((df['Name'] == name) & (df['Date'] == current_date)).any():
            now = datetime.now()
            dt_string = now.strftime('%H:%M:%S')
            f.write(f'\n{name},{current_date},{dt_string}')
            st.success(f"Attendance marked for {name} at {dt_string}!")

# --- Streamlit UI and Logic ---
st.title("Face Recognition Attendance System")
st.markdown("---")

# Navigation sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Register Face", "Mark Attendance", "View Records"])
st.sidebar.markdown("---")

# Main content area
if page == "Register Face":
    st.header("Register New Face")
    st.write("Enter your name and take a picture to register your face.")
    
    name_input = st.text_input("Enter your full name:", key="reg_name")
    
    col1, col2 = st.columns(2)
    with col1:
        start_capture = st.button("Start Camera")
    with col2:
        stop_capture = st.button("Stop Camera")

    if start_capture:
        if not name_input:
            st.error("Please enter a name before starting the camera.")
        else:
            st.session_state['capture_active'] = True
            st.session_state['name_to_register'] = name_input.replace(" ", "_")
    
    if stop_capture:
        st.session_state['capture_active'] = False
        st.info("Camera stopped.")

    if 'capture_active' in st.session_state and st.session_state['capture_active']:
        st.write("Taking pictures...")
        camera_placeholder = st.empty()
        
        cap = cv2.VideoCapture(0)
        face_count = 0
        img_dir = os.path.join('faces', st.session_state['name_to_register'])
        
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        
        while cap.isOpened() and face_count < 10:
            ret, frame = cap.read()
            if not ret:
                break
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
            camera_placeholder.image(frame, channels="BGR", use_column_width=True)
            
            # Save the face image
            if len(face_locations) > 0:
                cv2.imwrite(os.path.join(img_dir, f"{st.session_state['name_to_register']}_{face_count}.jpg"), frame)
                face_count += 1
                time.sleep(0.5)
        
        cap.release()
        st.session_state['capture_active'] = False
        st.success(f"{face_count} images for '{st.session_state['name_to_register']}' have been saved.")


elif page == "Mark Attendance":
    st.header("Mark Your Attendance")
    st.write("Look at the camera to mark your attendance. Your name will appear on the screen if you are recognized.")
    
    known_face_encodings, known_face_names = load_known_faces()

    if not known_face_encodings:
        st.warning("No faces registered yet. Please register a face first.")
    else:
        st.info("Camera is starting...")
        camera_placeholder = st.empty()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam.")
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"
                    
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        mark_attendance(name)
                    
                    face_names.append(name)
                
                # Display results
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                
                camera_placeholder.image(frame, channels="BGR", use_column_width=True)

elif page == "View Records":
    st.header("Attendance Records")
    if os.path.exists(ATTENDANCE_FILE) and os.path.getsize(ATTENDANCE_FILE) > 0:
        df = pd.read_csv(ATTENDANCE_FILE)
        st.dataframe(df)
    else:
        st.info("No attendance records found yet.")
