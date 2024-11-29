import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime
import sqlite3

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()

    def load_known_faces(self, faces_dir="faces"):
        # Create faces directory if it doesn't exist
        if not os.path.exists(faces_dir):
            os.makedirs(faces_dir)
            
        for filename in os.listdir(faces_dir):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(faces_dir, filename)
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)
                if len(face_encodings) == 0:
                    raise ValueError("No face detected in the image")
                face_encoding = face_encodings[0]
                self.known_face_encodings.append(face_encoding)
                # Get name from filename (remove extension)
                self.known_face_names.append(os.path.splitext(filename)[0])

    def enroll_face(self, image_path, user_id):
        # Save face image
        save_path = f"faces/{user_id}.jpg"
        cv2.imwrite(save_path, cv2.imread(image_path))
        
        # Add to known faces
        image = face_recognition.load_image_file(save_path)
        face_encoding = face_recognition.face_encodings(image)[0]
        self.known_face_encodings.append(face_encoding)
        self.known_face_names.append(user_id)
        return True

    def remove_face(self, user_id):
        try:
            index = self.known_face_names.index(user_id)
            self.known_face_encodings.pop(index)
            self.known_face_names.pop(index)
            os.remove(f"faces/{user_id}.jpg")
            return True
        except:
            return False

    def log_recognition(self, user_id, status="Identified"):
        conn = sqlite3.connect('face_recognition.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO entry_logs (user_id, timestamp, status)
            VALUES (?, datetime('now', 'localtime'), ?)
        """, (user_id, status))
        
        conn.commit()
        conn.close()

    def recognize_face(self, frame):
        # Convert to RGB (face_recognition uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations first
        face_locations = face_recognition.face_locations(rgb_frame)
        
        # Get face landmarks
        face_landmarks = face_recognition.face_landmarks(rgb_frame, face_locations)
        
        # Get face encodings - modify this line
        face_encodings = []
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=1)
        
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]

            face_names.append(name)

        if face_names:
            for name in face_names:
                self.log_recognition(name)  # Log when someone is recognized

        return face_locations, face_names 