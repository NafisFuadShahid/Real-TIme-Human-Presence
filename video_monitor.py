import cv2
import face_recognition
import numpy as np
import datetime
import time
from typing import List, Dict, Optional
import os

class VideoPresenceMonitor:
    def __init__(self, 
                 known_faces_dir: str,
                 absence_threshold: int = 5,  # seconds
                 unauthorized_threshold: int = 3):  # seconds
        
        self.known_face_encodings = []
        self.known_face_names = []
        self.absence_threshold = absence_threshold
        self.unauthorized_threshold = unauthorized_threshold
        self.last_presence_time = time.time()
        self.unauthorized_start_time = None
        
        # Load known faces
        self.load_known_faces(known_faces_dir)
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Could not open video capture device")

    def load_known_faces(self, faces_dir: str) -> None:
        """Load and encode known faces from directory."""
        print("Loading known faces...")
        for filename in os.listdir(faces_dir):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(faces_dir, filename)
                image = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(image)
                
                if encoding:
                    self.known_face_encodings.append(encoding[0])
                    # Use filename without extension as person's name
                    self.known_face_names.append(os.path.splitext(filename)[0])
                else:
                    print(f"Warning: No face found in {filename}")
        
        print(f"Loaded {len(self.known_face_encodings)} known faces")

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, str]:
        """Process a single frame and return the annotated frame and status."""
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find faces in frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        status = "No faces detected"
        face_names = []

        for face_encoding in face_encodings:
            # Compare with known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]

            face_names.append(name)

        # Update status and timing
        if not face_locations:
            if time.time() - self.last_presence_time > self.absence_threshold:
                status = "ALERT: No presence detected"
            self.unauthorized_start_time = None
        else:
            self.last_presence_time = time.time()
            if all(name == "Unknown" for name in face_names):
                if self.unauthorized_start_time is None:
                    self.unauthorized_start_time = time.time()
                elif time.time() - self.unauthorized_start_time > self.unauthorized_threshold:
                    status = "ALERT: Unauthorized presence detected"
            else:
                status = f"Authorized: {', '.join(name for name in face_names if name != 'Unknown')}"
                self.unauthorized_start_time = None

        # Draw results on frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw box
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Draw label
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        # Add status to frame
        cv2.putText(frame, status, (10, 30), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
        
        return frame, status

    def run(self):
        """Main loop for video processing."""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                processed_frame, status = self.process_frame(frame)
                
                # Display timestamp
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(processed_frame, timestamp, (10, processed_frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

                # Show frame
                cv2.imshow('Video Presence Monitor', processed_frame)

                # Log alerts to console
                if status.startswith("ALERT"):
                    print(f"{timestamp}: {status}")

                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cleanup()

    def cleanup(self):
        """Release resources."""
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    """Main entry point."""
    # Directory containing known face images
    known_faces_dir = "known_faces"
    
    # Ensure the known_faces directory exists
    if not os.path.exists(known_faces_dir):
        os.makedirs(known_faces_dir)
        print(f"Created directory '{known_faces_dir}'. Please add employee photos before running.")
        return

    try:
        monitor = VideoPresenceMonitor(
            known_faces_dir=known_faces_dir,
            absence_threshold=5,  # 5 seconds
            unauthorized_threshold=3  # 3 seconds
        )
        monitor.run()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()