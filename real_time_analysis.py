import cv2
import face_recognition
import time
import os

# Step 1: Load authorized employee face encodings
authorized_encodings = []
authorized_names = []

authorized_folder = "authorized_faces"

for filename in os.listdir(authorized_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(authorized_folder, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        authorized_encodings.append(encoding)
        authorized_names.append(filename.split(".")[0])  # Use filename (without extension) as the name

# Step 2: Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

absence_start_time = None
absence_threshold = 10  # Threshold for absence detection (in seconds)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error accessing the camera!")
        break

    # Convert frame from BGR to RGB (face_recognition requires RGB format)
    rgb_frame = frame[:, :, ::-1]

    # Step 3: Detect faces in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Check if no face is detected
    if len(face_encodings) == 0:
        if absence_start_time is None:
            absence_start_time = time.time()
        elif time.time() - absence_start_time > absence_threshold:
            print("No person detected for {} seconds!".format(absence_threshold))
    else:
        absence_start_time = None  # Reset absence timer if faces are detected

    # Step 4: Recognize faces
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(authorized_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"

        # If a match is found, get the name of the matched person
        if True in matches:
            first_match_index = matches.index(True)
            name = authorized_names[first_match_index]

        # Step 5: Draw a rectangle around the face and label it
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Check for unauthorized person
        if name == "Unknown":
            print("Unauthorized person detected!")

    # Step 6: Display the frame
    cv2.imshow("Real-Time Feed Analysis", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
