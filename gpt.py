import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Load known faces dynamically
known_face_encoding = []
known_face_names = []

for filename in os.listdir("known_faces"):
    if filename.endswith(".jpg"):
        image = face_recognition.load_image_file(f"known_faces/{filename}")
        encoding = face_recognition.face_encodings(image)
        if encoding:
            known_face_encoding.append(encoding[0])
            known_face_names.append(os.path.splitext(filename)[0])

students = known_face_names.copy()

face_locations = []
face_encodings = []
face_names = []

# Create CSV file
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

# Main loop
frame_count = 0
while True:
    _, frame = video_capture.read()
    frame_count += 1
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    if frame_count % 5 == 0:  # Process every 5th frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            if name in known_face_names and name in students:
                students.remove(name)
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                lnwriter.writerow([name, current_time])

    # Draw boxes and labels
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the frame
    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
f.close()
print("Attendance system terminated. Data saved.")
