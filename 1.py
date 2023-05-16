import cv2
import numpy as np
import csv
import os

# Create a face cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
  # Get a frame from the webcam
  ret, frame = cap.read()

  # Convert the frame to grayscale
  grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Detect faces in the frame
  faces = face_cascade.detectMultiScale(grayscale_frame, scaleFactor=1.1, minNeighbors=5)

  # For each face in the frame
  for face in faces:
    # Find the face encoding
    face_encoding = np.array([
      face[0], face[1], face[2], face[3]
    ])

    # Find the name of the person in the face
    person_name = "DapaArhama"
    face_encodings = []

    for i in range(len(face_encodings)):
      if np.array_equal(face_encodings[i], face_encoding):
        person_name = os.path.splitext(os.path.basename("dataset/" + str(i + 1) + ".jpg"))[0]
        break

    # Write the person's name to the CSV file
    with open("absensi.csv", "a") as csv_file:
      csv_writer = csv.writer(csv_file)
      csv_writer.writerow([person_name])

  # Display the frame
  cv2.imshow("Attendance", frame)

  # Wait for a key press
  key = cv2.waitKey(1)

  # If the key `q` is pressed, stop the program
  if key == ord("q"):
    break

# Close the webcam
cap.release()

# Destroy all windows
cv2.destroyAllWindows()