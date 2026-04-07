import cv2
import face_recognition
import numpy as np
import os

path = 'images'
images = []
classNames = []

# Check folder
if not os.path.exists(path):
    print("Images folder not found")
    exit()

myList = os.listdir(path)

if len(myList) == 0:
    print("No images in folder")
    exit()

# Load images
for cl in myList:
    curImg = face_recognition.load_image_file(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# Encode faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)

        if len(encodings) > 0:
            encodeList.append(encodings[0])
    return encodeList

encodeListKnown = findEncodings(images)

print("Encoding Complete ✅")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera not working ❌")
        break

    # Resize for speed
    imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faces = face_recognition.face_locations(imgS)
    encodes = face_recognition.face_encodings(imgS, faces)

    for encodeFace, faceLoc in zip(encodes, faces):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex] and faceDis[matchIndex] < 0.5:
            name = classNames[matchIndex].upper()
        else:
            name = "UNKNOWN"

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()