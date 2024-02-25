import glob

import face_recognition
import cv2
import os

imagePaths = glob.glob("faces/*.jpg")
knownEncodings = []
knownNames = []
# перебираем все изображения
for (i, imagePath) in enumerate(imagePaths):
    name = imagePath.split(os.path.sep)[-1][:-4]
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb,model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

data = {"encodings": knownEncodings, "names": knownNames}

print("Streaming started")
video_capture = cv2.VideoCapture(0)  # возможно тут надо будет указать другую цифру
while True:

    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_recognition.face_locations(gray)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"],
                                                 encoding)
        name = "Unknown"
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]

            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)

        names.append(name)
    for ((t, r, b, l), name) in zip(faces, names):
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, name, (l + 5, b - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()