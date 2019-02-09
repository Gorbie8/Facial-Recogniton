import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recogniser = cv2.face.LBPHFaceRecognizer_create()
recogniser.read("trainer.yml")
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

person_1 = 0
person_2 = 0

cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        # print(x, y ,w ,h)
        roi_gray = gray[y:y+h, x:x+w]

        # Recognition
        id_, conf = recogniser.predict(roi_gray)
        if conf <= 130:
            # print(id_)
            # prints the name of the id
            print(labels[id_])
            # prints confidence
            print(conf)
            if id_ == 1:
                person_1 += 1
            elif id_ == 2:
                person_2 += 1
            else:
                pass

            print(person_1, person_2)

        # save photo of last recognized frame
        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)

        color = (255, 0, 0)   # BGR
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
