import numpy as np
import cv2
import pickle


face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")

with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(frame, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]        
        id_, conf = recognizer.predict(roi_gray)        
        if conf>=45 and conf <= 85:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (0, 255, 0)    		
            cv2.putText(frame, name, (x,y), font, 1, color, 3, cv2.LINE_AA)            
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)            
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()