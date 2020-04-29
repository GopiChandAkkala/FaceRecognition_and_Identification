import cv2
import os
from PIL import Image
import numpy as np
import pickle

Base_Dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(Base_Dir,"Images")

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

label_ids = {}
current_id =0
y_labels = []
x_train = []

for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith("jpg"):
            path = os.path.join(root,file)            
            label = os.path.basename(root).replace(" ","_")
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
                
            id_ = label_ids[label] 
            
            pil_image = Image.open(path).convert("L")
            size = (400,400)
            final_image =pil_image.resize(size, Image.ANTIALIAS)
            
            image_array = np.array(final_image,"uint8")            
            faces = face_classifier.detectMultiScale(image_array, 1.3, 5) 
            
            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
            
with open("pickles/face-labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("recognizers/face-trainner.yml")