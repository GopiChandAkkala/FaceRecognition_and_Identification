import cv2
import os 

User_Name = input("Enter the name of the User: ")
parent_dir = 'E:\Python\Py_DS_ML\Practice\DeepLearning\Face_Recognition\Tutorial\Face_recog&Identify/Images/'
save_path = os.path.join(parent_dir, User_Name) 
os.mkdir(save_path)

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Initialize Webcam
cap = cv2.VideoCapture(0)
count = 0

# Collect 100 samples of your face from webcam input
while True:

    ret, frame = cap.read()
    faces = face_classifier.detectMultiScale(frame, 1.3, 5)
    for (x,y,w,h) in faces:
        x=x-10
        y=y-10
        cropped_face = frame[y:y+h+50, x:x+w+50]      
    count += 1
    face = cv2.resize(cropped_face, (400, 400))      
    file_name = str(count) + '.jpg'
    save_file = os.path.join(save_path , file_name)
    print(save_file)
    cv2.imwrite(save_file, face)

    # Put count on images and display live count
    cv2.putText(frame, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Face Cropper', frame)    
    
    if cv2.waitKey(1) & 0xFF == ord('q') or count == 200:
        break
   
        
cap.release()
cv2.destroyAllWindows()      
print("Collecting Samples Complete")
