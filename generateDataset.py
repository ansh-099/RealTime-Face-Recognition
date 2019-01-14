import cv2
import numpy as np
import time

data = []
name = input("Enter Name - ")
no_of_pics = int(input("Enter No Of Pics - "))
capture = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while no_of_pics > 0 :
    
    returned ,read_image = capture.read()
    
    if not  returned:
        continue
    
    faces = face_cascade.detectMultiScale(read_image, 1.3,5)
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    faces = faces[:1]

    for face in faces:
        x,y,w,h = face

        only_face = read_image[y:y+h,x:x+w]
        # cv2.imshow("face",only_face)
        # cv2.waitKey()
        only_face = cv2.resize(only_face , (100,100))
        data.append(only_face)
        no_of_pics -=1 


print(len(data))
data = np.array(data)
print("data shape ",data.shape)
print(data)
data = data.reshape((data.shape[0],-1))
print(data.shape)
np.save(("dataset/"+name),data)
capture.release()
cv2.destroyAllWindows()


    