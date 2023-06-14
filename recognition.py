import cv2 as cv
import numpy as np

harCas = cv.CascadeClassifier('haar_face.xml')

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

features = np.load('features.npy',allow_pickle=True)
labels = np.load('labels.npy')

faceRec = cv.face.LBPHFaceRecognizer_create()
faceRec.read('trainedFaces.yml')

img = cv.imread('Resources/Faces/val/ben_afflek/2.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
detect = harCas.detectMultiScale(gray,1.1,4)
for x,y,w,h in detect:
    roi = gray[y:y+h,x:x+w]
    label, confidence = faceRec.predict(roi)

    print(f'Label = {people[label]} with a confidence of {confidence}')
    cv.putText(img,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(255,0,0),2)
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

cv.imshow('Person',img)
cv.waitKey(0)
