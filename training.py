import os
from pyexpat import features
import cv2 as cv
import numpy as np

people = []
DIR = r'Resources/Faces/train'
for p in os.listdir(DIR):
    people.append(p)

# print(people)
harCas = cv.CascadeClassifier('haar_face.xml')
features = []
labels = []
def createTrain():
    for person in people:
        path = os.path.join(DIR,person)
        label = people.index(person)

        for img in os.listdir(path):
            imgPath = os.path.join(path,img)

            imgList = cv.imread(imgPath)

            if imgList is None:
                continue
            
            gray = cv.cvtColor(imgList,cv.COLOR_BGR2GRAY)
            detect = harCas.detectMultiScale(gray,1.1,4)

            for x,y,w,h in detect:
                roi = gray[y:y+h,x:x+w]
                features.append(roi)
                labels.append(label)

createTrain()

print('Training done...')

features = np.array(features,dtype='object')
labels = np.array(labels)

faceRec = cv.face.LBPHFaceRecognizer_create()
faceRec.train(features,labels)

faceRec.save('trainedFaces.yml')
np.save('features.npy',features)
np.save('labels.npy',labels)