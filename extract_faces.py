import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import os

print(cv2.__version__)

dict = "F:\Pictures\Training"
outputDict = "training_data/"

global count
count = 500

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_face_box(img_name):
    img = cv2.imread(dict + "\\" + img_name)

    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        print(bcolors.WARNING + "ERROR CONVERTING TO GRAY " + img_name + bcolors.ENDC)
        print(bcolors.WARNING + "SKIPPING FOR NOW" + bcolors.ENDC)
        return

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        tempFace = img[y:y + h, x:x + w]
        tempFace = cv2.resize(tempFace, (500,500))

        #cv2.imshow('tempFace', tempFace)
        global count
        cv2.imwrite(outputDict + "/img_" + str(count) + ".jpg", tempFace)
        count +=1
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if (faces.__len__() != 0):
        img = cv2.resize(img, (500, 500))

        #cv2.imshow('Face', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img

def get_faces ():
    for i in os.listdir(dict):
        try:
            img_raw = Image.open(dict + "\\" + i)
        except IOError:
            print(i + "is not valid image file")
            continue

        img = get_face_box(i)




get_faces()