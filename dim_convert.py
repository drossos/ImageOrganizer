#Just a util to itterate through and resize all images within a given directory to a given nxn dim

import cv2
import os

dir = "F:\OpenProjects\ImageOrganizer\\training_data\Ryan Mak"
dim = 160

for i in os.listdir(dir):
    img  = cv2.imread(dir + "\\" + i)
    img = cv2.resize(img, (dim,dim))
    cv2.imwrite(dir + "\\" + i,img)
