import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import os

def get_face_class(direct):
    imgs = list()
    for i in os.listdir(direct):
        imgs.append(cv2.imread(direct+"\\" + i))

    return imgs

def aggr_dataset(direct,train_set):
    x, y = list(), list()
    extens = "val"
    # print(os.listdir(direct))
    for i in os.listdir(direct):
        if (os.path.isdir(direct + "\\" + i)):
            if(train_set):
                extens = "train"
            faces = get_face_class(direct  + "\\" + i + "\\" + extens)
            labels = [i for _ in range(len(faces))]

            print("Loaded %d images for class %s" % (len(faces), labels[0]))
            x.extend(faces)
            y.extend(labels)
    return np.asarray(x), np.asarray(y)

trainX,trainY = aggr_dataset("F:\OpenProjects\ImageOrganizer\\training_data", train_set = True)
testX,testY = aggr_dataset("F:\OpenProjects\ImageOrganizer\\training_data", train_set = False)

np.savez_compressed('boys-faces-dataset.npz' , trainX, trainY, testX, testY)
