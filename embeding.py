import tensorflow as tf
import numpy as np

data = np.load('model_data\\boys-faces-dataset.npz')
trainX, trainY, testX, testY = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

print('Loaded: ', trainX.shape, trainY.shape, testX.shape, testY.shape)
model = tf.keras.models.load_model('model_data\\facenet_keras.h5')


def face_embeding(model, face_img):
    # scale pixel values
    face_img = face_img.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_img.mean(), face_img.std()
    face_pixels = (face_img - mean) / std
    # transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]


# convert each face in the train set to an embedding
newTrainX = list()
for face_pixels in trainX:
    embedding = face_embeding(model, face_pixels)
    newTrainX.append(embedding)
newTrainX = np.asarray(newTrainX)
print(newTrainX.shape)
# convert each face in the test set to an embedding
newTestX = list()
for face_pixels in testX:
    embedding = face_embeding(model, face_pixels)
    newTestX.append(embedding)
newTestX = np.asarray(newTestX)
print(newTestX.shape)
# save arrays to one file in compressed format
np.savez_compressed('model_data\\boys-faces-embeddings.npz', newTrainX, trainY, newTestX, testY)
