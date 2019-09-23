import tensorflow as tf
import numpy as np
import sklearn as skl
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

# load dataset
data = np.load('model_data\\boys-faces-embeddings.npz')
trainX, trainY, testX, testY = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

# normalize input vectors
in_encoder =Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainY)
trainy = out_encoder.transform(trainY)
testy = out_encoder.transform(testY)

# fit model
model = SVC(kernel='linear')
model.fit(trainX, trainy)


train_predict = model.predict(trainX)
test_predict = model.predict(testX)
# score
score_train = accuracy_score(trainy, train_predict)
score_test = accuracy_score(testy, test_predict)
# summarize
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))

joblib.dump(model, "trained_model.pkl")
