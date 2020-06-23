import numpy as np
from sklearn.svm import OneClassSVM
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
import copy

a = np.load('./featureSet/f128x128_real0.npy')
b = np.load('./featureSet/f128x128_real1.npy')
x_train = np.concatenate((a,b), axis=0)

x_test_real = np.load('./featureSet/f128x128_real2.npy')
x_test_PGGAN = np.load('./featureSet/f128x128_PGGAN.npy')

# print(len(list(map(np.unique,x_train))))

nuu=0.1

# clf = OneClassSVM(kernel='rbf', gamma='scale', nu=nuu).fit(x_train)
# #saving model for repeated use
# tmpfile = 'modelRc128.sav'
# joblib.dump(clf, tmpfile)

tmp = 'modelRc128.sav'
loadModel = joblib.load(tmp)

y_pred_test_real = loadModel.predict(x_test_real)
y_pred_test_PGGAN = loadModel.predict(x_test_PGGAN)

n_error_test_real = y_pred_test_real[y_pred_test_real == -1].size
n_error_test_PGGAN = y_pred_test_PGGAN[y_pred_test_PGGAN == -1].size

print("nu = ", nuu,"\n")

print("\nerror sample count",n_error_test_real,"/",len(y_pred_test_real)," Real Images")
print("accuracy testingRealImages= ", 1 - n_error_test_real / len(y_pred_test_real))

print("\nerror sample count", n_error_test_PGGAN,"/",len(y_pred_test_PGGAN)," DNGs")
print("accuracy testingPGGAN= ", 1 - n_error_test_PGGAN / len(y_pred_test_PGGAN))

print(accuracy_score([1 for i in range(len(y_pred_test_real))], y_pred_test_real))
print(accuracy_score([-1 for i in range(len(y_pred_test_PGGAN))], y_pred_test_PGGAN))
print(confusion_matrix([1 for i in range(len(y_pred_test_PGGAN))], y_pred_test_PGGAN))
