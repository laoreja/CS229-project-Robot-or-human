from sklearn import svm
from commons import *

name = "SVMRBF"

svc = svm.SVC(kernel='rbf')
X_train, y_train = prepareTrainData()
evaluateClassifier(svc, X_train, y_train, name)
printSubmission(svc, X_train, y_train, name)
