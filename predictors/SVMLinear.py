from sklearn import svm
from commons import *

name = "SVMLinear"

svc = svm.SVC(
    kernel='linear',
    cache_size=7000,
)
X_train, y_train = prepareTrainData()
evaluateClassifier(svc, X_train, y_train, name)
# printSubmission(svc, X_train, y_train, name)
