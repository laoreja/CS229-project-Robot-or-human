from sklearn.linear_model import LogisticRegression
from commons import *

name = "LR"

lr = LogisticRegression()
X_train, y_train = prepareTrainData()
evaluateClassifier(lr, X_train, y_train, name)
printSubmission(lr, X_train, y_train, name)
