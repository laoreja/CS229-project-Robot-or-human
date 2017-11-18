from sklearn.tree import DecisionTreeClassifier
from commons import *

name = "DT"

dt = DecisionTreeClassifier()

X_train, y_train = prepareTrainData()
evaluateClassifier(dt, X_train, y_train, name)
printSubmission(dt, X_train, y_train, name)
