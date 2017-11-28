from sklearn.tree import DecisionTreeClassifier
from commons import *

name = "DT"

dt = DecisionTreeClassifier()

featureList = [
    'tdiff_min', 'response_min'
]

X_train, y_train = prepareTrainData(featureList)
evaluateClassifier(dt, X_train, y_train, name)
printSubmission(dt, X_train, y_train, name, featureList)
