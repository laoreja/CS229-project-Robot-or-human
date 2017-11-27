from sklearn.ensemble import RandomForestClassifier
from commons import *

name = "RF"

rf = RandomForestClassifier(
    n_estimators=160,
    max_depth=8,
)

X_train, y_train = prepareTrainData()
evaluateClassifier(rf, X_train, y_train, name)
printSubmission(rf, X_train, y_train, name)
