from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from new_commons import *

name = "RF"

rf = RandomForestClassifier(
    n_estimators=160,
    max_depth=8,
    max_features=10,
)

X_train, y_train = prepareTrainData()
evaluateClassifier(rf, X_train, y_train, name)
printSubmission(rf, X_train, y_train, name)
