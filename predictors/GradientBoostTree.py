from sklearn.ensemble import GradientBoostingClassifier
from commons import *

name = "GBT"
gbt = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=2,
)
X_train, y_train = prepareTrainData()
evaluateClassifier(gbt, X_train, y_train, name)
printSubmission(gbt, X_train, y_train, name)
