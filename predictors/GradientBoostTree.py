from sklearn.ensemble import GradientBoostingClassifier
from commons import *

name = "GBT"
gbt = GradientBoostingClassifier(
    n_estimators=600,
    max_depth=15,
    min_samples_leaf=2,
)
X_train, y_train = prepareTrainData()
evaluateClassifier(gbt, X_train, y_train, name)
printSubmission(gbt, X_train, y_train, name)
