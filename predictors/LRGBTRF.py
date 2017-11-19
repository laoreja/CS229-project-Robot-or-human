from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from commons import *

name = "LRGBTRF"
lr = LogisticRegression()
gbt = GradientBoostingClassifier(
    n_estimators=600,
    max_depth=15,
    min_samples_leaf=2,
)
rf = RandomForestClassifier(
    n_estimators=600,
    max_depth=15,
    min_samples_leaf=2,
)

X_train, y_train = prepareTrainData()
printSubmissionAverage([lr, gbt, rf], X_train, y_train, name)
