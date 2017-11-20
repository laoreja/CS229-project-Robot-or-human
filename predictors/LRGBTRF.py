from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from commons import *

name = "LRGBTRF"
lr = LogisticRegression()
gbt = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=5,
    max_features=40,
    learning_rate=0.03,
)
rf = RandomForestClassifier(
    n_estimators=160,
    max_depth=8,
    max_features=35,
)

X_train, y_train = prepareTrainData()
printSubmissionAverage([lr, gbt, rf], X_train, y_train, name)
