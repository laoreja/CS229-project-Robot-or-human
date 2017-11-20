from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from commons import *
# import xgboost as xgb

name = "Many"


def gradient_model():
    model = GradientBoostingClassifier(n_estimators=200,
                                       random_state=1111,
                                       max_depth=5,
                                       learning_rate=0.03,
                                       max_features=40, )
    return model


def forest_model():
    model = RandomForestClassifier(n_estimators=160, max_features=35,
                                   max_depth=8, random_state=1111,
                                   criterion='entropy', )
    return model


def forest_ada_model():
    model = RandomForestClassifier(n_estimators=160, max_features=35,
                                   max_depth=8, random_state=1111,
                                   criterion='entropy', )
    model = AdaBoostClassifier(base_estimator=model, n_estimators=25)
    return model


def forest_calibrated():
    model = RandomForestClassifier(n_estimators=60, max_features=33,
                                   max_depth=8, random_state=1111,
                                   criterion='entropy', )
    model = CalibratedClassifierCV(model, method='isotonic', cv=5)
    return model


def forest_bagging():
    model = RandomForestClassifier(n_estimators=150, max_features=40,
                                   max_depth=8, random_state=1111,
                                   criterion='entropy', )
    model = BaggingClassifier(base_estimator=model, max_features=0.80,
                              n_jobs=-1, n_estimators=50)
    return model


X_train, y_train = prepareTrainData()
predictors = [
    gradient_model(),
    forest_model(),
    forest_ada_model(),
    forest_bagging(),
    forest_calibrated(),
]
printSubmissionAverage(predictors, X_train, y_train, name)
