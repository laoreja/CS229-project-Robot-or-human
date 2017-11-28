import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from RFLR import RFLREstimator
from GBTLR import GBTLREstimator
from GaussianDiscriminantAnalysis import GDAEstimator
from commons import prepareTrainData
from commons import evaluateClassifier
from commons import printSubmission

name = "Voting"


class VotingEstimator:

    def __init__(self):
        rf = RandomForestClassifier(
            n_estimators=160,
            max_depth=20,
        )
        gbt = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=2,
        )
        lr = LogisticRegression()
        rflr = RFLREstimator()
        gbtlr = GBTLREstimator()
        rfada = AdaBoostClassifier(
            base_estimator=RandomForestClassifier(
                n_estimators=160,
                max_depth=10,
            ),
            n_estimators=25,
        )
        gda = GDAEstimator()
        self.estimators = [
            rf,
            gbt,
            lr,
            rflr,
            gbtlr,
            rfada,
            # gda,
        ]
        self.numEstimators = len(self.estimators)

    def get_params(self, deep=False):
        return {}

    def fit(self, X, y):
        for estimator in self.estimators:
            estimator.fit(X, y)

    def predict_proba(self, X):
        res = np.zeros([X.shape[0], 2])
        for estimator in self.estimators:
            res += estimator.predict_proba(X)
        return res / self.numEstimators


v = VotingEstimator()
X_train, y_train = prepareTrainData(newFeature=False)
evaluateClassifier(v, X_train, y_train, name)
printSubmission(v, X_train, y_train, name, newFeature=False)
