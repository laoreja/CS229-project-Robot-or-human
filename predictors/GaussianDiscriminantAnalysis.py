from commons import *

name = "GDA"


# TODO: implementation of GDA
class GDAEstimator:
    def fit(self, X, y):
        self.phi = np.count_nonzero(y) * 1. / y.size
        self.mu0
        self.mu1
        self.sigma

    def predict(self, X):
        return [1 for x in X]


gda = GDAEstimator()
evaluateClassifier(gda, X_train, y_train, name)
printSubmission(gda, X_train, y_train, name)
