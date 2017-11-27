from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from commons import prepareTrainData
from commons import evaluateClassifier
from commons import printSubmission


name = "GBTLR"


class RFLREstimator:

    def get_params(self, deep=False):
        return {}

    def fit(self, X, y):
        X_train, X_train_lr, y_train, y_train_lr = train_test_split(
            X,
            y,
            test_size=0.5,
        )
        self.gbt = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=2,
        )
        self.lr = LogisticRegression()
        self.gbt.fit(X_train, y_train)
        self.enc = OneHotEncoder()
        self.enc.fit(self.gbt.apply(X_train)[:, :, 0])
        self.lr.fit(
            self.enc.transform(self.gbt.apply(X_train_lr)[:, :, 0]),
            y_train_lr,
        )

    def predict_proba(self, X):
        return self.lr.predict_proba(
            self.enc.transform(self.gbt.apply(X)[:, :, 0])
        )


if __name__ == "__main__":
    gbtlr = RFLREstimator()
    X_train, y_train = prepareTrainData()
    evaluateClassifier(gbtlr, X_train, y_train, name)
    printSubmission(gbtlr, X_train, y_train, name)
