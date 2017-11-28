from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from commons import prepareTrainData
from commons import evaluateClassifier
from commons import printSubmission


name = "RFLR"


class RFLREstimator:

    def get_params(self, deep=False):
        return {}

    def fit(self, X, y):
        X_train, X_train_lr, y_train, y_train_lr = train_test_split(
            X,
            y,
            test_size=0.5,
        )
        self.rf = RandomForestClassifier(
            n_estimators=160,
            max_depth=10,
        )
        self.lr = LogisticRegression()
        self.rf.fit(X_train, y_train)
        self.enc = OneHotEncoder()
        self.enc.fit(self.rf.apply(X_train))
        self.lr.fit(self.enc.transform(self.rf.apply(X_train_lr)), y_train_lr)

    def predict_proba(self, X):
        return self.lr.predict_proba(self.enc.transform(self.rf.apply(X)))


if __name__ == "__main__":
    rflr = RFLREstimator()
    featureList = [
        'bids_cnt', 'price_std', 'device_cnt', 'response_min',
        'mean_bids_per_auction', 'price_max', 'response_median', 'country_cnt', 
        'price_mean', 'response_mean'
    ]
    X_train, y_train = prepareTrainData(featureList)
    evaluateClassifier(rflr, X_train, y_train, name)
    printSubmission(rflr, X_train, y_train, name, featureList)
