import pandas as pd
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp

excludeBidders = [
    "74a35c4376559c911fdb5e9cfb78c5e4btqew",
    "7fab82fa5eaea6a44eb743bc4bf356b3tarle",
    "91c749114e26abdb9a4536169f9b4580huern",
    "bd0071b98d9479130e5c053a244fe6f1muj8h",
    "f35082c6d72f1f1be3dd23f949db1f577t6wd",
]


def filterFeatures(dfFeatures):
    lst = [x for x in list(dfFeatures) if len(x) > 2]
    return dfFeatures[lst]


def prepareTrainData():
    dfFeatures = pd.read_csv("../features/new_all_feat.csv")
    for b in excludeBidders:
        dfFeatures = dfFeatures[dfFeatures.bidder_id != b]
    dfLabels = pd.read_csv("../data/train.csv").drop(
        ['address', 'payment_account'],
        axis=1,
    )
    common = dfFeatures.merge(dfLabels, on='bidder_id')
    X_train = np.array(common.drop(['bidder_id', 'outcome'], axis=1))
    y_train = np.ravel(common[['outcome']])
    return X_train, y_train


def evaluateClassifier(classifier, X_train, y_train, name):
    print "[{}] Cross Validation Score: {}".format(
        name,
        cross_val_score(
            classifier,
            X_train,
            y_train,
            cv=4,
            n_jobs=3,
            scoring="roc_auc",
        ).mean()
    )


def printSubmission(classifier, X_train, y_train, name):
    classifier.fit(X_train, y_train)
    common, X_test = prepareTestFeatures()
    prediction = classifier.predict_proba(X_test)
    prediction = [float(x[1]) for x in prediction]
    predictionDf = pd.DataFrame(data={"prediction": prediction})
    pd.concat([common['bidder_id'], predictionDf], axis=1).to_csv(
        "submissions/{}.csv".format(name),
        index=False,
    )


def prepareTestFeatures():
    dfFeatures = pd.read_csv("../features/new_all_feat.csv")
    dfTest = pd.read_csv("../data/test.csv").drop(
        ['address', 'payment_account'],
        axis=1,
    )
    common = dfTest.merge(
        dfFeatures,
        on='bidder_id',
        how='left',
    ).replace(np.nan, 0)
    X_test = np.array(common.drop(['bidder_id'], axis=1))
    return common, X_test


def printSubmissionAverage(classifiers, X_train, y_train, name):
    for clf in classifiers:
        clf.fit(X_train, y_train)
    common, X_test = prepareTestFeatures()
    m = X_test.shape[0]
    totPr = np.zeros([m, 2])
    for clf in classifiers:
        pr = clf.predict_proba(X_test)
        totPr += pr
    prediction = [float(totPr[i][1]) / len(classifiers) for i in range(m)]
    predictionDf = pd.DataFrame(data={"prediction": prediction})
    pd.concat([common['bidder_id'], predictionDf], axis=1).to_csv(
        "submissions/{}.csv".format(name),
        index=False,
    )
