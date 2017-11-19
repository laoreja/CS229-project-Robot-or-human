import pandas as pd
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp


def filterFeatures(dfFeatures):
    lst = [x for x in list(dfFeatures) if len(x) > 2]
    return dfFeatures[lst]


def prepareTrainData():
    dfFeatures = pd.read_csv("./features/all_feat.csv")
    dfLabels = pd.read_csv("./data/train.csv").drop(
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
    dfFeatures = pd.read_csv("./features/all_feat.csv")
    dfTest = pd.read_csv("./data/test.csv").drop(
        ['address', 'payment_account'],
        axis=1,
    )
    common = dfFeatures.merge(
        dfTest,
        on='bidder_id',
        how='right',
    ).replace(np.nan, 0)
    X_test = np.array(common.drop(['bidder_id'], axis=1))
    prediction = classifier.predict(X_test)
    predictionDf = pd.DataFrame(data={"prediction": prediction})
    pd.concat([common['bidder_id'], predictionDf], axis=1).to_csv(
        "submissions/{}.csv".format(name),
        index=False,
    )


def printSubmissionAverage(classifiers, X_train, y_train, name):
    for clf in classifiers:
        clf.fit(X_train, y_train)
    dfFeatures = pd.read_csv("./features/all_feat.csv")
    dfTest = pd.read_csv("./data/test.csv").drop(
        ['address', 'payment_account'],
        axis=1,
    )
    common = dfFeatures.merge(
        dfTest,
        on='bidder_id',
        how='right',
    ).replace(np.nan, 0)
    X_test = np.array(common.drop(['bidder_id'], axis=1))
    m = X_test.shape[0]
    totPr = np.zeros([m, 2])
    for clf in classifiers:
        pr = clf.predict_proba(X_test)
        totPr += pr
    prediction = [1. if totPr[i][0] < totPr[i][1] else 0. for i in range(m)]
    predictionDf = pd.DataFrame(data={"prediction": prediction})
    pd.concat([common['bidder_id'], predictionDf], axis=1).to_csv(
        "submissions/{}.csv".format(name),
        index=False,
    )
