import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from predictors.commons import excludeBidders
from predictors.GaussianDiscriminantAnalysis import GDAEstimator
from predictors.RFLR import RFLREstimator
from predictors.GBTLR import GBTLREstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm

feature_table_filename = "./features/new_all_feat.csv"


def evaluateClassifier(classifier, X_train, y_train):
    return cross_val_score(
        classifier,
        X_train,
        y_train,
        cv=4,
        n_jobs=3,
        scoring="roc_auc",
    ).mean()


def featureSelection(clf, clfName):
    print "Feature Selection for {}".format(clfName)
    dfFeatures = pd.read_csv(feature_table_filename)
    for b in excludeBidders:
        dfFeatures = dfFeatures[dfFeatures.bidder_id != b]
    if "Unnamed: 0" in list(dfFeatures):
        dfFeatures = dfFeatures.drop(["Unnamed: 0"], axis=1)
    dfLabels = pd.read_csv("./data/train.csv").drop(
        ['address', 'payment_account'],
        axis=1,
    )
    common = dfFeatures.merge(dfLabels, on='bidder_id')
    y_train = np.ravel(common[['outcome']])
    dfX = common.drop(['bidder_id', 'outcome'], axis=1)
    lstFeatures = list(dfX)
    candidateFeatures = set(lstFeatures)

    numFeatures = 0
    selectedFeatures = []
    axisNumFeatures = []
    axisAvgAUC = []
    while len(candidateFeatures) > 0:
        numFeatures += 1
        candidates = list(candidateFeatures)
        AUCs = [
            evaluateClassifier(
                clf,
                np.array(common[selectedFeatures + [f]]),
                y_train,
            )
            for f in candidates
        ]
        idx = np.argmax(AUCs)
        print "# {}:\n\t{}\n\t{}\n---Choosing {}, {}".format(
            numFeatures,
            candidates,
            AUCs,
            candidates[idx],
            AUCs[idx],
        )

        axisNumFeatures.append(numFeatures)
        axisAvgAUC.append(AUCs[idx])
        selectedFeatures.append(candidates[idx])
        candidateFeatures -= set([candidates[idx]])

    plt.clf()
    plt.plot(axisNumFeatures, axisAvgAUC, 'x-b')
    plt.xlabel("# of Features")
    plt.ylabel("Average AUC (K=4)")
    plt.title("Feature Selection for {}".format(clfName))
    plt.savefig("./img/feature-selection-{}.eps".format(
        clfName.lower().replace(" ", "-")
    ))
    print "Feature Selection for {}".format(clfName)
    print "Feature sequence: {}".format(selectedFeatures)
    print "AUC sequence: {}".format(axisAvgAUC)
    return selectedFeatures, axisAvgAUC


def main():
    rf = RandomForestClassifier(
        n_estimators=160,
        max_depth=8,
    )
    gbt = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=2,
    )
    lr = LogisticRegression()

    # comment some classifiers if you want to skip them
    classifiers = {
        "LR": lr,
        "RF": rf,
        "GBT": gbt,
    }

    results = {
            name: featureSelection(
                classifiers[name],
                name,
            )
            for name in classifiers
        }

    print results
    plt.clf()
    xs = range(1, len(results.values()[0][0]) + 1)
    patterns = ['x-b', 'x-r', 'x-g', 'x-c', 'd-b', 'd-r', 'd-g', 'd-c']
    idx = 0
    for name in results:
        plt.plot(xs, results[name][1], patterns[idx], label=name)
        idx += 1
    plt.xlabel("# of features")
    plt.ylabel("Average AUC (K=4)")
    plt.title("Feature Selection (Summary)")
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.1),
        ncol=3,
        fancybox=True,
    )
    plt.savefig("img/feature-selection-summary.eps")


# main()
# gda = GDAEstimator()
# dt = DecisionTreeClassifier()
# rflr = RFLREstimator()
# gbtlr = GBTLREstimator()
# svc = svm.SVC(kernel='rbf', probability=True)
svc = svm.SVC(kernel='linear', cache_size=7000, probability=True)
# rf = RandomForestClassifier(
#     n_estimators=160,
#     max_depth=10,
# )
# rf = AdaBoostClassifier(base_estimator=rf, n_estimators=25)
# featureSelection(dt, "DT")
# featureSelection(rflr, "RFLR")
# featureSelection(gbtlr, "GBTLR")
# featureSelection(svc, "SVMRBF")
featureSelection(svc, "SVMLinear")
# featureSelection(rf, "RFAda")
