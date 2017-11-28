import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from predictors.commons import prepareTrainData
from sklearn.model_selection import cross_val_score

name = "RF"

random.seed()

featureList = [
  'mean_bids_per_auction', 'tdiff_median', 'device_cnt', 'price_min',
  'price_std', 'response_min', 'ip_entropy', 'country_cnt',
  'bids_cnt', 'tdiff_min', 'tdiff_ip', 'auction_cnt',
  'url_entropy', 'tdiff_mean', 'country_cnt_mean_auc'
]

X_train, y_train = prepareTrainData(featureList)
all_n_estimators = range(5, 500, 10)
all_max_depth = range(3, 30)
all_max_feature = range(1, len(featureList) + 1)

paramsList = []
aucList = []
for i in range(30000):
    n_estimators = random.choice(all_n_estimators)
    max_depth = random.choice(all_max_depth)
    max_feature = random.choice(all_max_feature)
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_feature,
    )
    evaluation = cross_val_score(
        rf,
        X_train,
        y_train,
        cv=3,
        n_jobs=3,
        scoring="roc_auc",
    ).mean()
    print "\t({}, {}, {}): {}".format(
        n_estimators, max_depth, max_feature, evaluation
    )
    aucList.append(evaluation)
    paramsList.append({
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "max_features": max_feature,
    })
    idx = np.argmax(aucList)
    print "Best params (AUC={}) are {}".format(
        aucList[idx],
        paramsList[idx],
    )


