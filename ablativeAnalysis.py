from __future__ import print_function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from predictors.commons import excludeBidders


LOG_FOUT = open('ablative_analysis.txt', 'w')

def log_print(output):
    print(output, file=LOG_FOUT)
    print(output)

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


def ablativeAnalysis(clf, clfName):
    log_print("Ablative Analysis for {}".format(clfName))
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
    
#    features = ['auctions_won_cnt', 'price_max', 'price_mean', 'price_median', 'price_min', 'price_std', 'tdiff_max', 'tdiff_mean', 'tdiff_median', 'tdiff_min', 'tdiff_std', 'tdiff_zeros', 'response_max', 'response_mean', 'response_median', 'response_min', 'response_std', 'auction_cnt', 'bids_cnt', 'country_cnt', 'device_cnt', 'ip_cnt', 'mean_bids_per_auction', 'url_cnt', 'tdiff_ip', 'ip_entropy', 'url_entropy', 'country_cnt_mean_auc']
    feature_sets = [
    ['price_max', 'price_mean', 'price_median', 'price_min', 'price_std'], 
    ['tdiff_max', 'tdiff_mean', 'tdiff_median', 'tdiff_min', 'tdiff_std', 'tdiff_zeros'], 
    ['response_max', 'response_mean', 'response_median', 'response_min', 'response_std'], 
    ['ip_entropy', 'url_entropy']]
    
    features = list(dfX.columns) + feature_sets
    
    AUC = evaluateClassifier(
        clf,
        np.array(dfX),
        y_train,
    )
    log_print('Full feature sets \t' + str(AUC)) 
    
    AUC_list = []
    for idx, features_to_drop in enumerate(features):
        AUC = evaluateClassifier(
                clf,
                np.array(dfX.drop(features_to_drop, axis=1)),
                y_train,
        )
        AUC_list.append(AUC)
        
    AUC_list = np.array(AUC_list)
    sorted_idx = np.argsort(AUC_list)
    
    for idx in sorted_idx:
        features_to_drop = features[idx]
        log_print( (features_to_drop if isinstance(features_to_drop, str) else ' '.join(features_to_drop)) +' \t' + str(AUC_list[idx]))
    log_print('')

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


for name, clf in classifiers.iteritems():
    ablativeAnalysis(clf, name)
    

LOG_FOUT.close()
