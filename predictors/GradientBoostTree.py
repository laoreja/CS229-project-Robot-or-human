from sklearn.ensemble import GradientBoostingClassifier
from commons import *

name = "GBT"
gbt = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=2,
)
featureList = [
    'tdiff_mean', 'price_std', 'bids_cnt', 'tdiff_median',
    'response_std', 'tdiff_zeros', 'country_cnt', 'response_min',
    'tdiff_std'
]
X_train, y_train = prepareTrainData(featureList)
evaluateClassifier(gbt, X_train, y_train, name)
printSubmission(gbt, X_train, y_train, name, featureList)
