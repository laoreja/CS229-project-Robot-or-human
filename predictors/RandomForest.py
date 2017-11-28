from sklearn.ensemble import RandomForestClassifier
from commons import *

name = "RF"

rf = RandomForestClassifier(
    n_estimators=160,
    max_depth=8,
)

featureList = [
  'mean_bids_per_auction', 'tdiff_median', 'device_cnt', 'price_min', 
  'price_std', 'response_min', 'ip_entropy', 'country_cnt', 
  'bids_cnt', 'tdiff_min', 'tdiff_ip', 'auction_cnt', 
  'url_entropy', 'tdiff_mean', 'country_cnt_mean_auc'
]

X_train, y_train = prepareTrainData(featureList)
evaluateClassifier(rf, X_train, y_train, name)
printSubmission(rf, X_train, y_train, name, featureList)
