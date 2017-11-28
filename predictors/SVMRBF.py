from sklearn import svm
from commons import *

name = "SVMRBF"

featureList = [
    'bids_cnt', 'mean_bids_per_auction', 'tdiff_max', 'country_cnt_mean_auc', 
    'tdiff_std', 'price_std', 'response_median', 'tdiff_mean', 'response_min', 
    'tdiff_ip'
]
svc = svm.SVC(kernel='rbf', probability=True)
X_train, y_train = prepareTrainData(featureList)
evaluateClassifier(svc, X_train, y_train, name)
printSubmission(svc, X_train, y_train, name, featureList)
