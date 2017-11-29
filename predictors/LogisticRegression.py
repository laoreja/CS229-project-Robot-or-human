from sklearn.linear_model import LogisticRegression
from new_commons import *

name = "LR"

featureList = [
  'mean_bids_per_auction', 'url_entropy', 'tdiff_min', 'tdiff_median'
]

lr = LogisticRegression()
X_train, y_train = prepareTrainData(featureList)
evaluateClassifier(lr, X_train, y_train, name)
printSubmission(lr, X_train, y_train, name, featureList)
