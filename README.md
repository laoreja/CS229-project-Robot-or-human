# Robot-or-human

## Features
(All features are merged in all_feat.csv, train: train_all_feat.csv, test: test_all_feat.csv)

- basic_feat_per_bidder.csv
- [x] # of bids
- [x] # of auctions
- [x] # of bids per auction
- [x] # of country
- [x] # of IP
- [x] # of URL
- [x] # of device
- [x] # of auctions for each merchandise category
- [x] whether in a certain country
- auctions_win.csv
- [x] # of auctions won
- time_response_feat.csv
- [x] time response (min, max, mean, std, median)
  - the time difference with previous bid
- tdiff_feat_per_bidder.csv
- [x] time series for user (min, max, mean, std, median, zeros)
  - operations of a certain user
- price_feat_per_bidder.csv
- [x] bid prices (min, max, mean, std, median)

~~# of merchandise (All human and robots are 1, except one human is 2)~~



## Models
- [x] Logistic Regression
- [x] SVM
- [x] Decision Tree
- [x] Random Forest
- [x] Gradient Boost Tree
- [x] GDA

### Cross-Validation Result (K=4)
#### use all features
Model  |Training AUC| CV Average AUC (K=4) | Features used
--|--|--|--
Logistic Regression|0.873173192722|0.808161240231|`new_all_feat.csv`
SVM(RBF)|1.0|0.887362701586|new_feat_for_dnn.csv
SVM(Linear)|0.998117588343|0.665470643041|new_feat_for_dnn.csv
Decision Tree|1.0|0.671056827935|`new_all_feat.csv`
Random Forest|0.99995660146|0.940371307088|`new_all_feat.csv`
Random Forest Adaboost|1.0|0.937282646135|`new_all_feat.csv`
Gradient Boost Tree|1.0|0.918035048245|`new_all_feat.csv`
GDA|0.752102116764|0.746763663678|`new_all_feat.csv`
DNN|0.965523|NA|`new_feat_for_dnn.csv`
DNN_sklearn|0.999875229199|0.838388269639|`new_feat_for_dnn.csv`
RFLR|0.993680087665|0.925524029599|`new_all_feat.csv`
GBTLR|0.953118727555|0.912516532117|`new_all_feat.csv`

#### use feature selection
Model  |Training AUC| CV Average AUC (K=4) | Features used
--|--|--|--
Logistic Regression|0.905434582126|0.905261880562|`new_all_feat.csv`
Logistic Regression|0.905434582126|0.0.845475487871|`new_all_feat.csv`
SVM(RBF)|0.998220659875|0.909697711975|`new_all_feat.csv`
SVM(Linear)|NA|NA|`new_feat_for_dnn.csv`
Decision Tree|0.867897557747|0.801296381624|`new_all_feat.csv`
Random Forest|0.999663661318|0.940295232529|`new_all_feat.csv`
Random Forest Adaboost|1.0|0.945450916452|`new_all_feat.csv`
Gradient Boost Tree|1.0|0.934069527262|`new_all_feat.csv`
GDA|0.849059879135|0.84798183378|`new_all_feat.csv`
DNN|NA (Python quit unexpectedly)|NA|`new_feat_for_dnn.csv`
RFLR|0.988754353416|0.932652492807|`new_all_feat.csv`
GBTLR|0.937573913138|0.91208076084|`new_all_feat.csv`

## Content
- [x] Feature selection
- [x] Ablative analysis on features
- [x] Model combination
- [ ] Deep Learning (optimization problems: data imbalance, overfitting)
- [x] Cross Validation to tune (hyper)parameters


### Feature Selection

Using `new_all_feat.csv` for now.

Plots are in `./img/`.

LR:
```
Feature sequence: [
  'mean_bids_per_auction', 'url_entropy', 'tdiff_min', 'tdiff_median',
  'response_median', 'response_mean', 'response_min', 'url_cnt',
  'price_min', 'response_std', 'country_cnt_mean_auc', 'tdiff_mean',
  'price_mean', 'auction_cnt', 'response_max', 'tdiff_std',
  'country_cnt', 'price_median', 'price_max', 'price_std',
  'tdiff_ip', 'ip_cnt', 'bids_cnt', 'ip_entropy',
  'tdiff_zeros', 'auctions_won_cnt', 'device_cnt', 'tdiff_max'
]
```
RF:
```
Feature sequence: [
  'mean_bids_per_auction', 'tdiff_median', 'device_cnt', 'price_min',
  'price_std', 'response_min', 'ip_entropy', 'country_cnt',
  'bids_cnt', 'tdiff_min', 'tdiff_ip', 'auction_cnt',
  'url_entropy', 'tdiff_mean', 'country_cnt_mean_auc', 'price_mean',
  'url_cnt', 'response_max', 'response_mean', 'ip_cnt',
  'tdiff_zeros', 'auctions_won_cnt', 'price_max', 'response_std',
  'tdiff_max', 'response_median', 'tdiff_std', 'price_median'
]
```

GBT:
```
Feature sequence: [
  'tdiff_mean', 'price_std', 'bids_cnt', 'tdiff_median',
  'response_std', 'tdiff_zeros', 'country_cnt', 'response_min',
  'tdiff_std', 'tdiff_ip', 'price_max', 'price_min',
  'tdiff_min', 'tdiff_max', 'response_max', 'ip_cnt',
  'response_median', 'auction_cnt', 'price_median', 'price_mean',
  'ip_entropy', 'response_mean', 'url_entropy', 'url_cnt',
  'country_cnt_mean_auc', 'auctions_won_cnt', 'device_cnt', 'mean_bids_per_auction'
]
```

### CV to tune (hyper-)parameters

After feature selection, we use CV to tune (hyper-)parameters for the random 
forest model. (`./cvChooseHyperParams.py`) 
The parameters are 
- `n_estimators`,
- `max_depth`, and
- `max_features`.

The best are `{'n_estimators': 315, 'max_features': 4, 'max_depth': 6}`.
And in this setting, the result of random forest model is
```
  Training AUC = 0.995768642385
  CV AUC (K=4) = 0.941281748317
  Test AUC = 0.92586.
```
And without this tuning, the test AUC of RF (with feature selection)
is 0.92298.
