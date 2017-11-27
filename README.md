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
Model  | CV Average AUC (K=4) | Features used
--|--|--
Logistic Regression|0.829539404964|
SVM(RBF)|0.887362701586|new_feat_for_dnn.csv
SVM(Linear)|0.665470643041|new_feat_for_dnn.csv
Decision Tree|0.648740361594|
Random Forest|0.927422406274|
Gradient Boost Tree|0.898488698003|
GDA|0.746763663678|new_all_feat.csv
DNN|0.861422|new_feat_for_dnn.csv


## Content
- [x] Feature selection
- [x] Ablative analysis on features
- [ ] Model combination
- [ ] Deep Learning (optimization problems: data imbalance, overfitting)
- [ ] Cross Validation to tune (hyper)parameters

### Feature Selection

Using `new_all_feat.csv` for now.

![alt text](./img/feature-selection-summary.eps "Feature Selection")

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

