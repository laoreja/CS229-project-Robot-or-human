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
- [ ] GDA

### Cross-Validation Result (K=4)
Model  | Average AUC
--|--
Logistic Regression|0.829539404964
SVM(RBF)|0.5
SVM(Linear)|TODO
Decision Tree|0.648740361594
Random Forest|0.927422406274
Gradient Boost Tree|0.898488698003
GDA| NA


## Content
- [ ] Features selection
- [ ] Error analysis
- [ ] Model combination
- [ ] Deep Learning
