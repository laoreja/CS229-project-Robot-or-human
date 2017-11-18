# Robot-or-human

## Features
(All features are merged in all_feat.csv)

- basic_feat_per_bidder.csv
- [ ] # of bids
- [ ] # of auctions
- [ ] # of bids per auction
- [ ] # of country
- [ ] # of IP
- [ ] # of URL
- [ ] # of device
- [ ] # of merchandise
- [ ] # of auctions for each merchandise category
- [ ] whether in a certain country
- auctions_win.csv
- [ ] # of auctions won
- time_response_feat.csv
- [ ] time response (min, max, mean, std, median)
  - the time difference with previous bid
- tdiff_feat_per_bidder.csv
- [ ] time series for user (min, max, mean, std, median, zeros)
  - operations of a certain user
- price_feat_per_bidder.csv
- [ ] bid prices (min, max, mean, std, median)



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
Random Forest|0.915233858385
Gradient Boost Tree|0.88609320803
GDA|TODO


## Content
- [ ] Features selection
- [ ] Error analysis
- [ ] Model combination
- [ ] Deep Learning
