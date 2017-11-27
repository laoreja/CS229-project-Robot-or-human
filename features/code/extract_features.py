#!/usr/bin/python
import pandas as pd
import numpy as np
from constants import *
from os.path import join

data_dir = '../../data'
feature_dir = '..'

bids = pd.read_csv(join(data_dir, 'bids.csv'))
bids_g_bidder = bids.groupby('bidder_id')
bids_g_auction = bids.groupby('auction')
bids_g_bidder_auction = bids.groupby(['bidder_id', 'auction'])

def generate_country_category_features_per_bidder(group):
    feature = dict()
    feature.update(dict.fromkeys(categories, 0))
    feature.update(dict.fromkeys(countries_list, 0))
    
    for country, value in group['country'].value_counts().iteritems():
        feature[str(country)] = value * 1.0 / group.shape[0]
    
    for cat in group['merchandise'].unique():
        feature[cat] = 1

    return pd.Series(feature)
    
country_category_feat_per_bidder = bids_g_bidder.apply(generate_country_category_features_per_bidder)
country_category_feat_per_bidder.to_csv(join(feature_dir, 'country_category_feat_per_bidder.csv'))
print country_category_feat_per_bidder.info()

    


