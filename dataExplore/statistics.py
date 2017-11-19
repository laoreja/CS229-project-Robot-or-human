#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

dfBids = pd.read_csv("./data/bids.csv", low_memory=False)
dfTrain = pd.read_csv("./data/train.csv")
dfTest = pd.read_csv("./data/test.csv")
dfFeatures = pd.read_csv("./features/all_feat.csv")

dfLabels = dfTrain.drop(
    ['address', 'payment_account'],
    axis=1
)
dfTrainData = dfFeatures.merge(dfLabels, on='bidder_id')


def main():
    bidsPerAuction = dfTrainData[
        ['mean_bids_per_auction', 'outcome']
    ]
    print "# of bids per auction:\n{}".format(
        bidsPerAuction.groupby('outcome').mean()
    )

    bidSpeed = dfTrainData[[
        "outcome",
        "tdiff_mean",
        "tdiff_median",
        "tdiff_min",
        "tdiff_max",
        "tdiff_std",
        "tdiff_zeros",
        "response_mean",
        "response_median",
        "response_min",
        "response_max",
        "response_std",
    ]].groupby("outcome").mean()
    print "Action speed:\n{}".format(bidSpeed)

    bidsCnt = dfTrainData[
        ['bids_cnt', 'outcome']
    ]
    print "# of bids:\n{}".format(
        bidsCnt.groupby('outcome').mean()
    )
    robotBidsCnt = np.array(bidsCnt.loc[bidsCnt['outcome'] == 1])
    cntDict = {}
    for row in robotBidsCnt:
        if int(row[0]) not in cntDict:
            cntDict[int(row[0])] = 1
        else:
            cntDict[int(row[0])] += 1
    maxBids = max(cntDict.keys())
    maxIdx = int(math.log(maxBids) / math.log(2)) + 1
    starts = [2 ** idx for idx in range(maxIdx)]
    width = [2 ** idx for idx in range(maxIdx)]
    ys = [0] * maxIdx
    for x in cntDict:
        y = cntDict[x]
        idx = 0
        while x >= starts[idx] + width[idx]:
            idx += 1
        ys[idx] += y
    plt.clf()
    plt.bar(range(maxIdx), ys)
    plt.xlabel('# of bids')
    plt.ylabel('# of robots')
    plt.xticks(
        range(maxIdx),
        ["{}".format(2 ** i) for i in range(maxIdx)],
    )
    plt.tick_params(axis='x', labelsize=6)
    plt.title('Distribution of bidding numbers of robots')
    plt.savefig('img/bidding-numbers-robots.eps')


main()
