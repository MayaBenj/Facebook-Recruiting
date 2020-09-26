import pandas as pd
import numpy as np

"""
features:
x_unique - # unique times x shows per bidder
bid_mean_x_unique - mean x_unique per  bids

num_auction_won - # auctions user placed last bid
first bid - # auctions user placed first bid
time_diff_user_bids - mean diff between same user's bids per auction
time_diff_prev_bid - mean diff between previous bid per auction
"""

bids_data = pd.read_csv("../data_csv/bids.csv")

# Data per user
# Number of bids per user
bids_per_user = bids_data.groupby("bidder_id")["bid_id"].count()
bids_data_group = pd.DataFrame({'bidder_id': bids_per_user.index, 'bids': bids_per_user.values})
# Unique counter and mean for each column in bids
for element in bids_data.columns[2:].values:
    bids_column_unique = bids_data.groupby("bidder_id")[element].nunique()
    bids_by_column = pd.DataFrame({'bidder_id': bids_column_unique.index,
                                   element + "_unique": bids_column_unique.values})
    bids_data_group = pd.merge(bids_data_group, bids_by_column, on="bidder_id")
# Mean bids per unique counter
for element in bids_data_group.columns[2:].values:
    bids_data_group["bids_mean_" + element] = bids_data_group[element] / bids_data_group["bids"]

# Initialization of dataframe for data per auction
columns = ["num_auctions_won", "first_bid", "time_diff_prev_bid", "time_diff_user_bids"]
auction_data = pd.DataFrame(np.zeros((bids_per_user.index.shape[0], len(columns))), columns=columns)
auction_data.set_index(bids_per_user.index, inplace=True)
# Data per auction
for auction in bids_data["auction"].unique():
    data_per_auction = bids_data.loc[bids_data['auction'] == auction]
    # User who placed last bid (winner)
    auction_data.loc[data_per_auction.bidder_id.iloc[-1], ["num_auctions_won"]] += 1
    # User who placed first bid
    auction_data.loc[data_per_auction.bidder_id.iloc[0], ["first_bid"]] += 1
    # Mean time between same user's bids
    auction_data["time_diff_user_bids"] = auction_data["time_diff_user_bids"]\
        .add(data_per_auction.groupby("bidder_id")["time"]\
             .apply(lambda x: np.mean(np.diff(x)) if len(x) > 1 else 0)\
             .reindex(bids_per_user.index, fill_value=0))
    # Mean time between previous bid
    auction_data["time_diff_prev_bid"] = auction_data["time_diff_prev_bid"]\
        .add(data_per_auction.set_index("bidder_id")["time"]\
             .diff()[1:].groupby("bidder_id").mean()\
             .reindex(bids_per_user.index, fill_value=0))
# Divide auction data by # of auctions
auction_data_per_auction = auction_data.div(bids_data_group.set_index("bidder_id")[["auction_unique"]].values)

# Join dataframes
joined_data = bids_data_group.set_index("bidder_id").join(auction_data_per_auction)
# Split into outcome known and unknown
train_data = pd.read_csv("../data_csv/train.csv")
train_features = joined_data[joined_data.index.isin(train_data.bidder_id)]
train_features.insert(train_features.shape[1], "y", train_data.set_index("bidder_id")["outcome"])
test_data = pd.read_csv("../data_csv/test.csv")
test_features = joined_data[joined_data.index.isin(test_data.bidder_id)]
# Saved grouped df to csv
train_features.to_csv("../data_csv/train_with_features.csv")
test_features.to_csv("../data_csv/test_with_features.csv")
