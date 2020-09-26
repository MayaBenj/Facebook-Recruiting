# Facebook-Recruiting
Facebook Recruiting IV: Human or Robot?

## Datasets:
Bidder dataset that includes a list of bidder information, including their id, payment account, and address. 
Bid dataset that includes 7.6 million bids on different auctions.

## Process Data
Main created features:
- x_unique - # unique times x shows per bidder
- bid_mean_x_unique - mean x_unique per  bids
- num_auction_won - # auctions user placed last bid
- first bid - # auctions user placed first bid
- time_diff_user_bids - mean diff between same user's bids per auction
- time_diff_prev_bid - mean diff between previous bid per auction

## Model Data
Best prediction was achieved using a random forest classifier. <br />
Accuracy: 0.9483 (+/- 0.02) <br />
Cost train: 0.030914, cost test: 0.038306 <br />
Train score: 0.969086, Test score: 0.961694
