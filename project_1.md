# Project 1
In this project, my objective is to predict the next 15-min and 30-min returns of the three stock index futures (both traded on China Financial Futures Exchange). IC represents the CSI 500 Index, IF represents the CSI 300 Index, and IH represents the SSE 50 Index. Both of them are major stock index futures contracts. In this project, I have only considered trading the main contracts, since liquidity is important for frequent position adjustments.

## Data preparation
For calculations of the prediction signal, I combined level-2 data of main contracts day by day for both underlyings. Sample data for a single trading day can be found in directory “sample_snapshot_data/”. The update frequency of Level-2 market data is once every 500 milliseconds. Level-2 data includes the following information: contract name, latest price, trading volume, open interest, bid price, ask price, bid volume, and ask volume.

Definitions of the main columns: ap1, ap2, ..., ap5 representing ask_price1, ask_price2, ..., ask_price5. bp1, bp2, ..., bp5 representing bid_price1, bid_price2, ..., bid_price5.

## Idea generation
The momentum effect in stock index futures has been widely discussed. I have met some quantitative traders achieved stable returns through intraday trading strategies incorperating momentum effect. The success of these strategies heavily relies on rapid response to market dynamics. That is why I would like to find a signal which target on short-term price fluctuations. As the prediction signal probably changes rapidly, the corresponding strategy would be a high-turnover strategy.

I look into the high-frequency data to observe the characteristics of the limit order book during the formation of price trends. And I found that when the mid-price is stable, (ask_price5 - ask_price1) and (bid_price1 - bid_price5) tend to stabilize within a relatively small range. This is probably because investors do not have significant disagreements on the future price movements. However, when the mid-price begins to show an upward trend, the spread between ask_price1 and ask_price5 tends to increase. Similarly, as the mid-price begins to show a downward trend, the spread between bid_price1 and bid_price5 tends to increase. It is reasonable to pay more attention to the price momentum when (ask_price5 - ask_price1) or (bid_price1 - bid_price5) increases.

The following two figures are examples:
![upward trend](./images/picture1.png)
![downward trend](./images/picture2.png)

(ask_price5 - ask_price1) or (bid_price1 - bid_price5) might not has a strong predictive power, while we can combine them with another basic predictor. In this case, I use an improved version of the RSI indicator, which is a momentum factor suitable for predicting short-term trends. 

## Expression of the price prediction signal

Define an improved version of the RSI indicator as:

$$
rs(price, N) = ema(ReLU(diff(price,1)), N) / ema(abs(diff(price,1)), N)
$$

The operator ema() is for "Exponential Moving Average", and diff() is equivalent to numpy.diff. N is the window size for rolling calculation. The operator rs() returns a value lies between \[0,1\].

The complete calculation process is as follows:

### step 1: calculations for every snapshots (updated every 500 milliseconds)

spread_avg = ts_avg(ask_price1 - bid_price1, 600), 

bid_range_avg = ts_avg((bid_price1 - bid_price5)/spread_avg, 120),

ask_range_avg = ts_avg((ask_price5 - ask_price1)/spread_avg, 120),

bid_rs = rs(bp1) - 0.5,

ask_rs = rs(ap1) - 0.5,

raw_alpha = (ask_rs * bid_range_avg) * (ask_rs < 0) + (bid_rs * ask_range_avg) * (bid_rs > 0)

The **raw_alpha** consists of two parts, which predict the upward trend and the downward trend, respectively. 

### step 2: Resample raw_alpha by taking the last value in every 30-second interval

Resample **raw_alpha** so that it is updated every 30 seconds (09:30:30, 09:31:00, 09:31:30, ..., 14:59:00, 14:59:30, 15:00:00).

*FinalSignal = ts_ema(ts_ema(raw_alpha,60),20)*

The complete calculations are in "calc_signal.py".

## Back testing results
The code related to backtesting is in the file "backtesting.py".

In this part, normalization processing would be applied to the **FinalSignal** (see the function scale_normal() in "backtesting.py"). After the normalization processing, we have one **normalized_signal_dataframe** and one **position_weight_dataframe**.

The **normalized_signal_dataframe** is used to calculate the IC table (Information Coefficient). In this case, "Information Coefficient" is the time-series Pearson correlation between **normalized_signal** and future returns of specified intervals. Note that all interval returns are shifted by 1 minute before calculating the Pearson correlation, assuming that I would need to adjust positions weight of the portfolio during (t, t+1min).

The **position_weight_dataframe** is used to calculate the performance table. **position_weight_dataframe** is derived from **normalized_signal_dataframe**, representing the target position in the next 30 seconds. With target positions we can simulate the trading performance of this strategy.

The IC table and performance table:
![backtesting](./images/backtest.png)

### IC table description:
The three columns on the right side of the table represent the Information Coefficient calculated from corresponding interval return (Both ICs are multiplied by 100). For example, "0min_15min_ic" represents the Pearson correlation between **signal at time t** with **future return calculated for (t+1min, t+16min)**.
According to the IC table, the predictive power of this signal is stable for the next 15-min and 30-min returns, with IC around 2.3% and 1.17%, respectively.






