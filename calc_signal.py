import pandas as pd
import numpy as np

def ts_rs(mat, n):
    diff = np.diff(mat, prepend=[0])
    diff[np.isnan(diff)] = 0 
    diff[np.isinf(diff)] = 0
    diff_up = np.maximum(0, diff)
    diff_abs = np.abs(diff)
    res_mat = pd.Series(diff_up).ewm(span=n).mean() / pd.Series(diff_abs).ewm(span=n).mean()
    res_mat.loc[:n] = 0.5
    res_mat[np.isnan(res_mat)] = 0.5
    res_mat[np.isinf(res_mat)] = 0.5
    return res_mat.values

def ts_ema(mat, n):
    res_mat = pd.Series(mat).ewm(span=n).mean()
    return res_mat.values

def ts_avg(mat, n):
    res_mat = pd.Series(mat).rolling(n, min_periods=1).mean()
    return res_mat.values

def ts_delta(mat, n):
    res_data = pd.Series(mat).diff(n)
    res_data[np.isnan(res_data)] = 0
    res_data[np.isinf(res_data)] = 0
    return res_data


if __name__ == '__main__':
    underlyings = ['IC','IF','IH']
    DATA_PATH = '/server/data/chenhaolin/tmp_data'

    close_30s_data = pd.read_pickle(DATA_PATH + '/index_close_2021_2023.pkl')
    print(close_30s_data.shape)
    print(close_30s_data.head())
    signal_index = close_30s_data.index

    signal_list = []
    for underlying in underlyings:
        file_path = DATA_PATH + f'/{underlying}_main_2021_2023.pkl'
        df_tick = pd.read_pickle(file_path)
        df_raw_alpha = pd.DataFrame(index=df_tick.index, columns=[underlying])
        print(df_tick.shape)
        # calculation process for the prediction signal
        spread_avg = ts_avg(df_tick['ap1']-df_tick['bp1'], 600)
        bid_range = np.minimum((df_tick['bp1']-df_tick['bp5'])/spread_avg, 8)
        ask_range = np.minimum((df_tick['ap5']-df_tick['ap1'])/spread_avg, 8)
        bid_range_avg = ts_avg(bid_range, 120)
        ask_range_avg = ts_avg(ask_range, 120)
        bid_rs = (ts_rs(df_tick['bp1'], 120) - 0.5)
        ask_rs = (ts_rs(df_tick['ap1'], 120) - 0.5)
        raw_alpha = np.where(ask_rs < 0, ask_rs*bid_range_avg, 0) + np.where(bid_rs > 0, bid_rs*ask_range_avg, 0)
        df_raw_alpha[underlying] = raw_alpha
        # resample raw alpha to be updated every 30 seconds
        resample_alpha = df_raw_alpha.iloc[pd.Series(df_raw_alpha.index).searchsorted(value=signal_index, side='right') - 1]
        signal_array = ts_ema(ts_ema(resample_alpha[underlying], 60), 20)
        signal_list.append(signal_array)

    final_signal = pd.DataFrame(index=signal_index, columns=underlyings)
    for idx in range(len(underlyings)):
        underlying = underlyings[idx]
        final_signal[underlying] = signal_list[idx]
    final_signal.to_pickle(DATA_PATH + '/raw_signal.pkl')

