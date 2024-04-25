import pandas as pd
import numpy as np


def calc_interval_return(raw_price_data, delay_num, interval, section_num=1):
    '''Calculate the corresponding interval return
    :raw_price_data: dataframe of price data, updated every 30 seconds
    :delay_num: the number of bars to be shifted before calculating returns
    :interval: window size for future returns
    :section_num: the number of intervals
    '''
    df_interval_return = (raw_price_data.shift(-(1 + delay_num + interval*section_num)) / raw_price_data.shift(-(1 + delay_num + interval*(section_num-1))) - 1)
    return df_interval_return


def scale_normal(raw_factor_data, window_1, window_2, num_time_cut, cut_off):
    ''' Normalize the raw signal values and convert them into position weights
    :raw_factor_data: raw signal values (DataFrame)
    :window_1: window size for normalizing a single-asset time series data
    :window_2: window size for cross-sectional normalization
    :num_time_cut: the maximum number of samples in a single trading day
    :cut_off: cut off applied to the position weights
    '''
    norm_factor = pd.DataFrame(index=raw_factor_data.index, columns=raw_factor_data.columns)
    for s in raw_factor_data.columns:
        raw_factor_s = raw_factor_data[s].dropna()
        ts_max_values = raw_factor_s.abs().rolling(window_1, min_periods=int(0.2*window_1)).max(skipna=True)
        ts_max_ser = ts_max_values.reindex(raw_factor_data.index, method='ffill')
        norm_factor[s] = raw_factor_data[s] / ts_max_ser
    norm_factor.iloc[:window_1, :] = np.nan
    norm_factor.dropna(how='all', inplace=True)
    ts_norm_factor = norm_factor.iloc[window_2:, :].astype(np.float64)
    ts_adj_data = norm_factor.fillna(method='ffill', limit=num_time_cut)
    cs_abs_sum = ts_adj_data.abs().sum(axis=1,skipna=True)
    cs_avg = cs_abs_sum.rolling(window_2, min_periods=int(0.2*window_2)).mean(skipna=True)
    cs_avg.iloc[0:window_2] = np.nan
    ts_adj_data.loc[:,:] = (ts_adj_data.values.T / cs_avg.values).T
    ts_adj_data.dropna(how='all', inplace=True)
    non_trading_time_mask = raw_factor_data.loc[raw_factor_data.index.isin(ts_adj_data.index)].isna()
    norm_signal = ts_adj_data.mask(non_trading_time_mask, np.nan)
    norm_signal.fillna(method='ffill', limit=num_time_cut, inplace=True)
    signal_values = norm_signal.values
    norm_signal.loc[:,:] = np.clip(signal_values, -cut_off, cut_off)
    return norm_signal.iloc[num_time_cut:,:], ts_norm_factor.iloc[num_time_cut:,:]


def calc_ic_table(scaled_signal, raw_price_data, delay_num, interval, section_num):
    '''Use normalized signal to calculate IC table (Information Coefficient)
    :scaled_signal: dataframe of normalized signal, updated every 30 seconds
    :raw_price_data: dataframe of price data, updated every 30 seconds
    :delay_num: the number of bars to be shifted before calculating returns
    :interval: window size for future returns
    :section_num: the number of intervals
    '''
    # prepare trading siganl
    scaled_signal['trading_date'] = raw_price_data.loc[raw_price_data.index.isin(scaled_signal.index), 'trading_date']
    scaled_signal['year'] = np.floor(scaled_signal['trading_date'] / 10000).astype(int)
    year_list = np.unique(scaled_signal['year'])
    start_dates = []
    end_dates = []
    # calculate IC for all samples
    all_sample_ic = []
    signal_array = scaled_signal[underlyings].values
    for section in range(1, section_num+1):
        df_return = calc_interval_return(raw_price_data, delay_num, interval, section) # 用bar数据的price计算出对应的Interval收益率
        return_array = df_return.loc[df_return.index.isin(scaled_signal.index), underlyings].values
        sig_flat_array = signal_array.flatten()
        return_flat_array = return_array.flatten()
        invalid_mask = (np.isnan(sig_flat_array)|np.isnan(return_flat_array))
        section_ic = np.corrcoef(sig_flat_array[~invalid_mask], return_flat_array[~invalid_mask])[0][1]
        all_sample_ic.append(section_ic)
    # calculate IC for each year
    ic_by_year = []
    for year in year_list:
        temp_ic_list = []
        signal_array = scaled_signal.loc[scaled_signal['year']==year, underlyings].values
        start = scaled_signal.loc[scaled_signal['year']==year,'trading_date'].values[0]
        end = scaled_signal.loc[scaled_signal['year']==year,'trading_date'].values[-1]
        start_dates.append(start)
        end_dates.append(end)
        for section in range(1, section_num+1):
            price_data_year = raw_price_data[(raw_price_data['trading_date'] >= start) & (raw_price_data['trading_date'] <= end)]
            df_return_year = calc_interval_return(price_data_year, delay_num, interval, section)
            return_array = df_return_year.loc[df_return_year.index.isin(scaled_signal.index),underlyings].values
            sig_flat_array = signal_array.flatten()
            return_flat_array = return_array.flatten()
            invalid_mask = (np.isnan(sig_flat_array)|np.isnan(return_flat_array))
            section_ic = np.corrcoef(sig_flat_array[~invalid_mask], return_flat_array[~invalid_mask])[0][1]
            temp_ic_list.append(section_ic)
        ic_by_year.append(temp_ic_list)
    # summary
    start_dates.append(scaled_signal['trading_date'].values[0])
    end_dates.append(scaled_signal['trading_date'].values[-1])
    num_1min = int(interval*0.5)
    ic_cols = [f'{(i-1)*num_1min}min_{(i)*num_1min}min_ic' for i in range(1, section_num+1)]
    ic_table = pd.DataFrame(index=[str(x) for x in year_list] + ['all'], columns=['start', 'end'] + ic_cols)
    ic_table.loc[:,ic_cols] = np.round(np.array(ic_by_year + [all_sample_ic])*100, 2) # 所有IC乘100
    ic_table['start'] = start_dates
    ic_table['end'] = end_dates
    return ic_table



def calc_performance_table(scaled_signal, raw_price_data, delay_num):
    ''' PnL performance
    :scaled_signal: dataframe of position weight, transformed from normalized signal
    :raw_price_data: dataframe of price data, updated every 30 seconds
    :delay_num: the number of bars to be shifted before calculating returns
    '''
    scaled_signal['trading_date'] = raw_price_data.loc[raw_price_data.index.isin(scaled_signal.index), 'trading_date']
    scaled_signal['year'] = np.floor(scaled_signal['trading_date'] / 10000).astype(int)
    year_list = np.unique(scaled_signal['year'])
    valid_date_list = np.unique(scaled_signal['trading_date'])
    price_data = raw_price_data.fillna(method='bfill', limit=MAX_NUM_TIME_CUT)
    df_return = pd.DataFrame(index=price_data.index, columns=underlyings)
    df_return = (price_data.shift(-(2+delay_num)) / price_data.shift(-(1+delay_num)) - 1)
    df_return['year'] = np.floor(price_data['trading_date'] / 10000).astype(int)
    start_dates = []
    end_dates = []
    long = []
    short = []
    tvr = []
    pnl = []
    mg_bp = []
    sharpe = []
    dd = []
    dd_start_list = []
    dd_end_list = []
    # all sample performance
    signal_array = np.nan_to_num(scaled_signal[underlyings])
    return_array = np.nan_to_num(df_return.loc[df_return.index.isin(scaled_signal.index), underlyings])
    pnl_array = (signal_array*return_array)
    df_pnl = pd.DataFrame(index=scaled_signal.index, columns=['year','trading_date','pnl_per_cut'])
    df_pnl['year'] = scaled_signal['year']
    df_pnl['trading_date'] = scaled_signal['trading_date']
    df_pnl['pnl_per_cut'] = np.sum(pnl_array, axis=1) 
    daily_pnl = df_pnl.groupby('trading_date')['pnl_per_cut'].apply(np.sum)
    tvr_array = np.sum(np.fabs(np.diff(signal_array, axis=0, prepend=signal_array[0:1,:])), axis=1)
    annualized_pnl = np.sum(daily_pnl) / (len(daily_pnl) / 250) 
    sharpe_all = np.sqrt(250) * np.mean(daily_pnl) / np.std(daily_pnl) 
    long_avg_all = np.mean(np.sum(signal_array * (signal_array > 0), axis=1)) 
    short_avg_all = np.mean(np.sum(signal_array * (signal_array < 0), axis=1)) 
    tvr_avg_all = np.sum(tvr_array) / len(valid_date_list) 
    mg_bp_all = np.sum(df_pnl['pnl_per_cut']) / np.sum(tvr_array) 
    cum_pnl = daily_pnl.cumsum()
    dd_array_all = (cum_pnl - cum_pnl.cummax())
    dd_all = dd_array_all.min() 
    dd_end = dd_array_all.index[dd_array_all.argmin()] 
    dd_start = cum_pnl.index[cum_pnl[:dd_array_all.argmin()+1].argmax()] 
    # performance of each year
    for year in year_list:
        signal_array = np.nan_to_num(scaled_signal.loc[scaled_signal['year']==year, underlyings])
        start = scaled_signal.loc[scaled_signal['year']==year,'trading_date'].values[0]
        end = scaled_signal.loc[scaled_signal['year']==year,'trading_date'].values[-1]
        df_pnl_year = df_pnl[df_pnl['year']==year].copy()
        num_days = len(np.unique(df_pnl_year['trading_date']))
        daily_pnl_year = df_pnl_year.groupby('trading_date')['pnl_per_cut'].apply(np.sum)
        tvr_array_year = np.sum(np.fabs(np.diff(signal_array, axis=0, prepend=signal_array[0:1,:])), axis=1)
        year_pnl = np.sum(daily_pnl_year) / (len(daily_pnl_year) / 250) 
        year_sharpe = np.sqrt(num_days) * np.mean(daily_pnl_year) / np.std(daily_pnl_year) 
        long_avg_year = np.mean(np.sum(signal_array * (signal_array > 0), axis=1)) 
        short_avg_year = np.mean(np.sum(signal_array * (signal_array < 0), axis=1)) 
        tvr_avg_year = np.sum(tvr_array_year) / num_days 
        mg_bp_year = np.sum(df_pnl_year['pnl_per_cut']) / np.sum(tvr_array_year) 
        cum_pnl_year = daily_pnl_year.cumsum()
        dd_array_year = (cum_pnl_year - cum_pnl_year.cummax())
        dd_year = dd_array_year.min()
        dd_end_year = dd_array_year.index[dd_array_year.argmin()]
        dd_start_year = cum_pnl_year.index[cum_pnl_year[:dd_array_year.argmin()+1].argmax()]
        # 保存分年stats
        start_dates.append(start)
        end_dates.append(end)
        long.append(long_avg_year)
        short.append(short_avg_year)
        tvr.append(tvr_avg_year)
        sharpe.append(year_sharpe)
        pnl.append(year_pnl)
        mg_bp.append(mg_bp_year)
        dd.append(dd_year)
        dd_start_list.append(dd_start_year)
        dd_end_list.append(dd_end_year)
    start_dates.append(scaled_signal['trading_date'].values[0])
    end_dates.append(scaled_signal['trading_date'].values[-1])
    long.append(long_avg_all)
    short.append(short_avg_all)
    tvr.append(tvr_avg_all)
    sharpe.append(sharpe_all)
    pnl.append(annualized_pnl)
    mg_bp.append(mg_bp_all)
    dd.append(dd_all)
    dd_start_list.append(dd_start)
    dd_end_list.append(dd_end)
    performance_cols = ['start','end','long','short','pnl','tvr','sharpe','dd','dd_start','dd_end','mg_bp']
    performance_table = pd.DataFrame(index=[str(x) for x in year_list]+['all'], columns=performance_cols)
    performance_table['start'] = start_dates
    performance_table['end'] = end_dates
    performance_table['long'] = np.round(long,3)
    performance_table['short'] = np.round(short,3)
    performance_table['pnl'] = np.round(np.array(pnl) * 100, 2) 
    performance_table['tvr'] = np.round(np.array(tvr) * 100, 2) 
    performance_table['sharpe'] = np.round(sharpe, 2)
    performance_table['dd'] = np.round(np.array(dd) * 100, 2) 
    performance_table['dd_start'] = dd_start_list
    performance_table['dd_end'] = dd_end_list
    performance_table['mg_bp'] = np.round(np.array(mg_bp) * 10000, 2) 
    return performance_table



if __name__ == '__main__':
    underlyings = ['IC','IF','IH']
    DATA_PATH = '/server/data/chenhaolin/tmp_data'
    
    MAX_NUM_TIME_CUT = 481 # the maximum number of samples in a single trading day
    DELAY_NUM = 1 # the number of bars to be shifted before calculating returns
    INTERVAL = 30
    SECTION = 3
    WINDOW_SIZE1 = 8000 # window size for normalizing a single-asset time series data
    WINDOW_SIZE2 = 20000 # window size for cross-sectional normalization
    CUT_OFF = 0.75 # cut off applied to the position weights

    close_30s_data = pd.read_pickle(DATA_PATH + '/index_close_2021_2023.pkl')
    raw_signal = pd.read_pickle(DATA_PATH + '/raw_signal.pkl')

    df_pos_weight, df_norm_signal = scale_normal(raw_signal, WINDOW_SIZE1, WINDOW_SIZE2, MAX_NUM_TIME_CUT, cut_off=CUT_OFF)

    # calculate IC table
    print(f'===================================== print alpha IC table =========================================')
    ic_table = calc_ic_table(df_norm_signal, close_30s_data, DELAY_NUM, INTERVAL, SECTION)
    print(ic_table)

    # calculate performance table
    print(f'===================================== print alpha performance table =========================================')
    performance_table = calc_performance_table(df_pos_weight, close_30s_data, DELAY_NUM)
    print(performance_table)

