import pandas as pd
from labdai.data_loaders import load_market_data
from labdai.metrics import beta
import numpy as np


# COMP FILTER

def comp_filter(driver, comp_df):
    df = pd.DataFrame()
    for i in range(len(comp_df)):
        comp = comp_df.iloc[i, :].fillna(0)
        driver_comp = driver.iloc[i, :].fillna(0)*comp
        df[i] = driver_comp
    return df.T



# PRICES CONSTRUCTION
def prices_construction(Top_Cumprod, Bottom_Cumprod, Perf_Cumprod, index_cumprod):
    Top_1 = pd.DataFrame()
    Top_1['Prices'] = (pd.DataFrame(Top_Cumprod)).mul(100).add(100)

    Bottom_1 = pd.DataFrame()
    Bottom_1["Prices"] = (pd.DataFrame(Bottom_Cumprod)).mul(100).add(100)

    Perf_1 = pd.DataFrame()
    Perf_1['Prices'] = (pd.DataFrame(Perf_Cumprod)).mul(100).add(100)

    Index_1 = pd.DataFrame()
    Index_1['Prices'] = (pd.DataFrame(index_cumprod)).mul(100).add(100)
    Index_1 = Index_1.reset_index()
    Index_1 = Index_1.iloc[:, 1:]

    return Top_1, Bottom_1, Perf_1, Index_1


# MAXIMUM DRAWDOWN
def maximum_drawdown(perf_series:pd.DataFrame):
    cum_max = perf_series.cummax().iloc[:-1]
    tap_series = (perf_series.iloc[1:] - cum_max) / cum_max
    return tap_series.min()


# ANNUALIZED AVERAGE RETURNS

def annualized_average_return(perf_series:pd.Series) -> float:
    return perf_series.mean() * 12.

def performance_AAR(Top_perf_cent, Bottom_perf_cent, Perf_Cumprod):
    anno = 2002
    top_AAR = []
    bottom_AAR = []
    perf_AAR = []
    for i in range(0, len(Top_perf_cent), 12):
        anno += 1
        globals()["top_ann_tot_rtn %d" % anno] = annualized_average_return(pd.DataFrame(Top_perf_cent).iloc[i:i+12])
        top_AAR.append(globals()["top_ann_tot_rtn %d" % anno])
        globals()["bottom_ann_tot_rtn %d" % anno] = annualized_average_return(pd.DataFrame(Bottom_perf_cent).iloc[i:i+12])
        bottom_AAR.append(globals()["bottom_ann_tot_rtn %d" % anno])
        globals()["perf_ann_tot_rtn %d" % anno] = annualized_average_return(pd.DataFrame(Perf_Cumprod).iloc[i:i+12])
        perf_AAR.append(globals()["perf_ann_tot_rtn %d" % anno])

    return top_AAR, bottom_AAR, perf_AAR



# ANNUALIZED AVG VOLATILITY
def price_to_returns(prices_df:pd.DataFrame, lag) -> pd.DataFrame:
    assert prices_df.index.is_monotonic_increasing
    prices_past = prices_df.iloc[:-lag, :]
    prices_future = prices_df.iloc[lag:,:]
    return (prices_future / prices_past.values) - 1.


def annualized_volatility(perf_series:pd.DataFrame) -> float:
    monthly_return_series = price_to_returns(perf_series, lag=1)
    ann_avg_vol = monthly_return_series.std() * (12 ** 0.5)
    return ann_avg_vol


yearly_dates = pd.date_range(start='2003', periods=15, freq='Y')
dates = pd.date_range(start="01/01/2003", end="12/31/2018", freq='m')
daily_dates = pd.date_range(start="01/01/2003", end="12/31/2018", freq='d')
md = load_market_data('data/SPX')


fama_french = pd.read_csv('FamaFrench.csv')
fama_french['Dates'] = dates

# CAPITALIZATION DATAFRAME
cap = md['capitalization'].reset_index().dropna(axis=1, how='all').fillna(0)
cap = cap.loc[cap['Dates'].dt.is_month_end]
cap = cap.iloc[1:len(cap)]
cap = cap.reindex(sorted(cap.columns), axis=1)


# COMPOSITION DATAFRAME
comp_df = md['composition'].reset_index().fillna(0)
daily_comp = comp_df.iloc[1:,:]
daily_comp.drop(daily_comp.index[[426, 1887, 3348, 4809]], inplace=True)
comp_df = comp_df.loc[comp_df['Dates'].dt.is_month_end]
comp_df = comp_df.iloc[1:len(comp_df)]
comp_df = comp_df.reindex(sorted(comp_df.columns), axis=1)
comp_df = comp_df.loc[:, (comp_df != 0).any(axis=0)]

# PRICES DATAFRAME
prices_df = md['close'].reset_index()
prices_df['Dates'] = pd.to_datetime(prices_df['Dates'], format='%Y-%m-%d')
prices_df = prices_df.reindex(sorted(prices_df.columns), axis=1)
daily_prices = prices_df
daily_prices = daily_prices.iloc[1:,:]
daily_prices.drop(daily_prices.index[[426, 1887, 3348, 4809]], inplace=True)
prices_df = prices_df.loc[prices_df['Dates'].dt.is_month_end]
prices_df = prices_df.iloc[1:len(prices_df)]
prices_df.pop('Dates')

# COMPUTING QUARTERLY and DAILY RETURNS

returns = []
for i in range(0, prices_df.shape[0]):
    ret_i = (prices_df.iloc[i, 0:]/prices_df.iloc[i-1, 0:] - 1)
    returns.append(ret_i)

ret_dates = pd.date_range(start="01/31/2003", end="12/31/2018", freq='m')
ret_df = pd.DataFrame(returns)
ret_df['Dates'] = ret_dates


# DAILY RETURNS
daily_prices.pop('Dates')
daily_comp.pop('Dates')
daily_comp = daily_comp.reindex(sorted(daily_comp.columns), axis=1)
daily_returns = []
for i in range(0, daily_prices.shape[0]):
    daily_ret_i = (daily_prices.iloc[i, 0:]/daily_prices.iloc[i-1, 0:] - 1)
    daily_returns.append(daily_ret_i)

daily_ret_df = pd.DataFrame(daily_returns)
daily_ret_df = comp_filter(daily_ret_df, daily_comp) #filtering for the spx composition
daily_ret_df = daily_ret_df.iloc[1:,:]
daily_ret_df_perc = daily_ret_df * 100
std_dev = daily_ret_df_perc.rolling(30, min_periods=30).std() #rolling monthly std dev
std_dev = std_dev.iloc[29::30]
std_dev = std_dev.iloc[:-2,:]
std_dev['Dates'] = ret_dates


def beta_driver(Top_1, Index_1, Bottom_1, Perf_1):
    top_ann_betas = []
    bottom_ann_betas = []
    perf_ann_betas = []

    year = 2002
    for i in range(0, len(Top_1["Prices"]) - 12, 12):
        year += 1
        globals()["top_ann_betas %d" % year] = beta(Top_1["Prices"].iloc[i:i + 12], Index_1['Prices'].iloc[i:i + 12])
        top_ann_betas.append(globals()["top_ann_betas %d" % year])
        globals()["bottom_ann_betas %d" % year] = beta(Bottom_1["Prices"].iloc[i:i + 12],
                                                       Index_1['Prices'].iloc[i:i + 12])
        bottom_ann_betas.append(globals()["bottom_ann_betas %d" % year])
        globals()["perf_ann_betas %d" % year] = beta(Perf_1['Prices'].iloc[i:i + 12], Index_1['Prices'].iloc[i:i + 12])
        perf_ann_betas.append(globals()["perf_ann_betas %d" % year])

    return top_ann_betas, bottom_ann_betas, perf_ann_betas


# ANNUALIZED AVG VOLATILITY
def Annualized_Volatility(Top_1, Bottom_1, Perf_1, Top_perf):
    bottom_annual_volatility = []
    top_annual_volatility = []
    perf_annual_volatility = []
    year = 2002
    for i in range(0, len(Top_perf), 12):
        year += 1
        globals()["top_ann_vol %d" % year] = annualized_volatility(pd.DataFrame(Top_1.Prices).iloc[i:i+12])
        top_annual_volatility.append(globals()["top_ann_vol %d" % year])
        globals()["bottom_Ann_vol %d" % year] = annualized_volatility(pd.DataFrame(Bottom_1.Prices).iloc[i:i+12])
        bottom_annual_volatility.append(globals()["bottom_Ann_vol %d" % year])
        globals()["perf_ann_vol %d" % year] = annualized_volatility(pd.DataFrame(Perf_1.Prices).iloc[i:i + 12])
        perf_annual_volatility.append(globals()["perf_ann_vol %d" % year])
    #top_ann_avg_vol = annualized_volatility(pd.DataFrame(Top_1.Prices))/15
    #bottom_ann_avg_vol = annualized_volatility(pd.DataFrame(Bottom_1.Prices))/15
    #perf_ann_avg_vol = annualized_volatility(pd.DataFrame(Perf_1.Prices))/15

    return top_annual_volatility, bottom_annual_volatility, perf_annual_volatility


# SHARPE RATIO   ------ VARIABILI CHIAMATE ANNUAL PER ORA SONO TRIMESTRALI
def Sharpe_Ratio(StdDev_Bottom, StdDev_Top, Top_perf_cent, rf, Bottom_perf_cent, Perf_cent):
    StdDev_Bottom = np.array(StdDev_Bottom)
    StdDev_Top = np.array(StdDev_Top)
    SR_Top = (Top_perf_cent - np.array(rf)) / (np.array(StdDev_Top))
    SR_Bottom = (Bottom_perf_cent - np.array(rf)) / (np.array(StdDev_Bottom))

    StdDev_Perf = []
    Annual_Perf = []
    y = 2002
    for i in range(0, len(Perf_cent)-3, 3):
        y += 1
        globals()["std_dev_perf %d" % y] = (Perf_cent.iloc[i:i+3]).std()
        StdDev_Perf.append(globals()["std_dev_perf %d" % y])
        globals()["avg_annual_perf%d" % y] = (Perf_cent.iloc[i:i+3]).mean()
        Annual_Perf.append(globals()["avg_annual_perf%d" % y])

    SR_top = pd.DataFrame(SR_Top)
    SR_bottom = pd.DataFrame(SR_Bottom)

    SR_top_Q = []
    SR_bottom_Q = []

    quarter = 0
    for i in range(0, len(SR_Top), 3):
        quarter += 1
        globals()["top_SR_Q %d" % quarter ] = SR_top.iloc[i:i+3]
        SR_top_Q.append(globals()["top_SR_Q %d" % quarter].mean())
        globals()["bottom_Q_vol %d" % quarter] = SR_bottom.iloc[i:i+3]
        SR_bottom_Q.append(globals()["bottom_Q_vol %d" % quarter].mean())

    SR_perf_Q = (np.array(Annual_Perf) / np.array(StdDev_Perf))

    AvgSR_top = np.array(SR_top_Q).mean()
    AvgSR_bottom = np.array(SR_bottom_Q).mean()
    AvgSR_perf = np.array(SR_perf_Q).mean()

    return AvgSR_top, AvgSR_bottom, AvgSR_perf, SR_top_Q, SR_bottom_Q, SR_perf_Q
