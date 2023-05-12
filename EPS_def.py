from labdai.metrics import beta
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from MAIN import *

plt.rcParams.update({'font.size': 12})

# EPS DATAFRAME
eps_df = pd.read_csv("Earnings_Per_Share.csv")
eps_df['Dates'] = daily_dates
eps_df['Dates'] = pd.to_datetime(eps_df['Dates'], format='%Y-%m-%d')
eps_df = eps_df.loc[(eps_df['Dates'].dt.is_month_end)]
eps_df.pop('Dates')

tickers_prov = pd.DataFrame({'Tickers': eps_df.columns})
tickers_prov = tickers_prov.reset_index()
tickers_prov = tickers_prov.iloc[:,1:]

#std_eps = eps_df.rolling(12).std()
std_annual_eps = pd.DataFrame()
year = 2002
for j in range(eps_df.shape[1]):
    std_ticker = []
    for i in range(0, len(eps_df), 12):
        year += 1
        globals()["std_eps_ann %d" % year] = (eps_df.iloc[i:i+12, j]).std()
        std_ticker.append((globals()["std_eps_ann %d" % year]))
    std_annual_eps[f'{eps_df.columns[j]}'] = std_ticker

reps = [12 for row in std_annual_eps.iterrows()]
driver3 = std_annual_eps.loc[np.repeat(std_annual_eps.index.values, reps)]
driver3.reset_index(inplace=True)
driver3 = driver3.iloc[:, 1:]

prices_int = driver3.columns.intersection(prices_df.columns)
prices_df = prices_df[prices_int]

# DIVIDE EPS INTO 30/40/30
# FOR LOOP TO CREATE QUARTERLY DATAFRAMES (EPS)
month = 0
x = ['Q0']

driver3 = comp_filter(driver3, comp_df)
driver3_int = cap.columns.intersection(driver3.columns)
cap = cap[driver3_int]
driver3 = driver3[driver3_int]
int_ret_df = ret_df[driver3_int]
std_dev = std_dev[driver3_int]
prices_df["Dates"] = ret_dates
prices_df = prices_df[driver3_int]


#TICKERS
tickers = pd.DataFrame({'Tickers': driver3.columns})
tickers = tickers.reset_index()
tickers = tickers.iloc[:,1:]

Top_Big = []
Bottom_Small= []
Top_Small = []
Bottom_Big = []
StdDev_Top = []
StdDev_Bottom = []
prices_top_mean = []
prices_bottom_mean = []

for i in range(0, driver3.shape[0]-1):
    month += 1
    eps = pd.DataFrame({'EPS': driver3.iloc[i, :]})
    eps = eps.reset_index()
    eps = eps.iloc[:, 1:]
    rtn = pd.DataFrame({'Returns': int_ret_df.iloc[i+1, :]})
    rtn = rtn.reset_index()
    rtn = rtn.iloc[:, 1:].fillna(0)
    capit = pd.DataFrame({'capitalization': cap.iloc[i, :]})
    capit = capit.reset_index()
    capit = capit.iloc[:, 1:]
    sd_monthly = pd.DataFrame({'STD_DEV': std_dev.iloc[i, :]})
    sd_monthly = sd_monthly.reset_index()
    sd_monthly = sd_monthly.iloc[:,1:]
    prices = pd.DataFrame({"Prices" : prices_df.iloc[i,:]})
    prices = prices.reset_index()
    prices = prices.iloc[:, 1:]
    frames = [tickers, eps, capit, rtn, sd_monthly, prices]


    # CREATE DATAFRAME FOR EACH PERIOD
    globals()['M_EPS%d' % month] = pd.concat(frames, axis=1)

    # ELIMINATE ROWS WITH EPS = 0
    globals()['M_EPS%d' % month] = globals()['M_EPS%d' % month].fillna(0)
    globals()['M_EPS%d' % month] = globals()['M_EPS%d' % month][globals()['M_EPS%d' % month].EPS != 0]

    # Quartili DIVISION BY EPS FOR EACH PERIOD
    globals()['M_EPS%d' % month]['Quintile'] = pd.qcut(globals()['M_EPS%d' % month]['EPS'].rank(method='first'), q=[0, .3, .7, 1],labels=["Bottom_30","Medium_40","Top_30"],duplicates="drop")

    x.append('M_EPS%d' % month)

    globals()['M_Bottom%d' % month] = globals()['M_EPS%d' % month].loc[globals()['M_EPS%d' % month].Quintile == "Top_30"]
    globals()['M_Top%d' % month] = globals()['M_EPS%d' % month].loc[globals()['M_EPS%d' % month].Quintile == "Bottom_30"]
    globals()['M_Mid%d' % month] = globals()['M_EPS%d' % month].loc[globals()['M_EPS%d' % month].Quintile == "Medium_40"]

    globals()['Prices_Top_M%d' % month] = globals()['M_Top%d' % month].Prices
    globals()['Prices_Bottom_M%d' % month] = globals()['M_Bottom%d' % month].Prices

    prices_top_mean.append(globals()['Prices_Top_M%d' % month].mean())
    prices_bottom_mean.append(globals()['Prices_Bottom_M%d' % month].mean())

    globals()['Bottom_median%d' % month] = globals()['M_Bottom%d' % month].capitalization.median()
    globals()['M_Bottom_SmallCap%d' % month] = globals()['M_Bottom%d' % month].loc[globals()['M_Bottom%d' % month].capitalization < globals()['Bottom_median%d' % month]]
    globals()['M_Bottom_BigCap%d' % month] = globals()['M_Bottom%d' % month].loc[globals()['M_Bottom%d' % month].capitalization >= globals()['Bottom_median%d' % month]]
    globals()['M_Bottom_SmallCap_AVG_RET%d' % month] = globals()['M_Bottom_SmallCap%d' % month].Returns.mean()
    globals()['M_Bottom_BigCap_AVG_RET%d' % month] = globals()['M_Bottom_BigCap%d' % month].Returns.mean()

    globals()['Top_median%d' % month] = globals()['M_Top%d' % month].capitalization.median()
    globals()['M_Top_SmallCap%d' % month] = globals()['M_Top%d' % month].loc[globals()['M_Top%d' % month].capitalization < globals()['Top_median%d' % month]]
    globals()['M_Top_BigCap%d' % month] = globals()['M_Top%d' % month].loc[globals()['M_Top%d' % month].capitalization >= globals()['Top_median%d' % month]]
    globals()['M_Top_SmallCap_AVG_RET%d' % month] = globals()['M_Top_SmallCap%d' % month].Returns.mean()
    globals()['M_Top_BigCap_AVG_RET%d' % month] = globals()['M_Top_BigCap%d' % month].Returns.mean()

    globals()['M_Top_StdDev_AVG%d' % month] = globals()['M_Top%d' % month].STD_DEV.mean()
    globals()['M_Bottom_StdDev_AVG%d' % month] = globals()['M_Bottom%d' % month].STD_DEV.mean()
    globals()['M_Top_avgRtn%d' % month] = globals()['M_Top%d' % month].Returns.mean()
    globals()['M_Bottom_avgRtn%d' % month] = globals()['M_Bottom%d' % month].Returns.mean()

    Top_Big.append(globals()['M_Top_BigCap_AVG_RET%d' % month])
    Top_Small.append(globals()['M_Top_SmallCap_AVG_RET%d' % month])
    Bottom_Small.append(globals()['M_Bottom_SmallCap_AVG_RET%d' % month])
    Bottom_Big.append(globals()['M_Bottom_BigCap_AVG_RET%d' % month])
    StdDev_Top.append(globals()['M_Top_StdDev_AVG%d' % month])
    StdDev_Bottom.append(globals()['M_Bottom_StdDev_AVG%d' % month])

del x[0]


EPS = ((0.5*((np.array(Top_Big)) + np.array(Top_Small))) - (0.5*((np.array(Bottom_Big) + np.array(Bottom_Small)))))*100
fama_french = fama_french.iloc[1:,:]
fama_french['EPS'] = EPS


# Modifiche dataframe pre regressione e salvataggio RMW e RF per uso futuro
RMW = fama_french['RMW']
rf = fama_french['RF']
reg_dates = fama_french['Dates']
fama_french.pop('RMW')
fama_french.pop('RF')
fama_french.pop('Dates')


# Costruzione variabili regressione
Top_Cumprod = (((np.array(Top_Big) + (np.array(Top_Small)))/2 + 1.).cumprod())-1
Bottom_Cumprod = (((np.array(Bottom_Small) + (np.array(Bottom_Big)))/2 + 1.).cumprod())-1
Top_perf = (((np.array(Top_Big) + (np.array(Top_Small)))/2))
Bottom_perf =(((np.array(Bottom_Small) + (np.array(Bottom_Big)))/2))
Perf = Top_perf - Bottom_perf
Perf_Cumprod = (((Top_perf - Bottom_perf)+1).cumprod())-1
y = Perf_Cumprod-rf
index = ret_df['SPX Index']
index = index[1:]
index_cumprod = ((1+index).cumprod())-1

Top_perf_cent = (((np.array(Top_Big) + (np.array(Top_Small)))/2))*100
Bottom_perf_cent =(((np.array(Bottom_Small) + (np.array(Bottom_Big)))/2))*100
Perf_cent = pd.DataFrame((Top_perf - Bottom_perf)*100)

# performance plot
plt.plot(reg_dates, y*100, label='Strategy Performance', linewidth=2)
plt.plot(reg_dates, index_cumprod*100, label='SPX Index', linewidth=2)
plt.title('Comparison between Long-Short strategy based on EPS and SPX Index')
plt.legend()
plt.show()
plt.close()

# Regression e summary
EPS_model = sm.OLS(y, fama_french).fit()
print(EPS_model.summary())

# PRICES CONSTRUCTION
Top_1, Bottom_1, Perf_1, Index_1 = prices_construction(Top_Cumprod, Bottom_Cumprod, Perf_Cumprod, index_cumprod)

# MAXIMUM DRAWDOWN
mdd_index = maximum_drawdown(Index_1)*100
mdd_top = maximum_drawdown(Top_1['Prices'])*100
mdd_bottom = maximum_drawdown(Bottom_1['Prices'])*100
mdd_perf = maximum_drawdown(Perf_1['Prices'])*100

# ANNUALIZED AVERAGE RETURNS
top_AAR, bottom_AAR, perf_AAR = performance_AAR(Top_perf_cent, Bottom_perf_cent, Perf_Cumprod)

# ANNUALIZED VOLATILITY
top_annual_volatility, bottom_annual_volatility, perf_annual_volatility = Annualized_Volatility(Top_1, Bottom_1, Perf_1, Top_perf)

# SHARPE RATIO  ------ VARIABILI CHIAMATE ANNUAL PER ORA SONO TRIMESTRALI
AvgSR_top, AvgSR_bottom, AvgSR_perf, SR_top_Q, SR_bottom_Q, SR_perf_Q = Sharpe_Ratio(StdDev_Bottom, StdDev_Top, Top_perf_cent, rf, Bottom_perf_cent, Perf_cent)

#plt.plot(yearly_dates, SR_top_annual, label='SR top')
#plt.plot(yearly_dates, SR_bottom_annual, label='SR bottom')
#plt.plot(yearly_dates, SR_perf_annual, label='SR perf')
#plt.legend()
#plt.show()
#plt.close()


#BETA
beta_top = beta(Top_1["Prices"], Index_1['Prices'])
beta_bottom = beta(Bottom_1["Prices"], Index_1['Prices'])
beta_perf = beta(Perf_1["Prices"], Index_1['Prices'])

top_ann_betas, bottom_ann_betas, perf_ann_betas = beta_driver(Top_1, Index_1, Bottom_1, Perf_1)

plt.plot(yearly_dates, top_ann_betas, label='Top Betas')
plt.plot(yearly_dates, bottom_ann_betas, label='Bottom Betas')
plt.plot(yearly_dates, perf_ann_betas, label='Betas')
plt.legend()
#plt.show()
plt.close()
print('STOP')