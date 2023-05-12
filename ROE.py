import matplotlib.pyplot as plt
import statsmodels.api as sm
from MAIN import *

plt.rcParams.update({'font.size': 12})

# ROE DATAFRAME
roe_df = pd.read_csv("roe.csv")
driver4 = roe_df.iloc[:-641, :]
driver4['Dates'] = daily_dates
driver4['Dates'] = pd.to_datetime(driver4['Dates'], format='%Y-%m-%d')
driver4 = driver4.loc[(driver4['Dates'].dt.is_month_end)]
driver4 = driver4.reindex(sorted(driver4.columns), axis=1)
driver4.pop('Dates')


# DIVIDE ROE INTO 30/40/30
# FOR LOOP TO CREATE QUARTERLY DATAFRAMES (ROE)
month = 0
x = ['Q0']

driver4 = comp_filter(driver4, comp_df)
driver4_int = cap.columns.intersection(driver4.columns)
cap = cap[driver4_int]
driver4 = driver4[driver4_int]
int_ret_df = ret_df[driver4_int]
std_dev = std_dev[driver4_int]
prices_df["Dates"] = ret_dates
prices_df = prices_df[driver4_int]


#TICKERS
tickers = pd.DataFrame({'Tickers': driver4.columns})
tickers = tickers.reset_index()
tickers = tickers.iloc[:, 1:]

Top_Big = []
Bottom_Small = []
Top_Small = []
Bottom_Big = []
StdDev_Top = []
StdDev_Bottom = []
prices_top_mean = []
prices_bottom_mean = []

for i in range(0, driver4.shape[0]-1):
    month += 1
    ROE = pd.DataFrame({'ROE': driver4.iloc[i, :]})
    ROE = ROE.reset_index()
    ROE = ROE.iloc[:, 1:]
    rtn = pd.DataFrame({'Returns': int_ret_df.iloc[i+1, :]})
    rtn = rtn.reset_index()
    rtn = rtn.iloc[:, 1:].fillna(0)
    capit = pd.DataFrame({'capitalization': cap.iloc[i, :]})
    capit = capit.reset_index()
    capit = capit.iloc[:, 1:]
    sd_monthly = pd.DataFrame({'STD_DEV': std_dev.iloc[i, :]})
    sd_monthly = sd_monthly.reset_index()
    sd_monthly = sd_monthly.iloc[:, 1:]
    prices = pd.DataFrame({"Prices": prices_df.iloc[i, :]})
    prices = prices.reset_index()
    prices = prices.iloc[:, 1:]
    frames = [tickers, ROE, capit, rtn, sd_monthly, prices]


    # CREATE DATAFRAME FOR EACH PERIOD
    globals()['M_ROE%d' % month] = pd.concat(frames, axis=1)

    # ELIMINATE ROWS WITH ROE = 0
    globals()['M_ROE%d' % month] = globals()['M_ROE%d' % month].fillna(0)
    globals()['M_ROE%d' % month] = globals()['M_ROE%d' % month][globals()['M_ROE%d' % month].ROE != 0]

    # Quartili DIVISION BY ROE FOR EACH PERIOD
    globals()['M_ROE%d' % month]['Quintile'] = pd.qcut(globals()['M_ROE%d' % month]['ROE'].rank(method='first'), q=[0, .3, .7, 1],labels=["Bottom_30","Medium_40","Top_30"],duplicates="drop")

    x.append('M_ROE%d' % month)

    globals()['M_Bottom%d' % month] = globals()['M_ROE%d' % month].loc[globals()['M_ROE%d' % month].Quintile == "Bottom_30"]
    globals()['M_Top%d' % month] = globals()['M_ROE%d' % month].loc[globals()['M_ROE%d' % month].Quintile == "Top_30"]
    globals()['M_Mid%d' % month] = globals()['M_ROE%d' % month].loc[globals()['M_ROE%d' % month].Quintile == "Medium_40"]

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


ROE = ((0.5*((np.array(Top_Big)) + np.array(Top_Small))) - (0.5*((np.array(Bottom_Big) + np.array(Bottom_Small)))))*100
fama_french = fama_french.iloc[1:,:]
fama_french['ROE'] = ROE


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
Bottom_perf = (((np.array(Bottom_Small) + (np.array(Bottom_Big)))/2))
Perf = Top_perf - Bottom_perf
Perf_Cumprod = (((Top_perf - Bottom_perf)+1).cumprod())-1
y = Perf_Cumprod-rf
index = ret_df['SPX Index']
index = index[1:]
index_cumprod = ((1+index).cumprod())-1

Top_perf_cent = (((np.array(Top_Big) + (np.array(Top_Small)))/2))*100
Bottom_perf_cent = (((np.array(Bottom_Small) + (np.array(Bottom_Big)))/2))*100
Perf_cent = pd.DataFrame((Top_perf - Bottom_perf)*100)

# Grafico confronto tra Top ROE, Bottom ROE, Indice --> TUTTI CUMULATI
plt.plot(reg_dates, y*100, label='Strategy Performance', linewidth=2)
plt.plot(reg_dates, index_cumprod*100, label='SPX Index', linewidth=2)
plt.title('Comparison between Long-Short strategy based on ROE and SPX Index')
plt.legend()
plt.show()
plt.close()

# Regression e summary
roe_model = sm.OLS(y, fama_french).fit()
print(roe_model.summary())

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
print('ciao')