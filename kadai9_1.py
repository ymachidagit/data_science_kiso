import openpyxl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import statsmodels.api as sm
from pmdarima import auto_arima

nikkei_daily_df = pd.read_excel('nikkei_stock_average_daily_jp.xlsx')

nikkei_daily_df=nikkei_daily_df.iloc[:-1, 0:2]
nikkei_daily_df.columns=['date', 'price']
nikkei_daily_df['date'] = pd.to_datetime(nikkei_daily_df['date'], format='%Y-%m-%d')
nikkei_daily_df=nikkei_daily_df.set_index('date')

#2021/1/1~2022/6/20までのデータを埋める
#dates_df = pd.DataFrame(index=pd.date_range('2021-01-01', periods=365*1+31*5-2+18, freq='D'))
#dates_df = pd.DataFrame(index=pd.date_range('2019-01-04', periods=365*3+31*5-2+16, freq='D'))

#2019/1/4~2022/6/10までのデータを埋める
dates_df = pd.DataFrame(index=pd.date_range('2019-01-04', periods=365*3+31*5-2+6, freq='D'))
#dates_df = pd.DataFrame(index=pd.date_range('2022-01-01', periods=31*5-2+8, freq='D'))

nikkei_daily_df=nikkei_daily_df.merge(dates_df, how="outer", left_index=True, right_index=True)
nikkei_daily_df=nikkei_daily_df.interpolate('time')

print(nikkei_daily_df.tail(5))

def autocorrelation_graph():
    #日経平均株価のグラフを表示
    fig = plt.figure(figsize=(8, 20))
    ax = fig.add_subplot(311)
    ax.plot(nikkei_daily_df)
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(0, y_max * 1.05)
    ax.grid(axis='both',linestyle='dotted', color='c')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # 自己相関係数と偏自己相関係数のグラフを表示
    ax1 = fig.add_subplot(312)
    fig = sm.graphics.tsa.plot_acf(nikkei_daily_df, ax=ax1)
    ax2 = fig.add_subplot(313)
    fig = sm.graphics.tsa.plot_pacf(nikkei_daily_df, ax=ax2, method='ywm')

    plt.show()

def decompose_graph():
    # 時系列データをトレンド、規則的変動成分、不規則変動成分に分解
    res = sm.tsa.seasonal_decompose(nikkei_daily_df, period=7)
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(411)
    ax1.set_ylabel('observed')
    ax1.plot(res.observed)
    ax2 = fig.add_subplot(412)
    ax2.set_ylabel('trend')
    ax2.plot(res.trend)
    ax3 = fig.add_subplot(413)
    ax3.set_ylabel('seasonal')
    ax3.plot(res.seasonal)
    ax4 = fig.add_subplot(414)
    ax4.set_ylabel('resid')
    ax4.plot(res.resid)

    plt.show()

# モデル構築と検証のためのデータを準備
# train = nikkei_daily_df[0:849]  # 学習用データ 2021年まで
# test = nikkei_daily_df[849:]  # 検証用データ 2022年分

train = nikkei_daily_df[0:365*3-2]  # 学習用データ 2021年まで
test = nikkei_daily_df[365*3-2:]  # 検証用データ 2022年分

#print(nikkei_daily_df[0:849].tail(5))
print(nikkei_daily_df[0:365*3-2].tail(5))

def autoARIMA():
    #stepwise_fit = auto_arima(train, seasonal=True, trace=True, m=7, stepwise=True)
    stepwise_fit = auto_arima(train, seasonal=False, trace=True, stepwise=True)
    stepwise_fit.summary()

def pred_nikkei_stock_average():
    train.index = pd.DatetimeIndex(train.index.values, freq=train.index.inferred_freq)
    SARIMA = sm.tsa.SARIMAX(train.price, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)).fit()
    pred = SARIMA.predict('2022-01-01', '2022-06-20')
    # 実データと予測結果の図示
    plt.plot(train.price, label="train")
    plt.plot(test.price, "g", label="gt")
    plt.plot(pred, "r", label="pred")
    plt.legend()

    plt.show()

    print(pred[-40:])

#autocorrelation_graph()
#decompose_graph()
#autoARIMA()
pred_nikkei_stock_average()
