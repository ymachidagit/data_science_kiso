import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import datetime
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np


# エクセルファイルの電力データのシートをデータフレームに読み込む (ヘッダはスキップし、必要部分の列名を指定)
covid19_df = pd.read_excel('nhk_news_covid19_prefectures_daily_data.xlsx', \
                            names=['日付', '', '都道府県名', '感染者数_1日ごとの発表数', '感染者数_累計',\
                                    '死者数_1日ごとの発表数', '死者数_累計', ''])

covid19_df=covid19_df[covid19_df['都道府県名']=='埼玉県'] #埼玉県のデータのみ抽出

covid19_df=covid19_df[covid19_df['日付']<datetime.datetime(2021,1,1)] #2020年のデータのみを抽出
#covid19_df=covid19_df[covid19_df['日付']>=datetime.datetime(2020,1,16)] #2020年1月16日からのデータのみを抽出
#covid19_df=covid19_df[covid19_df['日付']>=datetime.datetime(2020,2,21)] #2020年2月21日からのデータのみを抽出
covid19_df=covid19_df[covid19_df['日付']>=datetime.datetime(2020,3,31)] #2020年3月31日からのデータのみを抽出

weather_df=pd.read_excel('kumagaya.xlsx',\
                        skiprows=[0, 1, 2, 3, 4],\
                        names=['年月日', '平均気温(℃)', '', '', '最高気温(℃)', '', '', '最低気温(℃)', '', '', '降水量の合計', '', '', '',\
                                '10分間の降水量の最大', '', '', '', '日照時間(時間)', '', '', '', '平均風速(m/s)', '', '',\
                                '平均蒸気圧', '', '', '平均湿度(％)', '', '', '最小相対湿度', '', '', '平均現地気圧', '', '',])

weather_df['DATETIME'] = pd.to_datetime(weather_df['年月日'], format='%Y/%m/%d')

#weather_df=weather_df[weather_df['DATETIME']>=datetime.datetime(2020,1,16)] #2020年1月16日からのデータのみを抽出
#weather_df=weather_df[weather_df['DATETIME']>=datetime.datetime(2020,2,21)] #2020年2月21日からのデータのみを抽出
weather_df=weather_df[weather_df['DATETIME']>=datetime.datetime(2020,3,31)] #2020年3月31日からのデータのみを抽出


def Regression_analysis(x_name, y_name):
        # データフレームからX列とY列を取り出す
        x = weather_df[[x_name]]
        y = covid19_df[[y_name]]

        # 単回帰回帰分析を実施 (線形回帰のモデルを準備し、データを入力)
        model_p1 = LinearRegression()
        model_p1.fit(x, y)

        # モデルと寄与率を出力
        print(x_name + 'と' + y_name)
        print('単回帰分析')
        print('y= %.6f + %.6fx' % (model_p1.intercept_, model_p1.coef_))
        print('寄与率 R^2:', model_p1.score(x, y))

        # 多項式回帰のために x に 2次の項を追加
        pf = PolynomialFeatures(degree=2)
        x2 = pf.fit_transform(x)  # x2 -> [1.00000e+00 3.22000e+01 1.03684e+03]
                                  #        (バイアス項)  (1次の項)   (2次の項)

        # 多項式回帰分析を実施 (線形回帰のモデルを準備し、2次の項を含むデータを入力)
        model_p2 = LinearRegression()
        model_p2.fit(x2, y)

        # モデルと寄与率を出力
        print('二次の多項式回帰分析')
        print('y= %.6f + %.6fx + %.6fx^2' % (model_p2.intercept_, model_p2.coef_[0][1], model_p2.coef_[0][2]))
        print('寄与率 R^2:', model_p2.score(x2, y))

        # 元データの散布図を作成
        fig = plt.figure(figsize=(8, 7))
        ax1 = fig.add_subplot(111)
        ax1.plot(x, y, 'o')

        # 求めた単回帰分析の結果(直線)を描画
        x_linspace = pd.DataFrame(np.linspace(x.min(), x.max(), 100), columns=[x_name])
        ax1.plot(x_linspace, model_p1.predict(x_linspace))

        # 多項式回帰分析の結果(曲線)を描画
        x_linspace_pf = pf.fit_transform(x_linspace)
        y_pred = model_p2.predict(x_linspace_pf)
        ax1.plot(x_linspace, y_pred)

        ax1.set_title(y_name + 'と' + x_name + 'の関係',size=16,fontname="MS Gothic")
        ax1.set_xlabel(x_name,size=12,fontname="MS Gothic")
        ax1.set_ylabel(y_name,size=12,fontname="MS Gothic")

        plt.show()

#Regression_analysis('平均気温(℃)', '感染者数_1日ごとの発表数')
Regression_analysis('最高気温(℃)', '感染者数_1日ごとの発表数')
#Regression_analysis('最低気温(℃)', '感染者数_1日ごとの発表数')
#Regression_analysis('日照時間(時間)', '感染者数_1日ごとの発表数')
Regression_analysis('平均風速(m/s)', '感染者数_1日ごとの発表数')
Regression_analysis('平均湿度(％)', '感染者数_1日ごとの発表数')
