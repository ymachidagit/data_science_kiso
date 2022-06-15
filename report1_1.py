import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import datetime
import itertools

# エクセルファイルの電力データのシートをデータフレームに読み込む (ヘッダはスキップし、必要部分の列名を指定)
covid19_df = pd.read_excel('nhk_news_covid19_prefectures_daily_data.xlsx', \
                            names=['日付', '', '都道府県名', '感染者数_1日ごとの発表数', '感染者数_累計',\
                                    '死者数_1日ごとの発表数', '死者数_累計', ''])

covid19_df=covid19_df[covid19_df['都道府県名']=='埼玉県'] #埼玉県のデータのみ抽出

covid19_df=covid19_df[covid19_df['日付']<datetime.datetime(2021,1,1)] #2020年のデータのみを抽出
#covid19_df=covid19_df[covid19_df['日付']>=datetime.datetime(2020,1,16)] #2020年1月16日からのデータのみを抽出
#covid19_df=covid19_df[covid19_df['日付']>=datetime.datetime(2020,2,21)] #2020年2月21日からのデータのみを抽出
covid19_df=covid19_df[covid19_df['日付']>=datetime.datetime(2020,3,31)] #2020年3月31日からのデータのみを抽出

PM_df = pd.read_excel('さいたま市役所_微小粒子状物質rev.xlsx', \
                        skiprows=[0,1],\
                        names=['日付', '', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',\
                                '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',\
                                '時間', '合計', '平均', '最高', '最低'])

PM_df['DATETIME'] = pd.to_datetime(PM_df['日付'], format='%Y/%m/%d')

#PM_df=PM_df[PM_df['DATETIME']>=datetime.datetime(2020,1,16)] #2020年1月16日からのデータのみを抽出
#PM_df=PM_df[PM_df['DATETIME']>=datetime.datetime(2020,2,21)] #2020年2月21日からのデータのみを抽出
#PM_df=PM_df[PM_df['DATETIME']>=datetime.datetime(2020,3,31)] #2020年3月31日からのデータのみを抽出

weather_df=pd.read_excel('kumagaya.xlsx',\
                        skiprows=[0, 1, 2, 3, 4],\
                        names=['年月日', '平均気温(℃)', '', '', '最高気温(℃)', '', '', '最低気温(℃)', '', '', '降水量の合計', '', '', '',\
                                '10分間の降水量の最大', '', '', '', '日照時間(時間)', '', '', '', '平均風速(m/s)', '', '',\
                                '平均蒸気圧', '', '', '平均湿度(％)', '', '', '最小相対湿度', '', '', '平均現地気圧', '', '',])

weather_df['DATETIME'] = pd.to_datetime(weather_df['年月日'], format='%Y/%m/%d')

#weather_df=weather_df[weather_df['DATETIME']>=datetime.datetime(2020,1,16)] #2020年1月16日からのデータのみを抽出
#weather_df=weather_df[weather_df['DATETIME']>=datetime.datetime(2020,2,21)] #2020年2月21日からのデータのみを抽出
weather_df=weather_df[weather_df['DATETIME']>=datetime.datetime(2020,3,31)] #2020年3月31日からのデータのみを抽出

# print(covid19_df.head(5))
# print(PM_df.head(5))
# print(weather_df.head(5))

variables=['平均気温(℃)', '最高気温(℃)', '最低気温(℃)', '日照時間(時間)', '平均風速(m/s)', '平均湿度(％)'] #独立変数の候補

pair_list=[]
AIC_list=[]

y = covid19_df.loc[:, ['感染者数_1日ごとの発表数']].values

for i in range(1,len(variables)+1,1):
    for pair in itertools.combinations(variables,i):
        pair_list.append(pair)
        pair=list(pair)
        x = weather_df.loc[:, pair].values
        x_added_constant = sm.add_constant(x)  # 回帰計算のために定数項の列を追加
        model = sm.OLS(y, x_added_constant)
        result = model.fit()
        AIC_list.append(result.aic)

df_model=pd.DataFrame({'pair':pair_list,'AIC':AIC_list})

df_model=df_model.sort_values('AIC')[:5]
print(df_model)

