import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import datetime
import itertools

# エクセルファイルの電力データのシートをデータフレームに読み込む (ヘッダはスキップし、必要部分の列名を指定)
power_df = pd.read_excel('ds07_temp_power_exercise.xlsx', \
            sheet_name='東京電力パワーグリッド エリア需給実績データ', skiprows=[0,1],\
            names=['DATE', 'TIME', 'power', '', '', '', '', '', '', '', '', '', '', '', ''])
power_df['DATETIME'] = pd.to_datetime(power_df['DATE']+' '+power_df['TIME'], format='%Y/%m/%d %H:%M')

power_df=power_df[power_df['TIME']=='14:00'] #14:00のデータのみ抽出

power_df=power_df[power_df['DATETIME']<datetime.datetime(2020,2,1,0,0,0)] #2020年1月のデータのみを抽出

weather_df = pd.read_excel('ds07_temp_power_exercise.xlsx', \
            sheet_name='2020熊谷市気象データ_data', skiprows=[0,1,2],\
            names=['年月日', '平均気温', '', '', '最高気温', '', '', '', '', '最低気温', '', '', '', '', \
                    '降水量', '', '', '', '日照時間', '', '', '', '降雪量', '', '', '', '平均風速', '', '', \
                    '平均蒸気圧', '', '', '平均湿度', '', '', '平均現地気圧', '', '', '', '', '', '天気概況', '', ''])
weather_df['DATETIME'] = pd.to_datetime(weather_df['年月日'], format='%Y/%m/%d')

weather_df=weather_df[weather_df['DATETIME']<datetime.datetime(2020,2,1)] #2020年1月のデータのみを抽出

variables=['平均気温', '最高気温', '最低気温', '日照時間', '平均風速', '平均蒸気圧', '平均湿度'] #独立変数の候補

pair_list=[]
AIC_list=[]

y = power_df.loc[:, ['power']].values

for i in range(1,8,1):
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

