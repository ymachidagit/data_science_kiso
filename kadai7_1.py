import pandas as pd
import openpyxl
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np

# データをデータテーブルに読み込む
df = pd.read_excel('ds07_temp_power_exercise_rev.xlsx')

# データフレームからX列とY列を取り出す
x = df[['最高気温(℃)']].loc[0:30]
y = df[['東京エリア需要(万kWh)_14時']].loc[0:30]

# 単回帰回帰分析を実施 (線形回帰のモデルを準備し、データを入力)
model_p1 = LinearRegression()
model_p1.fit(x, y)

# モデルと寄与率を出力
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
print('y= %.6f + %.6fx + %.6fx^2' % (model_p2.intercept_, model_p2.coef_[0][1], model_p2.coef_[0][2]))
print('寄与率 R^2:', model_p2.score(x2, y))

# 元データの散布図を作成
fig = plt.figure(figsize=(8, 7))
ax1 = fig.add_subplot(111)
ax1.plot(x, y, 'o')

# 求めた単回帰分析の結果(直線)を描画
x_linspace = pd.DataFrame(np.linspace(x.min(), x.max(), 100), columns=['熊谷市の最高気温(℃)'])
ax1.plot(x_linspace, model_p1.predict(x_linspace))

# 多項式回帰分析の結果(曲線)を描画
x_linspace_pf = pf.fit_transform(x_linspace)
y_pred = model_p2.predict(x_linspace_pf)
ax1.plot(x_linspace, y_pred)
