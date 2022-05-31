import pandas as pd
import openpyxl

# データをデータテーブルに読み込む
df = pd.read_excel('ds07_temp_power_exercise_rev.xlsx')

# データフレームからX列とY列を取り出す
x = df[['最高気温(℃)']]
y = df[['東京エリア需要(万kWh)_14時']]

print(x)

# 最低値、最高値を確認 (.to_string() でデータ型の出力を抑制)
print('最高気温の最低値', x.min().to_string())
print('最高気温の最高値', x.max().to_string())
print('東京エリア需要の最低値', y.min().to_string())
print('東京エリア需要の最高値', y.max().to_string())