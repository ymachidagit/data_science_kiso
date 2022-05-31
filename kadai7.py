import pandas as pd
import openpyxl

# データをデータテーブルに読み込む
df = pd.read_excel('ds07_temp_power_exercise.xlsx')

# データフレームからX列とY列を取り出す
x = df[['最高気温(℃)']]
y = df[['東京エリア需要']]

print(x)

# 最低値、最高値を確認 (.to_string() でデータ型の出力を抑制)
print('最高気温の最低値', x.min().to_string())
print('最高気温の最高値', x.max().to_string())
print('電力使用実績の最低値', y.min().to_string())
print('電力使用実績の最高値', y.max().to_string())