import pandas as pd

ssdse_b_2022_df=pd.read_excel('C:\WorkSpace\DataScienceKiso\SSDSE-B-2022.xlsx')

population_df=ssdse_b_2022_df.loc[:,['SSDSE-B-2022', 'Prefecture', 'A1101']]
population_2019_df=population_df[population_df['SSDSE-B-2022']==2019]
population_2019_df=population_2019_df.sort_values('A1101',ascending=False)
print(population_2019_df.head(5))

move_in_df=ssdse_b_2022_df['A5101']
move_out_df=ssdse_b_2022_df['A5102']
#move_in_2019_df=ssdse_b_2022_df['A5101']
#move_out_2019_df=ssdse_b_2022_df['A5102']


