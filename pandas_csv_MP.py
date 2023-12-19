
import pandas as pd
import os


print(os.getcwd())

df = pd.read_csv('./dyskE/MojePrg/_Python/BasicTests/data-03.csv', skipinitialspace=True)

#pd.read_csv()
df

df.columns.values.tolist()

df['Duration']
df['Date']
df['Pulse']
df['Calories']

df.dropna()

