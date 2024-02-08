import pandas as pd

df = pd.read_csv(r'C:\Users\Akil0515\Desktop\KPI_Chan\ML_4\lab_1\non_cleared_data.csv', skiprows=4, skipfooter=5, usecols=[0,1,2,3,4,5,6,7,8,9], engine='python')

# ne ochushchenuy dataset
print('Non cleared dataset')
print(df)
print('\n')

# upuskayemo ryadku
df = df.drop(2, axis='index')
print('Dropped lines')
print(df)
print('\n')

# nalashtuvatu nazvu stovptsiv
df.columns = ['cities', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
print('Nalashtovani nazvau stovptsiv')
print(df)
print('\n')

# zminyuyemo indesu
print('Chu unicalni znachennya stovptsya \'cities\'?')
print(df['cities'].is_unique)
print('\n')
df = df.set_index('cities')
print('Dataset zi zminenumu indeksamu')
print(df)
print('\n')

# replace
print('Replaced NuNs')
df = df.replace('…', None)
print(df)
print('\n')