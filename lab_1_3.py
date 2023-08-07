import pandas as pd

data0 = pd.read_csv(r'C:\Users\Akil0515\Desktop\KPI_Chan\ML_4\lab_1\data.csv', usecols=[0,1,2,3,4,5,6,7,8,9])     # full dataframe
data = pd.read_csv(r'C:\Users\Akil0515\Desktop\KPI_Chan\ML_4\lab_1\data.csv', usecols=[1,2,3,4,5,6,7,8,9])
ukraine = data[26:27].values.tolist()[0]
cities = pd.read_csv(r'C:\Users\Akil0515\Desktop\KPI_Chan\ML_4\lab_1\data.csv', usecols=[0]).values.tolist()

# stvorennya Series z indeksamu za zamovchuvannyam
df = pd.Series(ukraine)
print('Series z indeksamu za zamovchuvannyam')
print(df)
print('\n')

# stvorennya Series z odnakovumy znachennyamu
df_odnak_znach = pd.Series(df[0], range(5))
print('Series z odnakovumy znachennyamu')
print(df_odnak_znach)
print('\n')

# obchuslennya opusovuh statustic dlya Series
print('Count of elements')
print(df.count())           # count()
print('\n')
print('Matematuchne spodivannya')
print(df.mean())            # mean()
print('\n')
print('Minimum element')
print(df.min())             # min()
print('\n')
print('Maximum element')
print(df.max())             # max()
print('\n')
print('Seredno kvadratuchne vidchulennya')
print(df.std())
print('\n')                 # std()
print('Describe')
print(df.describe())
print('\n')                 # describe()

# stvorennya Series z nestandartnumu indeksamu
df_extraordinary_indx = pd.Series(ukraine, index=[data.columns.values])
print('Series z nestandartnumu indeksamu')
print(df_extraordinary_indx)
print('\n')

# slovnuku yak inizializatoru
df_slovnuk = pd.Series(
    {data.columns.values[0]: df_extraordinary_indx['2013'], data.columns.values[1]: df_extraordinary_indx['2014'],
     data.columns.values[2]: df_extraordinary_indx['2015'], data.columns.values[3]: df_extraordinary_indx['2016'],
     data.columns.values[4]: df_extraordinary_indx['2017'], data.columns.values[5]: df_extraordinary_indx['2018']})
print('Slovnuku yak inizializatoru')
print(df_slovnuk)
print('\n')

# stvorennya Series z strkovumu elementamu
df_str = pd.Series([])
for i in range(len(cities)):
    df_str[i] = cities[i][0]
# df_str = pd.Series(data3)         - makes arrays
print('Series z strkovumu elementamu')
print(df_str)
print('\n')
print('Elementu scho mistyat \'C\'')
print(df_str.str.contains('C'))
print('\n')
print('Uppercase')
print(df_str.str.upper())
print('\n')

# stvorennya DataFrame na bazi slovnuka
df_dict = {cities[2][0]: data[2:3].values.tolist()[0], cities[12][0]: data[12:13].values.tolist()[0],
           cities[13][0]: data[13:14].values.tolist()[0], cities[14][0]: data[14:15].values.tolist()[0],}
# nalashtuvannya nestandartnuh indeksiv
df_slovnuk_2 = pd.DataFrame(df_dict, index=data.columns.values.tolist())
print('DataFrame na bazi slovnuka z nestandartnumu indeksamu')
print(df_slovnuk_2)
print('\n')

# zvernennya do stovptsiv DataFrame
print('Stovpets z indeksom \'Lviv\'')
print(df_slovnuk_2.Lviv)
print('\n')

# vukorustannya loc ta iloc
print('Vukorustannya loc na ryadku z indeksom \'2015\' ')
print(df_slovnuk_2.loc['2015'])
print('\n')
print('Vukorustannya loc na ryadkah z indeksamu \'2013-2015\' ')
print(df_slovnuk_2.loc['2013':'2015'])
print('\n')
print('Vukorustannya iloc na ryadku z indeksom \'2016\' ')
print(df_slovnuk_2.iloc[3])
print('\n')
print('Vukorustannya iloc na ryadkah z indeksamu \'2013, 2016\' ')
print(df_slovnuk_2.iloc[[0,3]])
print('\n')

# vubir pidmnozhun ryadkiv ta stovptsiv
print('Incomes Lviv, Mykolayiv za \'2020, 2021\' roku using loc')
print(df_slovnuk_2.loc['2020':'2021', 'Lviv':'Mykolayiv'])
print('\n')
print('Incomes Lviv, Vinnytsya za \'2013, 2021\' roku using iloc')
print(df_slovnuk_2.iloc[[0, 8], [2, 0]])
print('\n')

# lohichne indeksuvannya
print('Incomes >= 70000 ta <= 100000')
print(df_slovnuk_2[(df_slovnuk_2 >= 70000) & (df_slovnuk_2 <= 100000)])
print('\n')

# zvernennya po ryadku ta stovptsyu
print('Luhansk za 2019 using at')
print(df_slovnuk_2.at['2019', 'Luhansk'])
print('\n')
print('Previous Lviv za 2020 using iat')
print(df_slovnuk_2.iat[7, 2])
print('\n')
print('New Lviv za 2020 using iat')
df_slovnuk_2.iat[7, 2] += 100
print(df_slovnuk_2.iat[7, 2])
df_slovnuk_2.iat[7, 2] -= 100
print('\n')

# opusova statistica
print('Opusova statistica')
print(df_slovnuk_2.describe())
print('\n')
print('Zmina tochnosti')
pd.set_option('display.precision', 2)
print(df_slovnuk_2.describe())
print('\n')

# transponuvannya DataFrame using T
print('Transponuvannya DataFrame')
print(df_slovnuk_2.T)
print('\n')
# opusova statistica transponovanoho DataFrame
print('Opusova statistica transponovanoho DataFrame')
print(df_slovnuk_2.T.describe())
print('\n')

# sortuvannya ryadkiv za indeksamu
print('Sortuvannya ryadkiv za indeksamu - spadannya')
print(df_slovnuk_2.sort_index(ascending=False))
print('\n')

# sortuvannya stovptsiv za indeksamu
print('Sortuvannya stovptsiv za indeksamu - zrostannya')
print(df_slovnuk_2.sort_index(axis=1))
print('\n')

# sortuvannya stovptsiv za znachennyamu
print('Sortuvannya stovptsiv za znachennyamu po 2017 roku - spadannya')
print(df_slovnuk_2.sort_values(by='2017', axis=1, ascending=False))
print('\n')
print('Obednannya vuboru za 2020 rokom ta sortuvannya za stovptsyamu - spadannya')
print(df_slovnuk_2.loc['2020'].sort_values(ascending=False))
print('\n')

# kopiyuvannya ta sortuvannya na mistsi
print('Sortuvannya na mistsi')
df_slovnuk_2.sort_index(inplace=True, ascending=False)
print(df_slovnuk_2)
df_slovnuk_2.sort_index(inplace=True)
print('\n')