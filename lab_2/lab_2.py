import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

nyc = pd.read_csv(r'C:\Users\Akil0515\Desktop\KPI_Chan\ML_4\lab_2\1895-2018.csv', skiprows=4)

# zavantazhennya serednih temperatur v DataFrame
print('Serednya temperatura v New-York za 1895-2018 roky')
print(nyc.head())
print('\n')

# ochushennya danuh
nyc.columns = ['Date', 'Temperature', 'Anoomaly']
print('Pereymenuvannya stovtsiv')
print(nyc.head(3))
print('\n')

print('Perevirumo tup danuh stovtsya \'Date\'')
print(nyc.Date.dtype)
print('\n')
print('Vidsichemo 2 ostanni tsufru datu')
nyc.Date = nyc.Date.floordiv(100)
print(nyc.head(3))
print('\n')

print('Znaydemo osnovni statustuchni pokaznuku')
pd.set_option('display.precision', 2)
print(nyc.Temperature.describe())
print('\n')

# prognosuvannya maybutnih temperatur

# y = mx + b

# y - temperatura
# m - kut nahuly
# x - rik
# b - tochka peretuny

lr = stats.linregress(x=nyc.Date, y=nyc.Temperature)
print('Kut nahuly liniynoi regressii')
print(lr.slope)
print('\n')

print('Tochka peretuny regressiynoi pryamoi')
print(lr.intercept)
print('\n')

# graphik za vsi roku
gr0 = nyc.plot(x='Date', y='Temperature', style='.-')
plt.show()

# prognoz
print('2019:')
print(lr.slope * 2019 + lr.intercept)
print('\n')
print('2020:')
print(lr.slope * 2020 + lr.intercept)
print('\n')
print('2021:')
print(lr.slope * 2021 + lr.intercept)
print('\n')
print('2022:')
print(lr.slope * 2022 + lr.intercept)
print('\n')

# prognoz nazad
print('1894:')
print(lr.slope * 2019 + lr.intercept)
print('\n')
print('1893:')
print(lr.slope * 2020 + lr.intercept)
print('\n')
print('1892:')
print(lr.slope * 2021 + lr.intercept)
print('\n')
print('1891:')
print(lr.slope * 2022 + lr.intercept)
print('\n')
print('1890:')
print(lr.slope * 2022 + lr.intercept)
print('\n')

# regplot
sns.set_style('whitegrid')
gr1 = sns.regplot(x=nyc.Date, y=nyc.Temperature)
plt.show()

# mashtabuvannya
gr2 = sns.regplot(x=nyc.Date, y=nyc.Temperature)
gr2.set_ylim(1, 70)
plt.show()

# porivnynnya
full = pd.read_csv(r'C:\Users\Akil0515\Desktop\KPI_Chan\ML_4\lab_2\1895-2022.csv', skiprows=4)

print('Prognoz')
print(f'2019: {lr.slope * 2019 + lr.intercept}')
print(f'2020: {lr.slope * 2020 + lr.intercept}')
print(f'2021: {lr.slope * 2021 + lr.intercept}')
print(f'2022: {lr.slope * 2022 + lr.intercept}')
print('\n')

print('Real Data')
print(full.tail(4))