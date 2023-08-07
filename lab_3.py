import pandas as pd

nyc = pd.read_csv(r'C:\Users\Akil0515\Desktop\KPI_Chan\ML_4\lab_3\1895-2018.csv', skiprows=4)

# zavantazhennya serednih temperatur v DataFrame
print('Serednya temperatura v New-York za 1895-2018 roky')

nyc.columns = ['Date', 'Temperature', 'Anoomaly']

nyc.Date = nyc.Date.floordiv(100)
print(nyc.head(3))
print('\n')

# rozbuttya danuh dlya navchannya ta testuvannya
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    nyc.Date.values.reshape(-1, 1), nyc.Temperature.values, random_state=11
)

# perevirka spivvidnoshennya 75% (train) do 25% (test)
print('Rozmir x_train:')
print(x_train.shape)
print('Rozmir x_test:')
print(x_test.shape)
print('\n')

# navchannya modeli
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X=x_train, y=y_train)
print('Navchannya na danuh x_train ta y_train')
print('\n')
print('Kut nahuly')
print(lr.coef_)
print('\n')
print('Tochka peretuny')
print(lr.intercept_)
print('\n')

# testuvannya modeli
predicted = lr.predict(x_test)
expected = y_test
print('Testuvannya modeli')

# prognosuvannya i ochikuvannya dlya kozhnoho 5 elementu
for p, e in zip(predicted[::5], expected[::5]):
    print(f'predicted: {p:2f}, expected: {e:2f}')
print('\n')

# prognosuvannya maybutnih i otsinka mynuluh temperatur
# y = mx + b
# m - coef_
# b - intercept_

predict = (lambda x: lr.coef_ * x + lr.intercept_)

print('Prognoz na 2019:')
print(predict(2019))
print('\n')
print('Otsinka znachennya za 1890:')
print(predict(1890))
print('\n')

# vizualizatsiya naboru danuh na regresiyniy pryamiy
import seaborn as sns

# data - DataSet
# x - osi x (Date)
# y - osi y (Temperature)
# hue - kolir tochok (Temperature)
# palette - kolirna karta Matplotlib
# legend - umovni poznachennya

plot_1 = sns.scatterplot(data=nyc, x='Date', y='Temperature',
                         hue='Temperature', palette='winter', legend=False)
#plt.show()

# zmina mashtabu
plot_1.set_ylim(1, 70)
#plt.show()

# koordunatu x pochatku ta kintsya tochok regresiynoi pryamoi
import numpy as np
x = np.array([min(nyc.Date.values), max(nyc.Date.values)])
print('Krayni roku')
print(x)
print('\n')

# prognosovani y
y = predict(x)
print('Temperatura v krayni roku')
print(y)
print('\n')

# diagrama rozkudy danuh na regresiyniy pryamiy
import matplotlib.pyplot as plt
line = plt.plot(x, y)
plt.show()