import matplotlib.pyplot as plt
import pandas as pd

# zavantazhutu nabir danuh z URL
titanic = pd.read_csv('https://vincentarelbundock.github.io/' + 'Rdatasets/csv/carData/TitanicSurvival.csv')

# prerhlyanutu ryadku
print('Pershi 5 ryadkiv naboru danuh')
print(titanic.head())
print('\n')
print('Ostanny 5 ryadkiv naboru danuh')
print(titanic.tail())
print('\n')

# nalashtuvatu nazvu stovptsiv
titanic.columns = ['name', 'survived', 'sex', 'age', 'class']
print('Pershi 5 ryadkiv z nalashtovanumu nazvamu stovptsiv')
print(titanic.head())
print('\n')

# analiz danuh
print('Naymolodshuy pasazhir')
print(titanic.min())
print('\n')
print('Naystarshuy pasazhir')
print(titanic.max())
print('\n')
print('Seredniy vik pasazhiriv')
print(titanic.mean(numeric_only=True))
print('\n')
print('Statustika po pasazhirah, shco vuzhulu')
print(titanic[titanic['survived'] == 'yes'].describe())
print('\n')
print('Vidsortovani zhinku 1 klasu')
print(titanic[(titanic['class'] == '1st') & (titanic['sex'] == 'female')].sort_values(by='age', ascending=False))
print('\n')
print('Naymolodsha')
print(titanic[(titanic['class'] == '1st') & (titanic['sex'] == 'female')].sort_values(by='age', ascending=True).head(1))
print('\n')
print('Naystarsha')
print(titanic[(titanic['class'] == '1st') & (titanic['sex'] == 'female')].sort_values(by='age', ascending=False).head(1))
print('\n')
print('Kilkist vuzhuvshuh')
print(titanic[(titanic['class'] == '1st') & (titanic['sex'] == 'female') & (titanic['survived'] == 'yes')].T.loc['survived'].count())
print('\n')

# histograma viku pasazhiriv
histogram = titanic.hist()
plt.show()