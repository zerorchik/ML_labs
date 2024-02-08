import pandas as pd

x_train = pd.read_csv(r'train_dataset.csv')
y_train = pd.read_csv(r'train_mark.csv')
x_test = pd.read_csv(r'test_dataset.csv')
y_test = pd.read_csv(r'test_mark.csv')

# Створення та навчання моделі


'''
KNeighborsClassifier
'''

from sklearn.neighbors import KNeighborsClassifier
# Створення об'єкту оцінювача KNeighbors
knn = KNeighborsClassifier()
# Вибираємо всі стовпці, починаючи з другого стовпця (ігноруємо перший стовпець - ПІБ)
x_train_subset = x_train.iloc[:, 1:]
# Транспонуємо масив у
y_train_reshaped = y_train["Рішення"].values.ravel()
# Метод fit завантажує набір тестових даних
knn.fit(X=x_train_subset, y=y_train_reshaped)
print('\n')

# Прогнозування класів
# Вибираємо всі стовпці, починаючи з другого стовпця (ігноруємо перший стовпець - ПІБ)
x_test_subset = x_test.iloc[:, 1:]
predicted = knn.predict(X=x_test_subset)
expected = y_test

# Пояснення результатів, застосування метрик точності моделі

# Score оцінювач
print('KNeighborsClassifier K = 5')
print(f'{knn.score(x_test_subset, y_test):.2%}')
print('\n')

# Налаштування гіперпараметру k

knn_3 = KNeighborsClassifier(n_neighbors=3) # k=3
knn_3.fit(X=x_train_subset, y=y_train_reshaped)
print('KNeighborsClassifier K = 3')
print(f'{knn_3.score(x_test_subset, y_test):.2%}')
print('\n')
knn_7 = KNeighborsClassifier(n_neighbors=7) # k=7
knn_7.fit(X=x_train_subset, y=y_train_reshaped)
print('KNeighborsClassifier K = 7')
print(f'{knn_7.score(x_test_subset, y_test):.2%}')
print('\n')
knn_6 = KNeighborsClassifier(n_neighbors=6) # k=6
knn_6.fit(X=x_train_subset, y=y_train_reshaped)
print('KNeighborsClassifier K = 6')
print(f'{knn_6.score(x_test_subset, y_test):.2%}')
print('\n')
knn_8 = KNeighborsClassifier(n_neighbors=8) # k=8
knn_8.fit(X=x_train_subset, y=y_train_reshaped)
print('KNeighborsClassifier K = 8')
print(f'{knn_8.score(x_test_subset, y_test):.2%}')
print('\n')


'''
SVC
'''

from sklearn.svm import SVC
# Створення об'єкту класифікатора SVC
svc = SVC()
# Вибір всіх стовпців, крім першого (ігноруємо стовпець з ПІБ)
x_train_subset = x_train.iloc[:, 1:]
# Трансформація вектора у одновимірний масив
y_train_reshaped = y_train.values.ravel()
# Навчання моделі на тренувальних даних
svc.fit(X=x_train_subset, y=y_train_reshaped)

# Прогнозування класів
# Вибір всіх стовпців, крім першого (ігноруємо стовпець з ПІБ)
x_test_subset = x_test.iloc[:, 1:]
# Прогнозування класів для тестових даних
predicted = svc.predict(X=x_test_subset)
expected = y_test

# Оцінка точності моделі
print('Точність SVC')
print(f'{svc.score(x_test_subset, y_test):.2%}')
print('\n')


'''
GaussianNB
'''

from sklearn.naive_bayes import GaussianNB
# Створення об'єкту класифікатора GaussianNB
gnb = GaussianNB()
# Вибір всіх стовпців, крім першого (ігноруємо стовпець з ПІБ)
x_train_subset = x_train.iloc[:, 1:]
# Трансформація вектора у одновимірний масив
y_train_reshaped = y_train.values.ravel()
# Навчання моделі на тренувальних даних
gnb.fit(X=x_train_subset, y=y_train_reshaped)

# Прогнозування класів
# Вибір всіх стовпців, крім першого (ігноруємо стовпець з ПІБ)
x_test_subset = x_test.iloc[:, 1:]
# Прогнозування класів для тестових даних
predicted = gnb.predict(X=x_test_subset)
expected = y_test

# Оцінка точності моделі
print('Точність GaussianNB')
print(f'{gnb.score(x_test_subset, y_test):.2%}')
print('\n')


'''
RandomForestClassifier
'''

from sklearn.ensemble import RandomForestClassifier
# Створення об'єкту класифікатора RandomForestClassifier
rfc = RandomForestClassifier()
# Вибір всіх стовпців, крім першого (ігноруємо стовпець з ПІБ)
x_train_subset = x_train.iloc[:, 1:]
# Трансформація вектора у одновимірний масив
y_train_reshaped = y_train.values.ravel()
# Навчання моделі на тренувальних даних
rfc.fit(X=x_train_subset, y=y_train_reshaped)

# Прогнозування класів
# Вибір всіх стовпців, крім першого (ігноруємо стовпець з ПІБ)
x_test_subset = x_test.iloc[:, 1:]
# Прогнозування класів для тестових даних
predicted = rfc.predict(X=x_test_subset)
expected = y_test

# Оцінка точності моделі
print('Точність RandomForestClassifier')
print(f'{rfc.score(x_test_subset, y_test):.2%}')
print('\n')


'''
MLPClassifier
'''

from sklearn.neural_network import MLPClassifier
# Створення та навчання моделі
mlp_classifier = MLPClassifier(max_iter=500, solver='adam')
# Вибір всіх стовпців, крім першого (ігноруємо стовпець з ПІБ)
x_train_subset = x_train.iloc[:, 1:]
# Трансформація вектора у одновимірний масив
y_train_reshaped = y_train.values.ravel()
# Навчання моделі на тренувальних даних
mlp_classifier.fit(X=x_train_subset, y=y_train_reshaped)

# Прогнозування класів
predicted = mlp_classifier.predict(X=x_test_subset)
expected = y_test

# Оцінка точності моделі
print('Точність MLPClassifier')
print(f'{mlp_classifier.score(x_test_subset, y_test):.2%}')
print('\n')