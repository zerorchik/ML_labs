import pandas as pd
from sklearn.preprocessing import StandardScaler

# Завантаження даних

print('Zavantazhennya danuh')
print('\n')
x_train = pd.read_csv(r'train_dataset.csv')
y_train = pd.read_csv(r'train_mark.csv')
x_test = pd.read_csv(r'test_dataset.csv')
y_test = pd.read_csv(r'test_mark.csv')

# Первинна обробка даних

# Видалення стовпця з іменами
x_train = x_train.iloc[:, 1:]
x_test = x_test.iloc[:, 1:]
# Перетворення рядкових значень на числові
y_train = y_train.replace({'зараховано': 0, 'незараховано': 1}).to_numpy()
y_test = y_test.replace({'зараховано': 0, 'незараховано': 1}).to_numpy()
# Перетворення типу даних
x_train = x_train.astype('float32').to_numpy()
x_test = x_test.astype('float32').to_numpy()

# Нормалізація даних

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Архітектура моделі

from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
network = Sequential()
network.add(Conv1D(32, 2, activation='relu', input_shape=(x_train.shape[1], 1)))
network.add(MaxPooling1D(1))
network.add(Conv1D(64, 2, activation='relu'))
network.add(Flatten())
network.add(Dense(64, activation='relu'))
network.add(Dense(1, activation='sigmoid'))
print('Architektura modeli')
print('\n')

# Компіляція мережі

print('Kompilyatsiya merezhi')
network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print('\n')

# Навчання моделі

print('Navchannya modeli')
network.fit(x_train, y_train, epochs=10, batch_size=32)
print('\n')

'''# Перевірка моделі

print('Perevirka modeli')
test_loss, test_acc = network.evaluate(x_test, y_test)
print('\n')'''

# Перевірка моделі

print('Perevirka modeli')
predicted_probabilities = network.predict(x_test)
predicted_classes = (predicted_probabilities > 0.5).astype(int)
test_results = pd.DataFrame({'Predicted': predicted_classes.flatten(), 'Actual': y_test.flatten()})
test_loss, test_acc = network.evaluate(x_test, y_test)

# Збереження результатів у таблицю

results = pd.DataFrame({'Predicted': predicted_classes.flatten(), 'Actual': y_test.flatten()})
results.to_excel('model_5.xlsx', index=False, engine='openpyxl')