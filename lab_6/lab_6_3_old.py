import pandas as pd
from keras.utils import to_categorical
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
# Перетворення міток на one-hot encoding
num_classes = 2
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Архітектура моделі

from keras import models
from keras import layers
# Тип моделі - Sequential
network = models.Sequential()
# Прихований шар - Dense (512 нейронів)
# Вихідний шар (2 класифікаційні нейрони)
network.add(layers.Dense(512, activation='relu'))
network.add(layers.Dense(2, activation='softmax'))
print('Architektura modeli')
print('\n')

# Компіляція мережі

print('Kompilyatsiya merezhi')
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
print('\n')

# Навчання моделі

print('Navchannya modeli')
network.fit(x_train, y_train, epochs=24, batch_size=512)
print('\n')

# Перевірка моделі

print('Perevirka modeli')
predicted_probabilities = network.predict(x_test)
predicted_classes = predicted_probabilities.argmax(axis=1)
test_loss, test_acc = network.evaluate(x_test, y_test)

# Збереження результатів у таблицю

results = pd.DataFrame({'Predicted': predicted_classes, 'Actual': y_test[:, 1]})
results.to_excel('model_1.xlsx', index=False, engine='openpyxl')