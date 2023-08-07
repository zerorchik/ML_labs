# pidhotovka i zavantazhennya danuh

# downloading data
from keras.datasets import fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print('Zavantazhennya danuh')
print('\n')

# pershi 25 zobrazhen i vidpovidni yim indeksu
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])
plt.show()

# truvumirnuy masuv --> dvovumirnuy + normalizuyemo [0, 1]
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255
print('Pidhotovka danuh')
print('\n')

# zavantazhumo masuvu z vidhukamu na kozhne testove zobrazhennya
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# architektura modeli

from keras import models
from keras import layers

# tup modeli - Sequential
network = models.Sequential()
# pruhovanuy shar - Dense (512 neyroniv)
# vuhidnuy shar (10 klasufikatsiynuh neyroniv)
network.add(layers.Dense(100, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))
print('Architektura modeli')
print('\n')

# kompilyatsiya merezhi

print('Kompilyatsiya merezhi')
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
print('\n')

# navchannya modeli

print('Navchannya modeli')
network.fit(train_images, train_labels, epochs=10, batch_size=32)
print('\n')

# perevirka modeli

print('Perevirka modeli')
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('\n')

# robota z modellyu

# klasufikuyemo zobrazhennya
import cv2
# 1
# pidhonka zobrazhennya
tst = cv2.imread('C:/Users/Akil0515/Desktop/KPI_Chan/ML_4/lab_5/tshirt_test.jpg', 0)
tst = cv2.resize(tst, (28, 28))
tst = tst.reshape((1, 28*28))
tst = tst.astype('float32') / 255
# resultat
pred = list(network.predict(tst)[0])
print('Ochikuvane znachennya - 0')
print("Predicted - ", pred.index(max(pred)))
# zobrazhennya
plt.imshow(tst.reshape((28, 28)), cmap='gray')
plt.show()
# 2
# pidhonka zobrazhennya
tst = cv2.imread('C:/Users/Akil0515/Desktop/KPI_Chan/ML_4/lab_5/jeans_test.jpg', 0)
tst = cv2.resize(tst, (28, 28))
tst = tst.reshape((1, 28*28))
tst = tst.astype('float32') / 255
# resultat
pred = list(network.predict(tst)[0])
print('Ochikuvane znachennya - 1')
print("Predicted - ", pred.index(max(pred)))
# zobrazhennya
plt.imshow(tst.reshape((28, 28)), cmap='gray')
plt.show()
# 3
# pidhonka zobrazhennya
tst = cv2.imread('C:/Users/Akil0515/Desktop/KPI_Chan/ML_4/lab_5/dress_test.jpg', 0)
tst = cv2.resize(tst, (28, 28))
tst = tst.reshape((1, 28*28))
tst = tst.astype('float32') / 255
# resultat
pred = list(network.predict(tst)[0])
print('Ochikuvane znachennya - 3')
print("Predicted - ", pred.index(max(pred)))
# zobrazhennya
plt.imshow(tst.reshape((28, 28)), cmap='gray')
plt.show()
# 4
# pidhonka zobrazhennya
tst = cv2.imread('C:/Users/Akil0515/Desktop/KPI_Chan/ML_4/lab_5/boots_test.jpg', 0)
tst = cv2.resize(tst, (28, 28))
tst = tst.reshape((1, 28*28))
tst = tst.astype('float32') / 255
# resultat
pred = list(network.predict(tst)[0])
print('Ochikuvane znachennya - 7')
print("Predicted - ", pred.index(max(pred)))
# zobrazhennya
plt.imshow(tst.reshape((28, 28)), cmap='gray')
plt.show()
# 5
# pidhonka zobrazhennya
tst = cv2.imread('C:/Users/Akil0515/Desktop/KPI_Chan/ML_4/lab_5/bag_test.jpg', 0)
tst = cv2.resize(tst, (28, 28))
tst = tst.reshape((1, 28*28))
tst = tst.astype('float32') / 255
# resultat
pred = list(network.predict(tst)[0])
print('Ochikuvane znachennya - 8')
print("Predicted - ", pred.index(max(pred)))
# zobrazhennya
plt.imshow(tst.reshape((28, 28)), cmap='gray')
plt.show()