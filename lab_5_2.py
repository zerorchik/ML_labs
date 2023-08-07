# downloading data
from keras.datasets import cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
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
    plt.imshow(train_images[i])
    plt.xlabel(train_labels[i])
plt.show()

# truvumirnuy masuv --> dvovumirnuy + normalizuyemo [0, 1]
train_images = train_images.reshape((50000, 32*32*3))
train_images = train_images.astype('float32')/255
test_images = test_images.reshape((10000, 32*32*3))
test_images = test_images.astype('float32')/255
print('Pidhotovka danuh')
print('\n')

# zavntazhumo masuvu z vidhukamu na kozhne testove zobrazhennya
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
network.add(layers.Dense(512, activation='relu', input_shape=(32*32*3,)))
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
network.fit(train_images, train_labels, epochs=24, batch_size=512)
print('\n')

# perevirka modeli

print('Perevirka modeli')
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('\n')

# robota z modellyu

# klasufikuyemo zobrazhennya
import cv2
# 1
tst = cv2.imread('C:/Users/Akil0515/Desktop/KPI_Chan/ML_4/lab_5/w_auto_test.jpg')
tst = cv2.resize(tst, (32, 32))
tst = tst.reshape((1, 32*32*3))
tst = tst.astype('float32')/255
pred = list(network.predict(tst)[0])
print('Ochikuvane znachennya - 9')
print("Predicted - ", pred.index(max(pred)))
plt.imshow(cv2.cvtColor(tst.reshape((32, 32, 3)), cv2.COLOR_BGR2RGB))
plt.show()
# 2
tst = cv2.imread('C:/Users/Akil0515/Desktop/KPI_Chan/ML_4/lab_5/l_auto_test.jpg')
tst = cv2.resize(tst, (32, 32))
tst = tst.reshape((1, 32*32*3))
tst = tst.astype('float32')/255
pred = list(network.predict(tst)[0])
print('Ochikuvane znachennya - 1')
print("Predicted - ", pred.index(max(pred)))
plt.imshow(cv2.cvtColor(tst.reshape((32, 32, 3)), cv2.COLOR_BGR2RGB))
plt.show()
# 3
tst = cv2.imread('C:/Users/Akil0515/Desktop/KPI_Chan/ML_4/lab_5/airplane_test.jpg')
tst = cv2.resize(tst, (32, 32))
tst = tst.reshape((1, 32*32*3))
tst = tst.astype('float32')/255
pred = list(network.predict(tst)[0])
print('Ochikuvane znachennya - 0')
print("Predicted - ", pred.index(max(pred)))
plt.imshow(cv2.cvtColor(tst.reshape((32, 32, 3)), cv2.COLOR_BGR2RGB))
plt.show()