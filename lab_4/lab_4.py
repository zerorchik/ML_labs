from sklearn.datasets import load_digits

# zavantazhennya naboru danuh

digits = load_digits()
# opus datasetu
print(digits.DESCR)
print('\n')
# masuv target
print('Target kozhnoho 100 zrazka')
print(digits.target[::100])
print('\n')
# rozmir masuvu data
print('Kilkist zrazkiv ta oznak')
print(digits.data.shape)
print('\n')
# rozmir masuvu target
print('Kilkist tsilovih znachen')
print(digits.target.shape)
print('\n')

# vizualizatsiya danuh

# dvovumirnuy masuv images dlya zrazka z indeksom 13
print('Intensuvnist kozhnoho pikselya zobrazhennya zrazka u masuvi images (8*8) z indeksom 13')
print(digits.images[13])
print('\n')
# odnovumirnuy masuv data dlya zrazka z indeksom 13
print('Intensuvnist kozhnoho pikselya zobrazhennya zrazka u masuvi data (1*64) z indeksom 13')
print(digits.data[13])
print('\n')

# vuvedennya zobrazhennya zrazka z indeksom 13
import matplotlib.pyplot as plt

figure, axes = plt.subplots()
axes.imshow(digits.images[13], cmap=plt.cm.gray_r)
axes.set_xticks([])
axes.set_yticks([])
axes.set_title(digits.target[13])
plt.show()

# vuvedennya zobrazhen pershuh 24 zrazkiv datasetu
figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(6, 4))
for item in zip(axes.ravel(), digits.images, digits.target):
    axes, image, target = item
    axes.imshow(image, cmap=plt.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(target)
plt.tight_layout()
plt.show()

# vuvedennya zobrazhen pershuh 36 zrazkiv datasetu
figure, axes = plt.subplots(nrows=6, ncols=6, figsize=(6, 6))
for item in zip(axes.ravel(), digits.images, digits.target):
    axes, image, target = item
    axes.imshow(image, cmap=plt.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(target)
plt.tight_layout()
plt.show()

# rozbuttya danuh na navchalni ta testovi (75%, 25%)

from sklearn.model_selection import train_test_split

# dlya spivvidnoshennya train do test data 75% do 25% (standart)
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, random_state=11
)
# rozmir X_train
print('Kilkist trenuvalnuh zrazkiv ta oznak')
print(X_train.shape)
print('\n')
# rozmir X_test
print('Kilkist testovuh zrazkiv ta oznak')
print(X_test.shape)
print('\n')
# dlya spivvidnoshennya train do test data 80% do 20%
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, random_state=11, test_size=0.20
)

# stvorennya ta navchannya modeli

# stvorennya objektu otsinyvacha KNeighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
# metod fit zavantazhuye nabir testovuh danuh
print('Zavantazhennya naboru testovuh danuh') # k=5 (standart)
knn.fit(X=X_train, y=y_train)
print('\n')

# prognosuvannya klasiv

predicted = knn.predict(X=X_test)
expected = y_test

# porivnyannya prognosiv z ochikuvannyamu

print('Predicted pershi 20 zrazkiv')
print(predicted[:20])
print('Expected pershi 20 zrazkiv')
print(expected[:20])
print('\n')
print('Predicted pershi 24 zrazka')
print(predicted[:24])
print('Expected pershi 24 zrazka')
print(expected[:24])
print('\n')
print('Predicted pershi 36 zrazkiv')
print(predicted[:36])
print('Expected pershi 36 zrazkiv')
print(expected[:36])
print('\n')

# poyasnennya resultatu, zastosuvannya metruk tochnosti modeli

#score otsinyvach
print('Tochnist resultayiv')
print(f'{knn.score(X_test, y_test):.2%}')
print('\n')
# matrutsya nevidpovidnostey
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true=expected, y_pred=predicted)
print('Matrutsya nevidpovidnostey')
print(cm)
print('\n')

# zvit klasufikatsiyi

from sklearn.metrics import classification_report

names = [str(digit) for digit in digits.target_names]
print('Zvit klasufikatsiyi')
print(classification_report(expected, predicted, target_names=names))
print('\n')

# KNeighborsClassifier vs SVC vs GaussianNB

# SVC
from sklearn.svm import SVC

svm = SVC()
svm.fit(X=X_train, y=y_train)
print('Tochnist SVC')
print(f'{svm.score(X_test, y_test):.2%}')
print('\n')

# GaussianNB
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X=X_train, y=y_train)
print('Tochnist GaussianNB')
print(f'{gnb.score(X_test, y_test):.2%}')
print('\n')

# nalashtuvannya hiperparametru k

knn_3 = KNeighborsClassifier(n_neighbors=3) # k=3
knn_3.fit(X=X_train, y=y_train)
print('Tochnist K = 3')
print(f'{knn_3.score(X_test, y_test):.2%}')
print('\n')
knn_8 = KNeighborsClassifier(n_neighbors=8) # k=8
knn_8.fit(X=X_train, y=y_train)
print('Tochnist K = 8')
print(f'{knn_8.score(X_test, y_test):.2%}')
print('\n')
knn_2 = KNeighborsClassifier(n_neighbors=2) # k=2 - the best choise
knn_2.fit(X=X_train, y=y_train)
print('Tochnist K = 2')
print(f'{knn_2.score(X_test, y_test):.2%}')
print('\n')
knn_1 = KNeighborsClassifier(n_neighbors=1) # k=1
knn_1.fit(X=X_train, y=y_train)
print('Tochnist K = 1')
print(f'{knn_1.score(X_test, y_test):.2%}')
print('\n')
