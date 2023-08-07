import matplotlib.pyplot as plt
import pandas as pd
import statistics as stat

data = pd.read_csv(r'C:\Users\Akil0515\Desktop\KPI_Chan\ML_4\lab_1\data.csv', usecols=[1,2,3,4,5,6,7,8,9])
df = data[26:27].values.tolist()[0]
print('Kyiv income')
print(df)
print('\n')

mat_spod = stat.mean(df)        # matematuchne spodivannya
print('Matematuchne spodivannya')
print(mat_spod)
print('\n')

median = stat.median(df)        # mediana
print('Mediana')
print(median)
print('\n')

mode = stat.mode(df)            # moda
print('Moda')
print(mode)
print('\n')

dispers = stat.pvariance(df)    # dispersiya
print('Dispersiya')
print(dispers)
print('\n')

ser_kvad = stat.pstdev(df)      # seredno kvadratuchne vidchulennya
print('Seredno kvadratuchne vidchulennya')
print(ser_kvad)
print('\n')

# roku
years = data.columns.values.tolist()
plt.figure(figsize=(18,5))

plt.subplot(131)
plt.bar(years, df)

plt.subplot(132)
plt.scatter(years, df)

plt.subplot(133)
plt.plot(years, df)

plt.suptitle('Dohodu v Kuivi za 2013-2021 roky')
plt.show()