import pandas as pd
import statistics as stat
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns

data = pd.read_csv(r'C:\Users\Akil0515\Desktop\KPI_Chan\ML_4\lab_1\data.csv', usecols=[1,2,3,4,5,6,7,8,9])
df = data[26:27].values.tolist()[0]

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