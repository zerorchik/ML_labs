import pandas as pd

# Зчитування першого CSV-файлу із заголовком
df_1 = pd.read_csv('dataset_1.csv', header=None)

# Зчитування другого CSV-файлу без заголовка
df_2 = pd.read_csv('dataset_2.csv', header=None, skiprows=1)

# Зчитування третього CSV-файлу без заголовка
df_3 = pd.read_csv('dataset_3.csv', header=None, skiprows=1)

# Об'єднання даних без повторення заголовків
df_combined = pd.concat([df_1, df_2, df_3], ignore_index=True)

# Збереження об'єднаного результату у новому CSV-файлі
df_combined.to_csv('train_dataset.csv', index=False)

# Зчитування об'єднаного результату без заголовка
df_combined = pd.read_csv('train_dataset.csv', skiprows=1)

# Видалення останнього стовпця
df_without_last_column = df_combined.iloc[:, :-1]

# Збереження нового об'єднаного результату у новому CSV-файлі
df_without_last_column.to_csv('train_dataset.csv', index=False)
df_without_last_column.to_excel('train_dataset.xlsx', index=False, engine='openpyxl')

# Вибір останнього стовпця
last_column = df_combined.iloc[:, -1]

# Збереження нового об'єднаного результату у новому CSV-файлі
last_column.to_csv('train_mark.csv', index=False)
last_column.to_excel('train_mark.xlsx', index=False, engine='openpyxl')