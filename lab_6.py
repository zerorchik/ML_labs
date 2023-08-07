import random
import pandas as pd

# Параметр кількості абітурієнтів
kilkist_abiturientiv = 1500

# Функція для перевірки унікальності ПІБ
def is_unique_name(pib, dataset):
    for abiturient in dataset:
        if pib == abiturient[0]:
            return False
    return True

# Функція для генерації ПІБ
def generate_name():
    first_names = ['Павло', 'Оксана', 'Дмитро', 'Ольга', 'Михайло', 'Анастасія', 'Ірина',
                   'Оксана', 'Наталія', 'Марія', 'Анна', 'Катерина', 'Вікторія', 'Олена',
                   'Тетяна', 'Людмила', 'Ярослава', 'Іванна', 'Ганна', 'Лідія', 'Ніна',
                   'Софія', 'Ірина', 'Ольга', 'Марта', 'Юлія', 'Іван', 'Олександр',
                   'Михайло', 'Андрій', 'Володимир', 'Віктор', 'Сергій', 'Петро',
                   'Дмитро', 'Олег', 'Максим', 'Роман', 'Ярослав', 'Ігор', 'Богдан',
                   'Тарас', 'Артем', 'Ілля', 'Олексій', 'Анатолій']
    last_names = ['Дмитришенко', 'Загвойко', 'Неборачко', 'Товтиш', 'Ковтун', 'Алібіба',
                  'Іваниш', 'Оксаниш', 'Наталіїш', 'Маріїш', 'Анниш', 'Катериниш',
                  'Вікторіїш', 'Олениш', 'Тетяниш', 'Людмилиш', 'Ярославиш', 'Іванниш',
                  'Ганниш', 'Лідіїш', 'Ніниш', 'Софіїш', 'Іриниш', 'Дмитришенко',
                  'Неборачко', 'Ковтун', 'Іваниш', 'Олександріш', 'Михайлиш', 'Андріїш',
                  'Володимириш', 'Вікторіш', 'Сергіїш', 'Петріш', 'Дмитріш', 'Олегіш',
                  'Максиміш', 'Романіш', 'Ярославіш', 'Ігоріш', 'Богданіш', 'Тарасіш',
                  'Артеміш', 'Ілляш', 'Олексіїш', 'Анатоліїш']
    return random.choice(first_names) + ' ' + random.choice(last_names)

# Генерація датасету з 1500 абітурієнтів
dataset = []
for _ in range(kilkist_abiturientiv):
    pib = generate_name()
    while not is_unique_name(pib, dataset):
        pib = generate_name()
    pilga = random.choice([True, False])
    matematyka = random.randint(100, 200)
    angl_mova = random.randint(100, 200)
    ukr_mova = random.randint(100, 200)
    rating = 0.4 * matematyka + 0.3 * angl_mova + 0.3 * ukr_mova
    decision = 'незараховано'
    dataset.append([pib, pilga, matematyka, angl_mova, ukr_mova, rating, decision])

# Створення DataFrame з датасету
columns = ['ПІБ', 'Пільги', 'Бал з математики', 'Бал з англійської', 'Бал з української', 'Рейтинг', 'Рішення']
df = pd.DataFrame(dataset, columns=columns)

# Сортування за рейтингом у порядку спадання
df.sort_values(by='Рейтинг', ascending=False, inplace=True)

# Визначення кількість абітурієнтів з пільгами, які можуть бути зараховані
kilkist_pilgovukiv = (df['Пільги'] == True).count()
if kilkist_pilgovukiv > 35: kilkist_pilgovukiv = 35
# Визначення загальної кількость абітурієнтів, які можуть бути зараховані
if kilkist_abiturientiv >= 350: kilkist_zagalna = 350 - kilkist_pilgovukiv
else: kilkist_zagalna = kilkist_abiturientiv - kilkist_pilgovukiv

# Відберемо до 35 пільговиків
pilgovyky_list = df[(df['Пільги'] == True) &
                    (df['Бал з математики'] >= 120) &
                    (df['Бал з англійської'] >= 120) &
                    (df['Бал з української'] >= 120) &
                    (df['Рейтинг'] >= 144)].head(kilkist_pilgovukiv)
# Відберемо з загального списку тих, хто може бути зарахований як безпільговик
general_list = df[(df['Бал з математики'] >= 140) &
                      (df['Рейтинг'] >= 160)]
# Виключимо абітурієнтів, які вже пройшли по пільгах
general_list = general_list[~general_list['ПІБ'].isin(pilgovyky_list['ПІБ'])]
# Відберемо решту з загального списку, щоб у сумі було до 350 вступивших
general_list = general_list.head(kilkist_zagalna)

# Об'єднання двох списків
combined_list = pd.concat([general_list, pilgovyky_list])

# Зміна рішення для всіх абітурієнтів з комбінованого списку
combined_list['Рішення'] = 'зараховано'

# Створення фінального датасету з мітками
# Виключимо абітурієнтів, які вже пройшли
ne_proysli = df[~df['ПІБ'].isin(combined_list['ПІБ'])]
total_list = pd.concat([combined_list, ne_proysli])
# Видалення останнього стовпця
df_without_last_column = total_list.iloc[:, :-1]

# Збереження набору даних
df_without_last_column.to_csv('test_dataset.csv', index=False)
df_without_last_column.to_excel('test_dataset.xlsx', index=False, engine='openpyxl')

# Вибір останнього стовпця
last_column = total_list.iloc[:, -1]

# Збереження міток
last_column.to_csv('test_mark.csv', index=False)
last_column.to_excel('test_mark.xlsx', index=False, engine='openpyxl')