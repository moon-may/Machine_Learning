from tkinter import *
from tkinter.scrolledtext import ScrolledText
from tkinter import ttk
from tensorflow import keras
import pandas as pd
import numpy as np

# Загрузка модели
model = keras.models.load_model('model.keras')

# Логика работы приложения
def predict():
    # Получить данные из поля ввода (путь к файлу)
    file_path = data_entry.get()
    data = pd.read_csv(file_path, dtype={'id_student': str})

    id_list = data['id_student'] # получаем список "имен"
    X_new = data.drop(columns=['id_student']) # удаляем ненужную колонку

    # Предсказание, вероятность попадания в группу риска
    predicts = model.predict(X_new)
    # Представление класса 1 или 0
    pre_res = (predicts.flatten() > 0.5).astype(int)

    print(predicts)
    print(pre_res)

    total = len(pre_res)
    dropout = pre_res.sum()
    percent = dropout / total * 100

    result_label.config(text=f'Из {total} студентов риск отчисления у {dropout}.\n'
                             f'Группа риска составляет {percent:.2f} % от общего числа')

    result = pd.DataFrame({
        'id_student': id_list,
        'predicts': predicts.flatten(),
        'result': pre_res
    })

    risk_group = result[result['result'] == 1].sort_values('predicts', ascending=False)

    result_entry.delete('1.0', 'end')
    result_entry.insert('end', 'Студенты в группе риска:\n')

    for _, i in risk_group.iterrows():
        result_entry.insert('end', f"{i['id_student']}, {i['predicts']:.1%}\n")







# Интерфейс приложения

root = Tk()
# Заголовок окна
root.title('Прогнозирование отчисленных студентов')
# Фиксированные размеры окна
root.geometry('1000x500')
root.resizable(False, False)

# Главный фрейм для ввода данных и вывода основного результата
main_frame = Frame(root, width=600, height=500)
main_frame.pack(side='left', fill='both', expand=True)

# Дополнительный фрейм для списка студентов
result_frame = Frame(root, width=400, height=500)
result_frame.pack(side='right', fill='both', expand=True)

istr_label = ttk.Label(main_frame,
                   text='Загрузите данные для анализа',
                   font=('Arial', 14))
istr_label.pack(anchor='w', pady=10, padx=10)

data_entry = ttk.Entry(main_frame, font='20', width=50)
data_entry.pack(padx=10, pady=10)

pred_btn = ttk.Button(main_frame, text='Рассчитать',
                  command=predict)
pred_btn.pack()

# Переменные для вывода результата количеством и в процентах
total = 100     # общее количество
dropout = 20    # количество в группе риска
percent = dropout / total * 100 # процент

result_label = ttk.Label(main_frame, text='',
                         font=('Arial', 14))
result_label.pack(anchor='w', pady=10, padx=10)

result_entry = ScrolledText(result_frame, width=400, height=500)
result_entry.pack(anchor='w', pady=10, padx=10)

root.mainloop()
