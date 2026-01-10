from tkinter import *
from tkinter.scrolledtext import ScrolledText
from tkinter import ttk
from tkinter.messagebox import showinfo
from tensorflow import keras
import pandas as pd
from src.users import users


def load_model():
    '''
    Загрузка модели
    Автор: Стасевич Екатерина Алексеевна
    Тема ВКР: Прогнозирование количества отчисленных студентов с помощью методов машинного обучения
    '''
    # Возвращает модель
    return keras.models.load_model('model.keras')


# Интерфейс приложения
def open_app():
    # Сохраняет модель в переменную
    model = load_model()
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

        # Вывод результата
        result_label.config(text=f'Из {total} студентов риск отчисления у {dropout}.\n'
                                 f'Группа риска составляет {percent:.2f} % от общего числа')

        result = pd.DataFrame({
            'id_student': id_list,
            'predicts': predicts.flatten(),
            'result': pre_res
        })

        # Сортировка только группы риска по убыванию
        risk_group = result[result['result'] == 1].sort_values('predicts', ascending=False)

        # Вставка заголовка
        result_entry.delete('1.0', 'end')
        result_entry.insert('end', 'Студенты в группе риска:\n\n')

        # Итерированная вставка списка студентов
        for _, i in risk_group.iterrows():
            result_entry.insert('end', f"{i['id_student']}, вероятность отчисления: {i['predicts']:.1%}\n")

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

    # Подпись для окошка ввода имени файла
    istr_label = ttk.Label(main_frame,
                       text='Загрузите данные для анализа в формате .csv',
                       font=('Arial', 14))
    istr_label.pack(anchor='w', pady=10, padx=10)

    # Окно ввода имени файла
    data_entry = ttk.Entry(main_frame, font='20', width=50)
    data_entry.insert(0, 'datasets/new_data.csv')
    data_entry.pack(padx=10, pady=10)

    # Кпопка для расчета
    pred_btn = Button(main_frame, text='Рассчитать', font=20,
                      command=predict) # привязка к функции
    pred_btn.pack()

    # Блок для вывода результата
    result_label = ttk.Label(main_frame, text='',
                             font=('Arial', 14))
    result_label.pack(anchor='w', pady=10, padx=10)

    # Поле вывода списка студентов с результатами
    result_entry = ScrolledText(result_frame, width=400, height=500,
                                font=('Arial', 14))
    result_entry.pack(anchor='w', pady=10, padx=10)

    root.mainloop()
    return root     # возвращаем окно для дальнейшего использования в predict()


# Успешный вход
def login_success():
    login_window.destroy()
    open_app()
    print('Доступ разрешен')


def login_wrong():
    showinfo(title='Ошибка входа', message='Неверное имя пользователя или пароль')
    print('Доступ не разрешен')


'''
Проверка имени и пароля пользователя
Получает строки из полей ввода: 
    name - имя пользователя из поля username_entry;
    password - пароль из поля password_entry
'''
def user_check():
    # Получает данные из полей ввода:
    name = username_entry.get()
    password = password_entry.get()

    # Проверяет совпадение логина и пароля
    if users.get(name) == password:
        login_success()     # успешный вход
    else:
        login_wrong()   # ошибка входа

    print(name, password)

login_window = Tk()

# Фиксированные размеры окна
login_window.geometry('1000x500')
login_window.resizable(False, False)

login_window.title('Вход – Прогнозирование отчисленных студентов')

# Поле ввода имени пользователя
label_name = Label(login_window, text='Введите имя пользователя', font='20', width=50)
username_entry = Entry(login_window, font='20', width=50)
label_name.pack(padx=10, pady=10)
username_entry.pack(padx=10, pady=10)

# Поле ввода пароля
label_password = Label(login_window, text='Введите пароль', font='20', width=50)
password_entry = Entry(login_window, font='20', width=50, show='*') # закрывает символы *
label_password.pack(padx=10, pady=10)
password_entry.pack(padx=10, pady=10)

# Кнопка "Вход"
entered_btn = Button(login_window, text='Войти в систему', font='20', command=user_check)
entered_btn.pack(padx=10, pady=10)

login_window.mainloop()
